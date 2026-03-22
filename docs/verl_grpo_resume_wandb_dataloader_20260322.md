# 2026-03-22 记忆：verl GRPO 续训、WandB 接入与 dataloader state 排障

场景：

- 项目根目录：`/data/zyk_data/RL`
- 模型：`Qwen2.5-3B-Instruct`
- 任务：`GSM8K`
- 训练框架：`verl` + `GRPO`
- 旧实验：
  - `/data/zyk_data/RL/checkpoints/verl_grpo_gsm8k/qwen2_5_3b_lora_8x4090_20260321_182216`
- 旧最终 checkpoint：
  - `/data/zyk_data/RL/checkpoints/verl_grpo_gsm8k/qwen2_5_3b_lora_8x4090_20260321_182216/global_step_435`

---

## 1. WandB 这次为什么一开始没看见

要区分两层：

1. `verl` 基础配置默认支持 `["console","wandb"]`
2. 但实际是否启用，要看启动脚本有没有显式传 `trainer.logger`

本地脚本已经做过包装，后来调整成支持：

- `USE_WANDB=1`
- `WANDB_REQUIRE_LOGIN=1`
- `VAL_BEFORE_TRAIN=1`
- `LOG_VAL_GENERATIONS=8`
- `ROLLOUT_DATA_DIR=/some/path`

本地验证 WandB 是否可用，不要只看 `wandb status`。

更稳的命令：

```bash
source /home/zyk/miniconda3/etc/profile.d/conda.sh
conda activate verl
wandb login --verify
```

这次机器上验证成功，账号是：

- `2575293344-peking-university`

如果只想检查当前 env 里脚本能不能看到 key：

```bash
source /home/zyk/miniconda3/etc/profile.d/conda.sh
conda run -n verl python -c "import wandb; print(bool(getattr(getattr(wandb, 'api', None), 'api_key', None)))"
```

---

## 2. 第一次 resume 失败的根因

第一次尝试 resume 时使用了：

- `DATALOADER_NUM_WORKERS=0`
- 但旧 checkpoint 是按多进程 dataloader 保存的

实际报错是：

```text
AssertionError: State doesn't contain key '_num_yielded' expected for single process dataloader
```

这说明：

- 旧 checkpoint 里的 `data.pt` 对应的是多进程 dataloader 状态
- 现在恢复时换成了单进程 dataloader
- `torchdata` 的 state schema 不一致，直接炸

结论：

- 如果 **恢复 dataloader state**
- 那么 `DATALOADER_NUM_WORKERS` 必须和旧 checkpoint 保存时一致

---

## 3. 第二次 resume 还是不对的根因

第二次尝试把：

- `DATALOADER_NUM_WORKERS` 临时改回 `8`

这样确实绕过了 `_num_yielded` 的 schema mismatch。

但是仍然没有真正继续训练，只完成了：

- checkpoint 恢复
- `val_before_train`
- wandb 初始化和初始 validation 上报

随后立即退出，并在退出时出现：

```text
Exception ignored in atexit callback ...
BrokenPipeError: [Errno 32] Broken pipe
```

这个 `BrokenPipeError` 不是主因，只是 WandB 在主进程退出后的清理噪声。

真正原因是：

- `verl` 恢复 checkpoint 时不仅恢复模型/优化器，还会恢复 `data.pt`
- 而 `global_step_435` 本身是旧 run 的最终 checkpoint
- 它对应的 train dataloader state 基本已经耗尽
- 所以 resume 后没有新的 batch 可训练
- 主流程很快正常退出
- WandB 退出时才又报一个 BrokenPipe

关键代码路径：

- `/data/zyk_data/RL/verl/verl/trainer/ppo/ray_trainer.py`

原逻辑：

```python
dataloader_local_path = os.path.join(global_step_folder, "data.pt")
if os.path.exists(dataloader_local_path):
    dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
    self.train_dataloader.load_state_dict(dataloader_state_dict)
```

所以：

- 从“最终 checkpoint”续训
- 如果恢复了 dataloader state
- 很容易直接拿到一个“已经读完”的 dataloader

---

## 4. 最终确定的修法

### 核心原则

从最终 checkpoint 继续下一轮训练时：

- 要恢复模型/优化器
- 但 **不要恢复 train dataloader state**

### 最终方案

新增一个开关：

- `trainer.restore_dataloader_state`

默认值：

- `True`

在“从最终 checkpoint 继续下一轮”这种场景中，显式设：

- `False`

这样 dataloader 会从头开始迭代，不再被 `data.pt` 卡死。

---

## 5. 这次实际改了哪些代码

### 5.1 配置新增字段

文件：

- `/data/zyk_data/RL/verl/verl/trainer/config/ppo_trainer.yaml`

新增：

```yaml
restore_dataloader_state: True
```

含义：

- `True`：resume 时加载 checkpoint 里的 `data.pt`
- `False`：resume 时跳过 dataloader state 恢复

### 5.2 trainer 恢复逻辑加开关

文件：

- `/data/zyk_data/RL/verl/verl/trainer/ppo/ray_trainer.py`

新增行为：

1. 读取：
   - `self.config.trainer.get("restore_dataloader_state", True)`
2. 如果为 `True`：
   - 保持旧行为，加载 `data.pt`
3. 如果为 `False`：
   - 打印：
     - `Skipping dataloader state restore; train dataloader will start from the beginning.`

### 5.3 启动脚本暴露环境变量

文件：

- `/data/zyk_data/RL/run_qwen2_5_3b_gsm8k_grpo_8x4090.sh`

新增环境变量：

- `RESTORE_DATALOADER_STATE="${RESTORE_DATALOADER_STATE:-1}"`

启动时显式传：

- `trainer.restore_dataloader_state="${RESTORE_DATALOADER_STATE}"`

日志里也会打印：

- `[run] restore_dl     : ...`

---

## 6. 这次 resume 的最终推荐参数

如果目标是：

- 从 `global_step_435` 继续
- 再训练 1 个 epoch
- 开 WandB 看曲线
- 避免 dataloader state 问题

推荐组合是：

```bash
DATALOADER_NUM_WORKERS=0
RESTORE_DATALOADER_STATE=0
USE_WANDB=1
WANDB_REQUIRE_LOGIN=1
VAL_BEFORE_TRAIN=1
LOG_VAL_GENERATIONS=8
```

注意：

- **旧逻辑下**，为了兼容 checkpoint 里的 dataloader state，需要 `DATALOADER_NUM_WORKERS=8`
- **新开关启用后**，因为不再恢复 `data.pt`，所以 `DATALOADER_NUM_WORKERS=0` 完全可以用
- 反而更稳，也更符合之前为 clean exit 做的 dataloader 修复方向

---

## 7. “再训练 1 个 epoch” 的 step 算法

旧 checkpoint：

- `global_step_435`

当前 GSM8K 配置下：

- `Size of train dataloader: 29`

如果想再训练 1 个 epoch：

```text
新的 total_training_steps = 435 + 29 - 1 = 463
```

所以应传：

```bash
trainer.total_training_steps=463
```

原因是这版 `verl` 的 `is_last_step` 判断发生在 step 开始前。

---

## 8. 这次正式启动的最终命令

最终采用的 tmux 启动方式：

```bash
tmux new-window -t grpo -n resume463skipdl -c /data/zyk_data/RL \
  "DATALOADER_NUM_WORKERS=0 RESTORE_DATALOADER_STATE=0 USE_WANDB=1 WANDB_REQUIRE_LOGIN=1 VAL_BEFORE_TRAIN=1 LOG_VAL_GENERATIONS=8 ROLLOUT_DATA_DIR=/data/zyk_data/RL/rollout_debug/qwen2_5_3b_lora_8x4090_resume_skipdl_20260322_173251 EXPERIMENT_NAME=qwen2_5_3b_lora_8x4090_resume_skipdl_20260322_173251 TOTAL_EPOCHS=1 bash /data/zyk_data/RL/run_qwen2_5_3b_gsm8k_grpo_8x4090.sh trainer.resume_mode=resume_path trainer.resume_from_path=/data/zyk_data/RL/checkpoints/verl_grpo_gsm8k/qwen2_5_3b_lora_8x4090_20260321_182216/global_step_435 trainer.total_training_steps=463"
```

这条命令对应的新实验：

- `qwen2_5_3b_lora_8x4090_resume_skipdl_20260322_173251`

对应路径：

- 日志：
  - `/data/zyk_data/RL/logs/qwen2_5_3b_lora_8x4090_resume_skipdl_20260322_173251.log`
- checkpoint：
  - `/data/zyk_data/RL/checkpoints/verl_grpo_gsm8k/qwen2_5_3b_lora_8x4090_resume_skipdl_20260322_173251`
- rollout dump：
  - `/data/zyk_data/RL/rollout_debug/qwen2_5_3b_lora_8x4090_resume_skipdl_20260322_173251`

---

## 9. 下次再遇到类似问题时的判断顺序

### 情况 A：resume 后直接报 `_num_yielded`

优先判断：

- 是否一边恢复 `data.pt`
- 一边把 `DATALOADER_NUM_WORKERS` 改掉了

修法：

- 要么恢复时保持和旧 checkpoint 一样的 worker 数
- 要么直接 `RESTORE_DATALOADER_STATE=0`

### 情况 B：resume 后只做了 validation 就退出

优先判断：

- 这是不是从最终 checkpoint 恢复
- checkpoint 里是否包含 `data.pt`

修法：

- 跳过 dataloader state restore

### 情况 C：只看到 WandB 的 `BrokenPipeError`

不要先把它当主因。

先查：

1. 前面有没有真正的主错误
2. 是否只是主流程先退出了
3. WandB 只是退出清理时报噪声

---

## 10. 一句话结论

从 `global_step_435` 这种“最终 checkpoint”继续训练，最稳的方案不是硬兼容旧 dataloader state，而是：

```text
恢复模型/优化器，跳过 data.pt，重新从头迭代训练数据。
```
