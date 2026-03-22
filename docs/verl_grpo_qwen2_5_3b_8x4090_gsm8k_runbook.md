# verl GRPO 实战 Runbook

适用场景：在单机 `8 x RTX 4090 24GB` 上，用本地 `Qwen2.5-3B-Instruct` 复现 `verl` 的 `GRPO`，数据集为 `GSM8K`。

本文不是泛泛教程，而是基于本次真实跑通过程整理的故障复盘和排障手册。目标是让后面的人遇到类似报错时，能直接按关键字定位并修掉。

## 1. 本次实验的最终落地状态

- 时间：`2026-03-21`
- 机器：`8 x NVIDIA GeForce RTX 4090 24GB`
- 模型路径：`/data/zyk_data/models/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1`
- 数据路径：
  - `train`: `/data/zyk_data/RL/data/gsm8k/train.parquet`
  - `test`: `/data/zyk_data/RL/data/gsm8k/test.parquet`
- 启动脚本：`/data/zyk_data/RL/run_qwen2_5_3b_gsm8k_grpo_8x4090.sh`
- 数据准备脚本：`/data/zyk_data/RL/prepare_gsm8k_dataset.sh`

本次真正跑通时的核心训练参数：

- `algorithm.adv_estimator=grpo`
- `data.train_batch_size=256`
- `rollout.n=5`
- `rollout.tensor_model_parallel_size=2`
- `actor.ppo_mini_batch_size=128`
- `actor.ppo_micro_batch_size_per_gpu=16`
- `rollout.log_prob_micro_batch_size_per_gpu=16`
- `max_prompt_length=512`
- `max_response_length=1024`
- `trainer.total_epochs=15`

对应的批量关系：

- 每个外层 GRPO step 的 prompt 数：`256`
- 每个 prompt 采样 `5` 条 response
- 每个外层 step 实际生成的 response 总数：`256 * 5 = 1280`
- 当前日志显示总训练步数：`435`

当前环境里的关键版本组合：

- `python 3.12.13`
- `torch 2.8.0+cu128`
- `transformers 5.3.0`
- `vllm 0.11.0`
- `ray 2.54.0`
- `numpy 2.4.3`
- `numba 0.61.2`

这组版本组合本身就是问题来源之一，后文会详细解释。

## 2. verl 这次是怎么跑起来的

本次执行链路不是单进程训练，而是下面这条链：

`launcher shell -> conda run -n verl python -m verl.trainer.main_ppo -> Ray TaskRunner -> verl worker group -> FSDP actor/ref + vLLM rollout`

这会带来两个很重要的后果：

1. 不是所有日志都会回到你启动脚本所在的终端。
2. 不是所有 Python 子进程都会自动吃到你在当前 shell 里加的补丁。

模型在 8 张卡上的放置方式也不是“每张卡一整份模型”：

- 训练侧 `actor/ref` 使用 `FSDP`，本质上是参数分片。
- rollout 侧 `vLLM` 使用 `tensor_model_parallel_size=2`，也就是 2 卡一组做张量并行。
- 因此更准确的理解是：这 8 张卡承载的是训练分片和推理分片，而不是每张卡都单独持有完整模型副本。

## 3. 成功运行时应当看到什么

很多人会把“主日志没刷”误认为“没开始训练”。这在 `verl + Ray + vLLM + FSDP` 组合里非常常见。

本次成功运行时，真正有效的信号是：

- Ray worker 日志里出现：
  - `dataset len: 7473`
  - `filter dataset len: 7473`
  - `dataset len: 1319`
  - `Size of train dataloader: 29, Size of val dataloader: 1`
  - `Total training steps: 435`
  - `Training Progress: 1/435`
  - `step:1 ...`
- 首个 step 耗时大约 `137.55s`
- 8 张卡都有显存占用和波动中的 GPU util，而不是只有 1 张卡动

这说明：

- 数据已经加载完，不是卡在 data loading
- Ray actor 已经拉起
- vLLM 和 FSDP 初始化完成
- rollout / old log prob / ref / actor update 都已经实际执行

## 4. 日志到底去哪了

### 4.1 启动脚本日志

启动脚本自己的日志文件在：

- `/data/zyk_data/RL/logs/qwen2_5_3b_lora_8x4090_<timestamp>.log`

但是这个文件不会完整包含 Ray worker 的输出，所以经常看起来“很安静”。

### 4.2 真正最有信息量的日志

真正要盯的是 Ray session 日志目录：

- `/tmp/ray/session_latest/logs/`

本次成功运行的核心 TaskRunner 日志是：

- stdout:
  - `/tmp/ray/session_latest/logs/worker-ad7266b17f020a7230d291b7351b7736b6ad68e146619a10a15054eb-01000000-1961780.out`
- stderr:
  - `/tmp/ray/session_latest/logs/worker-ad7266b17f020a7230d291b7351b7736b6ad68e146619a10a15054eb-01000000-1961780.err`

其中：

- `stdout` 里通常有 `dataset len`、`step:1`、吞吐和各类训练指标
- `stderr` 里通常有 `Training Progress` 进度条、warning 和部分异常栈

### 4.3 推荐的盯日志方式

直接 tail：

```bash
tail -f /tmp/ray/session_latest/logs/worker-...out
tail -f /tmp/ray/session_latest/logs/worker-...err
```

如果想把 Ray 的输出重新汇总到一个固定文件：

```bash
stdbuf -oL -eL tail -F \
  /tmp/ray/session_latest/logs/worker-...out \
  /tmp/ray/session_latest/logs/worker-...err \
  | tee -a /data/zyk_data/RL/logs/<experiment>.ray.log
```

本次已经汇总出的文件是：

- `/data/zyk_data/RL/logs/qwen2_5_3b_lora_8x4090_20260321_182216.ray.log`

## 5. 这次真正踩到的坑

下面按时间顺序整理。

### 5.1 数据准备阶段的代理环境变量问题

#### 现象

这类问题通常出现在拉取或预处理 `GSM8K` 时，表现为 `datasets` / `huggingface_hub` 访问异常，尤其是机器上残留了 `ALL_PROXY`、`HTTP_PROXY`、`HTTPS_PROXY` 之类环境变量时。

本次虽然没有把报错保存在项目日志里，但实际处理时已经确认需要在脚本里主动清理代理环境变量，否则后续很容易再次踩坑。

#### 根因

`conda run` 拉起的新 Python 进程会继承当前 shell 的环境变量。如果机器上挂着不可用的 SOCKS/HTTP 代理，数据预处理和部分模型元信息访问都可能被代理劫持。

#### 排查方法

先查环境变量：

```bash
env | grep -i proxy
```

如果看到以下变量，就要提高警惕：

- `ALL_PROXY`
- `all_proxy`
- `HTTP_PROXY`
- `http_proxy`
- `HTTPS_PROXY`
- `https_proxy`

#### 解决方案

在数据准备脚本和训练启动脚本里都显式清理代理变量：

```bash
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
```

本次已经落在：

- `/data/zyk_data/RL/prepare_gsm8k_dataset.sh`
- `/data/zyk_data/RL/run_qwen2_5_3b_gsm8k_grpo_8x4090.sh`

#### 原理说明

这不是 `verl` 本身的 bug，而是实验机环境污染导致的启动不稳定。对实验室公用机器来说，这一步应该视为训练前置清理动作。

### 5.2 batch 参数组合的隐藏约束

#### 现象

用户希望跑的命令是：

```bash
TRAIN_BATCH_SIZE=256 PPO_MICRO_BATCH_SIZE_PER_GPU=16 LOGPROB_MICRO_BATCH_SIZE_PER_GPU=16 \
  bash /data/zyk_data/RL/run_qwen2_5_3b_gsm8k_grpo_8x4090.sh
```

如果脚本默认的 `PPO_MINI_BATCH_SIZE` 设置不对，会在真正训练前就因为 batch 形状不合法而失败，或者在 `verl` 内部出现不直观的 shape/divisibility 错误。

#### 根因

`GRPO` 下真正参与 PPO 切分的不是 prompt batch，而是 rollout 展开后的 sample batch。

必须同时满足：

```text
real_train_batch = TRAIN_BATCH_SIZE * ROLLOUT_N
real_train_batch % N_GPUS == 0

PPO_MINI_BATCH_SIZE * ROLLOUT_N % N_GPUS == 0

normalized_ppo_mini_batch = PPO_MINI_BATCH_SIZE * ROLLOUT_N / N_GPUS
normalized_ppo_mini_batch % PPO_MICRO_BATCH_SIZE_PER_GPU == 0
```

本次参数代入后：

- `TRAIN_BATCH_SIZE = 256`
- `ROLLOUT_N = 5`
- `N_GPUS = 8`
- `PPO_MINI_BATCH_SIZE = 128`
- `PPO_MICRO_BATCH_SIZE_PER_GPU = 16`

得到：

- `real_train_batch = 256 * 5 = 1280`
- `1280 % 8 = 0`
- `normalized_ppo_mini_batch = 128 * 5 / 8 = 80`
- `80 % 16 = 0`

因此这组配置合法。

如果还用旧默认值 `PPO_MINI_BATCH_SIZE=64`，则：

- `64 * 5 / 8 = 40`
- `40 % 16 != 0`

这就是为什么这次脚本里把默认值改成了 `128`。

#### 解决方案

不要把 batch 关系留给运行时碰运气，直接在 launcher 层做前置校验并 `exit 1`。

本次已经在 `/data/zyk_data/RL/run_qwen2_5_3b_gsm8k_grpo_8x4090.sh` 里把这些检查显式写出来。

#### 原理说明

这个问题不属于库兼容 bug，但属于最容易让实验反复重跑的参数层错误。对于实验室共享脚本，应该优先在 shell 层做硬约束检查。

### 5.3 `transformers 5.3.0` 缺少 `AutoModelForVision2Seq`

#### 典型报错

首轮失败日志里直接报：

```text
ImportError: cannot import name 'AutoModelForVision2Seq' from 'transformers'
```

对应日志文件：

- `/data/zyk_data/RL/logs/qwen2_5_3b_lora_8x4090_20260321_180454.log`

#### 根因

本机环境是：

- `transformers 5.3.0`
- `verl` 当前源码里直接静态导入 `AutoModelForVision2Seq`

但在这个版本组合下，`transformers` 已经没有这个符号，或者符号路径和旧版本不兼容。`verl` 这里默认假设该类存在，导致在 `main_ppo` 还没真正启动训练前就 import 失败。

#### 排查方法

看主日志最开始的 import stack 即可，通常会落在：

- `verl/utils/model.py`
- `verl/workers/fsdp_workers.py`

快速验证：

```bash
source /home/zyk/miniconda3/etc/profile.d/conda.sh
conda run -n verl python -c "from transformers import AutoModelForVision2Seq"
```

如果这里直接炸，说明不是业务逻辑问题，就是 API 断裂。

#### 解决方案

在 `verl` 源码中加入兼容导入逻辑：

- 优先尝试 `AutoModelForVision2Seq`
- 不存在时回退到 `AutoModelForImageTextToText`
- 后续使用时先判空，再访问 `_model_mapping`

本次修改文件：

- `/data/zyk_data/RL/verl/verl/utils/model.py`
- `/data/zyk_data/RL/verl/verl/workers/fsdp_workers.py`

#### 原理说明

这里不是说我们要训练视觉模型，而是 `verl` 的通用模型加载逻辑在 import 阶段就把视觉相关类也拉进来了。即使实际任务是纯文本，只要 import 链上有不兼容符号，也会直接启动失败。

#### 长期建议

更干净的做法有两个：

1. 统一 pin 到与 `verl` 当前提交兼容的 `transformers` 版本。
2. 把这类兼容导入补丁提到实验室维护的 `verl` 分支里，不要靠每次手改。

### 5.4 `Qwen2Tokenizer` 缺少 `all_special_tokens_extended`

#### 典型报错

第二轮失败时，Ray worker 报：

```text
AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
```

对应日志文件：

- `/data/zyk_data/RL/logs/qwen2_5_3b_lora_8x4090_20260321_180738.log`

关键调用链是：

`vllm -> transformers_utils.tokenizer.get_cached_tokenizer -> tokenizer.all_special_tokens_extended`

#### 根因

`vllm 0.11.0` 在 tokenizer 缓存逻辑中假设 tokenizer 对象有：

- `all_special_tokens_extended`
- `special_tokens_map_extended`

但当前这套 `Qwen2Tokenizer + transformers 5.3.0` 组合没有这个属性，于是 vLLM 初始化 tokenizer 时直接炸掉。

#### 排查方法

看到错误关键字：

- `all_special_tokens_extended`
- `get_cached_tokenizer`
- `Qwen2Tokenizer`

基本就能判定不是模型权重问题，而是 `vllm` 对 tokenizer API 的假设和当前 `transformers` 不一致。

也可以直接本地验证：

```bash
source /home/zyk/miniconda3/etc/profile.d/conda.sh
conda run -n verl python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('/data/zyk_data/models/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1'); print(hasattr(type(t), 'all_special_tokens_extended')); print(hasattr(type(t), 'special_tokens_map_extended'))"
```

#### 解决方案

用 `sitecustomize.py` 在 Python 启动时给 `PreTrainedTokenizerBase` 动态补两个 property：

- `all_special_tokens_extended`
- `special_tokens_map_extended`

本次先在 repo 内添加：

- `/data/zyk_data/RL/verl/sitecustomize.py`

但这还不够。

#### 为什么 repo 里的 `sitecustomize.py` 不够

因为 `verl` 是通过 Ray 拉 worker，再由 worker 拉 vLLM 子进程。只在当前 shell 里设置 `PYTHONPATH=/data/zyk_data/RL/verl`，并不能保证所有子进程都在足够早的时刻导入到 repo 里的 `sitecustomize.py`。

所以本次最终把同样的补丁放到了环境级：

- `/home/zyk/miniconda3/envs/verl/lib/python3.12/site-packages/sitecustomize.py`

这样任何用该环境启动的 Python 进程都会自动加载补丁，包括：

- 主 `main_ppo` 进程
- Ray worker
- vLLM 相关子进程

#### 原理说明

`sitecustomize.py` 是 Python 标准启动钩子。只要文件位于解释器可搜索到的位置，Python 在启动阶段就会自动 import 它。这类跨进程兼容补丁，放在 env 的 `site-packages` 比放在业务 repo 更稳。

#### 长期建议

最好的长期方案仍然是版本 pin，而不是永久依赖 monkey patch。但在实验要尽快跑通时，这种补丁是成本最低、可验证性最高的方案。

### 5.5 `vllm.v1.spec_decode.ngram_proposer` 触发 `numba` / `NumPy` 不兼容

#### 典型报错

第三轮失败时，Ray worker 报：

```text
ImportError: Numba needs NumPy 2.2 or less. Got NumPy 2.4.
```

调用链是：

```text
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from numba import ...
```

对应日志文件：

- `/data/zyk_data/RL/logs/qwen2_5_3b_lora_8x4090_20260321_181519.log`

#### 根因

当前环境实际版本是：

- `numpy 2.4.3`
- `numba 0.61.2`

而这个 `numba` 版本要求：

- `numpy <= 2.2`

`vllm 0.11.0` 在 import `gpu_model_runner` 时会间接 import 到 `spec_decode/ngram_proposer.py`。即便你这次并没有显式使用 speculative decoding，这条 import 链也照样会发生，于是直接在导入阶段崩掉。

#### 排查方法

看到下面几个关键字的组合就能快速定位：

- `ngram_proposer`
- `numba`
- `NumPy 2.2 or less`

可以本地复现：

```bash
source /home/zyk/miniconda3/etc/profile.d/conda.sh
conda run -n verl python -c "import vllm.v1.spec_decode.ngram_proposer"
```

如果这里直接报 `Numba needs NumPy 2.2 or less`，说明这不是训练脚本本身的问题，而是底层依赖树炸了。

#### 本次为什么没有直接降级 NumPy

理论上更干净的做法是统一重建环境，把 `numpy` pin 到 `<=2.2`。但这在实验中途会带来额外风险：

- 可能牵连别的依赖重新解算
- 可能影响当前已安装的 `torch/vllm/transformers`
- 可能需要重新验证整套环境

本次目标是“先把实验跑起来”，所以采用了更局部的兼容方案。

#### 解决方案

在 `sitecustomize.py` 里提前拦截：

- 如果 `numba` 可以正常 import，则不做事
- 如果 `numba` 因 `numpy` 版本不兼容而 import 失败，则注入一个假的 `vllm.v1.spec_decode.ngram_proposer` 模块
- 在假模块里只放一个会抛错的 `NgramProposer`

本次补丁落在：

- `/data/zyk_data/RL/verl/sitecustomize.py`
- `/home/zyk/miniconda3/envs/verl/lib/python3.12/site-packages/sitecustomize.py`

#### 为什么这个补丁能跑通

因为这次训练并没有真正走 speculative decoding 路径。我们要做的是避免 `vllm` 在 import 阶段就因为可选组件不兼容而整个进程起不来。

换句话说，这里是把“导入时硬依赖”降级成“真正使用时才报错”。只要当前训练路径不使用 `NgramProposer`，训练就能正常继续。

#### 长期建议

长期还是建议二选一：

1. 统一 pin `numpy<=2.2`
2. 升级到一套官方确认兼容的 `vllm + numba + numpy` 版本组合

如果实验室要复用这个环境很多次，优先做环境治理，不要永久依赖假模块补丁。

### 5.6 “主日志没输出”并不等于没开始训练

#### 现象

启动脚本日志文件几乎不刷，但 GPU 有显存占用，用户容易误判成：

- 卡在数据加载
- 卡在 Ray 初始化
- 根本没开始训练

#### 根因

Ray worker 的 stdout/stderr 默认不会完整回流到启动脚本里 `tee` 的那个主日志文件。结果是：

- 训练主指标在 Ray worker 的 `.out`
- 进度条和 warning 在 Ray worker 的 `.err`
- 启动脚本自己的 `.log` 反而信息最少

#### 排查方法

不要先盯主日志，先看：

```bash
ls -1 /tmp/ray/session_latest/logs
```

然后用关键字搜活跃日志：

```bash
rg -n "Training Progress|step:|dataset len|Total training steps" /tmp/ray/session_latest/logs/worker-*
```

本次真正证明训练已启动的标志是：

- `Training Progress: 1/435`
- `step:1`
- `timing_s/step:137.5525...`

#### 解决方案

把 Ray worker 的 out/err 聚合到一个固定 `.ray.log` 文件里，或者在 `tmux` 里专门开一个窗口 `tail -F`。

#### 原理说明

这是分布式训练框架的日志拓扑问题，不是训练失败。对 `verl + Ray` 来说，先找对日志，比盲目重启更重要。

### 5.7 首轮静默期较长是正常现象

#### 现象

启动后的前一两分钟里，除了 GPU 显存变化，终端可能几乎不刷 step 日志。

#### 根因

首轮在做的事情很多：

- Ray actor 创建
- 模型加载到 8 卡
- FSDP 初始化
- vLLM engine 初始化
- 首轮 rollout
- old log prob / ref log prob
- 第一次 actor update

这几步全部堆在 step 1 前面，所以首步特别慢很正常。

#### 本次实测

step 1 总耗时约：

- `137.55s`

这不是异常，而是这套组合在 8x4090 上的真实冷启动代价。

#### 什么时候才算真的卡住

可以用下面的经验判断：

- 如果 8 张卡持续有显存但完全 0 util，很久没有 `Training Progress`，要怀疑卡死
- 如果 Ray worker 日志持续出现同一栈 trace，说明在反复重试同一个初始化错误
- 如果已经看到 `step:1`，那就不是数据加载阶段了

## 6. 这次实际打过的补丁和脚本

### 6.1 新增脚本

- `/data/zyk_data/RL/prepare_gsm8k_dataset.sh`
- `/data/zyk_data/RL/run_qwen2_5_3b_gsm8k_grpo_8x4090.sh`

作用：

- 固化本地模型和数据路径
- 清理代理环境变量
- 做 batch 参数合法性检查
- 用 8 卡启动 `verl` 的 `main_ppo`

### 6.2 修改的 `verl` 源码

- `/data/zyk_data/RL/verl/verl/utils/model.py`
- `/data/zyk_data/RL/verl/verl/workers/fsdp_workers.py`

作用：

- 兼容 `transformers 5.3.0` 缺失 `AutoModelForVision2Seq`

### 6.3 运行时兼容补丁

- repo 级：
  - `/data/zyk_data/RL/verl/sitecustomize.py`
- env 级：
  - `/home/zyk/miniconda3/envs/verl/lib/python3.12/site-packages/sitecustomize.py`

作用：

- 给 tokenizer 补 `all_special_tokens_extended`
- 给 tokenizer 补 `special_tokens_map_extended`
- 在 `numba` 不可导入时，拦截 `vllm.v1.spec_decode.ngram_proposer`

## 7. 一套通用排查流程

以后再跑 `verl`，建议固定按下面顺序查，不要上来就重装环境。

### 第一步：先看是不是参数层问题

检查：

- GPU 数和 `rollout.tensor_model_parallel_size` 是否整除
- `TRAIN_BATCH_SIZE * ROLLOUT_N` 是否能被 `N_GPUS` 整除
- `PPO_MINI_BATCH_SIZE * ROLLOUT_N / N_GPUS` 是否能被 `PPO_MICRO_BATCH_SIZE_PER_GPU` 整除

如果这里不合法，先改 launcher，不要急着怪 `verl`。

### 第二步：看 import 栈是炸在谁身上

典型关键字和归因：

- `AutoModelForVision2Seq`: `transformers` API 变更
- `all_special_tokens_extended`: `vllm` 和 tokenizer API 不兼容
- `Numba needs NumPy 2.2 or less`: `numba/numpy` 版本冲突

### 第三步：确认问题发生在哪一层

- 主日志一开始就崩：通常是主进程 import 失败
- Ray worker 才崩：通常是 worker/vLLM 初始化期问题
- 有 `Training Progress` 但没 `step:`：通常在首轮 rollout 或 log prob
- 有 `step:1`：说明训练已经真的开始了

### 第四步：用最小命令单独验证补丁是否生效

验证 `transformers` 兼容：

```bash
source /home/zyk/miniconda3/etc/profile.d/conda.sh
conda run -n verl python -c "from transformers import AutoModelForImageTextToText; print('ok')"
```

验证 tokenizer 扩展属性：

```bash
source /home/zyk/miniconda3/etc/profile.d/conda.sh
conda run -n verl python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('/data/zyk_data/models/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1'); print(t.all_special_tokens_extended)"
```

验证 vLLM import 是否已绕过 `numba` 问题：

```bash
source /home/zyk/miniconda3/etc/profile.d/conda.sh
conda run -n verl python -c "import vllm.v1.worker.gpu_model_runner; print('ok')"
```

### 第五步：确认训练是否真的在跑

看三件事：

1. Ray worker 日志里有没有 `Training Progress`
2. Ray worker 日志里有没有 `step:1`
3. `nvidia-smi` 里 8 张卡是否都有显存占用和 util 波动

## 8. 哪些 warning 是正常的

下面这条 warning 在 `GRPO` 下不是故障：

```text
Disabled critic as algorithm.adv_estimator != gae
```

原因是：

- 你现在跑的是 `algorithm.adv_estimator=grpo`
- 不是 `gae`
- 所以 `verl` 会提示 critic 被禁用

这条 warning 可以记录，但不要把它当成致命错误。

## 9. 后续建议

### 9.1 如果目标是先跑通

优先保留当前方案：

- 用现成环境
- 保留 launcher 参数检查
- 保留 `sitecustomize.py` 兼容补丁
- 固定通过 Ray worker 日志看训练进度

这是当前成本最低、成功率最高的路径。

### 9.2 如果目标是长期稳定复用

建议单独做一版“干净环境”：

1. 把 `transformers` pin 到与当前 `verl` 源码兼容的版本
2. 把 `numpy` pin 到 `<=2.2`，或者升级到与 `numba` 兼容的新组合
3. 把 `AutoModelForVision2Seq` 的兼容逻辑提交到实验室维护分支
4. 评估是否还能去掉 env 级 `sitecustomize.py`

换句话说：

- 当前方案适合“先把实验跑起来”
- 干净环境方案适合“作为实验室标准环境长期复用”

## 10. 一页式速查表

### 看到这个错

```text
ImportError: cannot import name 'AutoModelForVision2Seq'
```

先查：

- `transformers` 版本
- `verl/utils/model.py`
- `verl/workers/fsdp_workers.py`

### 看到这个错

```text
AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
```

先查：

- `vllm` 和 `transformers` 版本组合
- `sitecustomize.py` 是否真的被 Ray worker 加载

### 看到这个错

```text
ImportError: Numba needs NumPy 2.2 or less. Got NumPy 2.4.
```

先查：

- `numpy` 当前版本
- `numba` 当前版本
- `vllm.v1.spec_decode.ngram_proposer` 是否被 import

### 没报错但就是没日志

先看：

- `/tmp/ray/session_latest/logs/worker-*.out`
- `/tmp/ray/session_latest/logs/worker-*.err`

不要先看启动脚本自己的 `.log`。

### 不确定有没有开始训练

只要看到：

- `Training Progress: 1/...`
- `step:1`

就说明已经真正进入训练，不是在数据加载。

## 11. 新增排障记忆：训练跑完后仍报 `DataLoader worker ... killed by signal: Killed`

### 11.1 现象

一次完整的 8 卡 GRPO 训练日志文件：

- `/data/zyk_data/RL/logs/qwen2_5_3b_lora_8x4090_20260321_182216.ray.log`

表面上看训练已经结束：

- 进度到 `435/435`
- 已打印最终 checkpoint 路径：
  - `/data/zyk_data/RL/checkpoints/verl_grpo_gsm8k/qwen2_5_3b_lora_8x4090_20260321_182216/global_step_435`
- 已打印 final validation metrics

但日志最后仍然报错：

- `Exception ignored in: <function _StatefulMultiProcessingDataLoaderIter.__del__ ...>`
- `RuntimeError: DataLoader worker (pid 921998) is killed by signal: Killed.`

### 11.2 非常关键的判断

这个报错不代表训练主体失败。

这次 case 里，真正的顺序是：

1. 训练 step 已经全部完成
2. 最终 validation 已完成
3. `global_step_435` checkpoint 已经落盘
4. 报错发生在 dataloader 销毁/进程回收阶段

也就是说，这是“训练成功但退出不干净”，不是“训练中途炸了”。

### 11.3 为什么会这样

代码路径在：

- `verl/verl/trainer/ppo/ray_trainer.py`
- `torchdata.stateful_dataloader`

原始行为是：

1. `train_dataloader` 和 `val_dataloader` 使用 `StatefulDataLoader`
2. 脚本没有显式传 `data.dataloader_num_workers`
3. 默认配置来自：
   - `/data/zyk_data/RL/verl/verl/trainer/config/data/legacy_data.yaml`
   - 默认值是 `dataloader_num_workers: 8`
4. trainer 正常 `return` 时没有显式关闭 dataloader
5. worker 清理主要依赖 `torchdata` iterator 的 `__del__`

因此退出时如果某个 dataloader worker 已经先被系统或父进程清掉，`__del__ -> _shutdown_workers() -> join()` 会把这个状态重新抛出来。

### 11.4 根因判断

这是一个“多进程 dataloader 收尾竞态 + 高内存压力”的问题。

更具体地说：

1. `StatefulDataLoader` 的 worker 清理依赖析构路径，不够稳。
2. 这次训练末尾日志里的 CPU 内存使用已经接近：
   - `perf/cpu_memory_used_gb ~= 195.5 GB`
3. 在这种内存压力下，dataloader worker 被 `SIGKILL` 并不奇怪。

因此，最合理的判断不是“GRPO 算法错了”，而是：

- 训练主体完成
- dataloader worker 在退出阶段被杀
- `torchdata` 在析构 join 时把它报告成 `RuntimeError`

### 11.5 解决思路

分成两层：

1. 修退出路径
2. 降低这个 GSM8K 脚本对 dataloader 多进程的依赖

第一层更本质：

- 不要再完全依赖 `__del__` 清理 dataloader
- trainer 正常结束时主动 shutdown dataloader iterator

第二层更实用：

- 这个 GSM8K 任务数据集小、训练主要耗时在 rollout 和 actor update
- 默认 `num_workers=8` 的收益很小
- 但会额外增加多进程退出复杂度和内存压力
- 所以脚本默认改成 `DATALOADER_NUM_WORKERS=0`

### 11.6 实际修改

#### 1. trainer 显式关闭 dataloader

修改文件：

- `/data/zyk_data/RL/verl/verl/trainer/ppo/ray_trainer.py`

新增方法：

- `_shutdown_dataloader()`
- `_shutdown_dataloaders()`

关键作用：

1. 读取 dataloader 的内部 `_iterator`
2. 如果 iterator 有 `_shutdown_workers()`，在 trainer 正常结束时主动调用
3. 把 dataloader 的 `_iterator` 置空，避免后续再走不受控析构路径

调用时机：

- `val_only` 返回前
- 正常训练完成、打印 final validation metrics 后返回前

#### 2. 训练脚本显式传 dataloader worker 数

修改文件：

- `/data/zyk_data/RL/run_qwen2_5_3b_gsm8k_grpo_8x4090.sh`

新增行为：

- 增加环境变量：
  - `DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"`
- 打印当前值
- 启动训练时显式传：
  - `data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}"`

### 11.7 为什么这里敢把默认值设成 0

这是针对这个脚本的工程判断，不是通用结论。

适用理由：

1. 任务是 `GSM8K`
2. 数据集不大
3. 训练中真正耗时的是：
   - rollout/generation
   - old log prob
   - ref log prob
   - actor update
4. 不在 dataloader
5. 小数据集下 `8` 个 worker 的吞吐收益有限，但退出成本明显增加

所以这里默认 `0` 是“稳定性优先”的合理折中。

如果后续有人要改回更高并行度，建议：

- 先试 `1`
- 再试 `2`
- 不要默认直接回到 `8`

### 11.8 如何快速确认是不是同一个问题

如果下次日志最后又报 dataloader 相关错误，不要先看最后一行，要按这个顺序判断：

1. 先搜有没有最终 step：
   - 例如 `step:435`
2. 再搜有没有最终 checkpoint 路径：
   - 例如 `local_global_step_folder: .../global_step_435`
3. 再搜有没有 final validation metrics
4. 如果上面三者都在，最后才出现：
   - `Exception ignored in: ... __del__`
   - `DataLoader worker ... killed by signal: Killed`
   那基本就是同类问题：训练成功，退出不干净

### 11.9 验证结果

为了验证修复不是“看起来合理”，而是真的有效，跑了两次 smoke run。

#### smoke run 1

- 实验名：
  - `qwen2_5_3b_smoke_20260322_123000`
- 配置：
  - `2 GPU`
  - `train_max_samples=4`
  - `val_max_samples=4`
  - `total_epochs=1`
  - 总步数 `1`
- 结果：
  - 正常训练
  - 正常 validation
  - 正常保存 `global_step_1`
  - 退出码 `0`
  - 日志中未出现：
    - `Exception ignored`
    - `DataLoader worker`
    - `killed by signal`

#### smoke run 2

- 实验名：
  - `qwen2_5_3b_smoke32x16_20260322_123500`
- 配置：
  - `2 GPU`
  - `train_max_samples=32`
  - `val_max_samples=16`
  - `total_epochs=1`
  - 总步数 `8`
- 结果：
  - 正常训练到 `8/8`
  - 正常按 `save_freq=2` 保存 `global_step_2/4/6/8`
  - 最终 `global_step_8` 落盘
  - 退出码 `0`
  - 同样未出现 dataloader cleanup 报错

### 11.10 下次遇到同类问题时的推荐动作

1. 先判断是否已经训练成功，不要看到最后一行报错就直接判 run 失败。
2. 先看最终 checkpoint 是否存在。
3. 如果是同类 dataloader cleanup 问题：
   - 优先检查 trainer 是否还带着显式 dataloader shutdown 补丁
   - 再检查当前脚本是否又把 `data.dataloader_num_workers` 调高了
4. 对 GSM8K 这类小数据集：
   - 默认保持 `DATALOADER_NUM_WORKERS=0`
5. 如果必须提高 worker 数：
   - 先从 `1` 或 `2` 开始压测
   - 同时盯：
     - `perf/cpu_memory_used_gb`
     - 尾部是否出现 dataloader worker kill

### 11.11 一句话总结

`435/435` 跑完后出现 `DataLoader worker ... killed by signal: Killed`，优先把它当成“退出阶段的 dataloader 清理问题”，不是“GRPO 没跑完”；修复重点是“trainer 显式关闭 dataloader + 这个脚本默认不用多进程 dataloader”。
