# 踩坑记录

> 开发过程中遇到的问题、排查过程和解决方案。每次遇到新问题都追加到这里。

---

## 2026-04-01: Qwen3.5-9B 在 Windows 上无法训练

### 现象

1. 首次运行 `training/train.py`，电脑直接关机（无错误日志）
2. 降低参数后重跑，报 `CUBLAS_STATUS_INTERNAL_ERROR`
3. 加 `CUDA_LAUNCH_BLOCKING=1` 后确认真实错误为 `CUDA error: out of memory`

### 根因

Qwen3.5-9B 使用**混合注意力架构**（标准注意力 + gated delta rule 线性注意力）。其线性注意力的 torch 回退实现会将 q/k/v/beta/g 全部转为 **float32**：

```python
# transformers/models/qwen3_5/modeling_qwen3_5.py:250
x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
```

这导致注意力计算的显存占用翻倍，RTX 5080 16GB 即使 batch=1 + seq_len=512 也 OOM。

优化路径需要 `flash-linear-attention` + `causal-conv1d`，但这两个库都依赖 **triton**，而 triton 不支持 Windows。

### 尝试过的方案

| 方案 | 结果 |
|------|------|
| 降 seq_len 2048→1024→512 | OOM，fp32 转换是根本瓶颈 |
| 降 LoRA r 64→32 | 无济于事，瓶颈不在 LoRA |
| `pip install flash-linear-attention --no-deps` | 装了个空壳，反而导致模型无法加载（`No module named 'fla.modules'`） |
| `pip install causal-conv1d` | 编译失败，Windows 上找不到 nvcc |

### 解决方案

**换基座模型为 Qwen3-8B**。Qwen3 使用标准 Transformer 架构，不依赖 triton，QLoRA 兼容性好。

### 教训

- 选基座模型时必须确认其**训练时**的依赖兼容性，不只是推理
- 混合架构（Qwen3.5、Mamba 等）在 Windows + 消费级 GPU 上微调要谨慎
- `CUBLAS_STATUS_INTERNAL_ERROR` 不一定是驱动问题，加 `CUDA_LAUNCH_BLOCKING=1` 能暴露真实错误
- 电脑突然关机 ≈ GPU OOM 导致驱动崩溃，先查显存再查电源/散热

---

## 2026-04-01: TRL 1.0.0 中 SFTConfig 参数名变更

### 现象

训练脚本报错：`TypeError: SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'`

### 根因

TRL 1.0.0 将 `max_seq_length` 改名为 `max_length`。很多网上教程和旧版代码用的是 `max_seq_length`。

### 解决方案

```python
# 错误
sft_config = SFTConfig(max_seq_length=1024, ...)

# 正确（TRL 1.0.0）
sft_config = SFTConfig(max_length=1024, ...)
```

查参数名的方法：
```python
from trl import SFTConfig; import inspect
sig = inspect.signature(SFTConfig.__init__)
print([p for p in sig.parameters if 'max' in p or 'length' in p])
```

### 教训

- TRL 版本迭代快，参数名会变，不要盲信教程，先查当前版本的签名
- 同时注意 `warmup_ratio` 也已 deprecated，改用 `warmup_steps`

---

## 2026-03-31: huggingface-cli 在 conda 环境中找不到

### 现象

`huggingface_hub` 已安装（1.8.0），但 PowerShell 中运行 `huggingface-cli` 报"无法识别为 cmdlet"。
`python -m huggingface_hub.commands.cli` 也报 `ModuleNotFoundError: No module named 'huggingface_hub.commands'`。

### 根因

`huggingface_hub` 1.8.0 使用 `typer` 作为 CLI 框架，入口点脚本没有正确安装到 conda 环境的 `Scripts/` 目录下。这在 Windows conda 环境中偶发。

### 解决方案

不用 CLI，直接用 Python API：

```python
import os
from huggingface_hub import snapshot_download
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
snapshot_download(repo_id='Qwen/Qwen3-8B', local_dir='models/Qwen3-8B', token='hf_xxx')
```

### 教训

- Windows conda 环境下 CLI 入口点经常出问题，Python API 是更可靠的替代
- `pip install --force-reinstall huggingface_hub` 有时能修复，但不保证

---

## 2026-03-31: HuggingFace 模型下载各种问题汇总

### 问题1：不挂镜像下载极慢

**现象**：不设 `HF_ENDPOINT`，直接连 huggingface.co 下载，速度只有 400KB/s-2MB/s，还会被警告 "unauthenticated requests"。

**解决方案**：
```python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内镜像
token = 'hf_xxx'  # 加 token 提高速率限制
```

### 问题2：大文件下载频繁超时断开

**现象**：下载 safetensors 分片（~4-5GB/个）时反复报 `The read operation timed out` 和 `peer closed connection without sending complete message body`，需要多次重试才能完成。

**根因**：hf-mirror.com 的 xethub CDN 对大文件连接不稳定，超时阈值较短。

**解决方案**：
- `snapshot_download` 自带断点续传（`resume_download` 在新版已默认开启），断了重跑即可
- 多跑几次，每次会跳过已完成的文件，逐步下完
- token 有助于减少限速导致的超时

### 问题3：下载进度条显示的总大小不断膨胀

**现象**：Qwen3.5-9B 实际约 18GB，但进度条显示总大小从 19.3G → 38.6G → 52.6G → 57.9G 不断增长。

**根因**：`snapshot_download` 的进度条是所有线程的累计字节数，包含了重试下载的重复计数。不影响实际结果，文件大小是正确的。

### 问题4：PowerShell 多行 Python 命令缩进报错

**现象**：在 PowerShell 中粘贴带缩进的多行 `python -c "..."` 命令，报 `IndentationError: unexpected indent`。

**解决方案**：确保 `python -c "` 后面的代码行**顶格写**，不要有前导空格：
```powershell
# 错误（有缩进）
python -c "
    import os
    print('hello')
"

# 正确（顶格）
python -c "
import os
print('hello')
"
```

### 问题5：新开 PowerShell 终端环境变量丢失

**现象**：之前设置的 `$env:HF_ENDPOINT` 在新终端窗口中失效。

**根因**：`$env:VAR = "value"` 只对当前 PowerShell 会话有效，关闭即丢失。

**解决方案**：
- 每次新终端重新设置：`$env:HF_ENDPOINT = "https://hf-mirror.com"`
- 或者直接在 Python 代码里用 `os.environ['HF_ENDPOINT'] = ...`（推荐，不依赖终端状态）

---

## 2026-04-01: PowerShell 中 `&&` 不能用

### 现象

```
cd D:\Workspace\Projects\HuChatFun && python training/train.py
```
报错：`标记"&&"不是此版本中的有效语句分隔符`。

### 根因

Windows PowerShell 5.x 不支持 `&&`（PowerShell 7+ 才支持）。

### 解决方案

- 用 `;` 代替：`cd D:\Workspace\Projects\HuChatFun; python training/train.py`
- 或者先 `cd`，再单独运行命令

---

## 2026-04-01: 训练中 eval 步骤 OOM

### 现象

Qwen3-8B QLoRA 训练 100 步正常（loss 3.8→2.47），但第 100 步触发 eval 时 OOM：
```
File "transformers/loss/loss_utils.py", line 55, in ForCausalLMLoss
    logits = logits.float()
torch.AcceleratorError: CUDA error: out of memory
```

Checkpoint 也没保存上（save 和 eval 在同一步触发，eval 先执行就崩了）。

### 根因

训练时 gradient checkpointing 节省显存，但 eval 不用 gradient checkpointing，需要把完整 logits 张量转 float32。Qwen3 词表 ~152K tokens，logits 大小 = 152K × 1024(seq_len) × 4 bytes ≈ **600MB**，加上模型本身占用，16GB 不够。

### 解决方案

关掉训练中的 eval，训练完单独评估：
```python
sft_config = SFTConfig(
    eval_strategy="no",  # 不在训练中 eval
    ...
)
```

### 教训

- 大词表模型（>100K）在消费级 GPU 上训练时，eval 容易 OOM
- `save_steps` 和 `eval_steps` 不要设成同一个值，否则 eval 崩了 checkpoint 也丢
- 如果一定要训练中 eval，可以设 `eval_accumulation_steps=1` 分批计算

---

## 2026-04-01: V1 模型部署后严重重复循环

### 现象

V1 模型（Qwen3-8B QLoRA, 3 epoch）部署到 Ollama 后：
- Q4_K_M 量化版：直接输出乱码（"嗯嗯嗯嗯"、"说说说说"、重复输入问题）
- f16 版：开头 1-2 句正常且有户晨风风格，但之后进入重复循环（"苹果是好，苹果是好，苹果是好"）

### 根因（多重）

1. **训练数据过长被截断**：平均 26.3 轮/条，seq_len=1024 截断后模型没见过对话结尾，不知道何时停止
2. **3 epoch 过拟合**：1581 条数据跑 3 遍，过拟合了直播语料中的重复口头禅模式
3. **Qwen3 对 Q4_K_M 量化敏感**：LoRA 微调后的权重分布不适合激进量化，f16 明显好于 Q4
4. **Ollama 缺少 chat template**：合并模型时 tokenizer_config.json 丢失了 chat_template，需在 Modelfile 中手动指定 TEMPLATE
5. **Qwen3 thinking 模式干扰**：在模板中加 `<think>\n\n</think>` 反而更差（训练数据没有 think tokens）

### 解决方案

V2 数据重构 + 重新训练：
- 加入 2024 年数据，增加多样性
- 每条对话限制 ≤8 轮，确保完整结尾在 token 范围内
- 训练改为 1 epoch，seq_len=2048
- 部署使用 f16 或 Q6_K 量化
- Modelfile 中手动指定 ChatML TEMPLATE + stop tokens

### 教训

- 微调后的模型对量化更敏感，先用 f16 验证效果再量化
- chat template 必须在 Modelfile 中明确指定，不能依赖 GGUF 内嵌
- 训练数据的长度分布必须和推理时的使用场景匹配（短对话场景就训短对话）
- Qwen3 的 thinking 模式需要特殊处理，如果训练数据没用 think tokens 就不要在推理时加

---

## 2026-04-01: V2 模型仍有重复循环（改善但未解决）

### 现象

V2 模型（1 epoch / 2667 条 / 8 轮上限 / seq_len=2048）相比 V1 有明显改善：
- 偶尔产出高质量回复（"iPhone要5000，你月薪3000，买不起，这是事实，你明不明白啊？"）
- 学到了户晨风风格词（"纯纯的笑话"、"你给我记住了"、"你这个逻辑是错的"）
- 但大约 70% 的回复仍然进入重复循环或回显输入

### 尝试过的 Ollama 参数组合

| 配置 | 结果 |
|------|------|
| mirostat 2 + repeat_penalty 1.8 | 最好，偶尔出金句 |
| temperature 0.5 + repeat_penalty 2.0 | 太压制，只会回显输入 |
| temperature 0.7 + repeat_penalty 1.3 | 有好有坏，不稳定 |
| temperature 0.7 + repeat_penalty 1.5 | 差不多，不稳定 |

### Python 直连测试

用 transformers 直连（temperature=0.7, repetition_penalty=1.3）能生成较长的连贯文本，说明模型本身有能力，但 Ollama 的模板或采样实现可能有差异。

### 根因分析

重复循环是 LoRA 微调的固有局限，不是参数能解决的：
1. **LoRA 容量不足**：r=32 只占 0.37% 参数，不足以完全覆盖基座模型的生成行为
2. **数据量仍然不够**：2667 条 × 1 epoch ≈ 模型只见过每条数据一次，风格印记不够深
3. **缺少负反馈**：SFT 只教模型"说什么"，没教"不要重复"（需要 DPO/RLHF）

### 可能的改进方向

1. 提高 LoRA rank（r=64 或 r=128）增加微调容量
2. 增加到 2 epoch（在过拟合和欠拟合之间找平衡）
3. 换用 Qwen2.5-7B-Instruct（指令遵循能力更成熟，可能更鲁棒）
4. 加 DPO 训练阶段，专门惩罚重复生成
5. 在训练数据中加入 `<|im_end|>` 后的截断信号，强化停止行为

---

## 2026-04-07: Unsloth 安装替换 PyTorch CUDA 版本

### 现象

`pip install unsloth` 成功后，`torch.cuda.is_available()` 返回 `False`。

### 根因

Unsloth 依赖 `torch<2.11.0`，pip 从 PyPI 拉取了 CPU 版 torch 2.10.0 替换了原来的 CUDA 版 torch 2.11.0+cu128。PyPI 默认的 torch 包是 CPU 版。

### 解决方案

安装 Unsloth 后，强制重装 CUDA 版 torch：

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

虽然会产生 Unsloth 的版本兼容性警告（`requires torch<2.11.0`），实测 torch 2.11.0+cu128 可以正常使用。

### 教训

- 安装 Unsloth 后必须检查 `torch.cuda.is_available()`
- pip 的 `--index-url` 只对该次安装生效，后续安装其他包时 pip 会从 PyPI 拉取 CPU 版 torch
- `pip install torch` 不会重装（认为版本已满足），必须 `--force-reinstall`

---

## 2026-04-07: Unsloth 在 Windows 上输出 emoji 导致 UnicodeEncodeError

### 现象

Unsloth SFTTrainer 初始化时报错：
```
UnicodeEncodeError: 'gbk' codec can't encode character '\U0001f9a5' in position 0
```

### 根因

Unsloth 的日志输出含有 🦥（树懒 emoji），Windows 默认终端编码是 GBK，无法编码 emoji。

### 解决方案

在脚本最顶部加：
```python
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
```

或者在命令行设环境变量：`PYTHONIOENCODING=utf-8 python train.py`

### 教训

- Windows 上跑 Unsloth 必须处理编码问题
- `os.environ["PYTHONIOENCODING"]` 运行时设无效（Python 启动时就确定了 IO 编码），要用 `sys.stdout.reconfigure`

---

## 2026-04-07: Unsloth SFTTrainer 要求 formatting_func 或预处理 text 字段

### 现象

使用 `messages` 格式数据时，Unsloth 的 SFTTrainer 报错：
```
RuntimeError: Unsloth: You must specify a `formatting_func`
```

### 根因

Unsloth 重写了 SFTTrainer，不像原生 trl 那样自动处理 messages 格式。需要显式提供格式化函数或预处理数据。

### 解决方案

在数据集上预先做 map，手动拼 ChatML 格式文本：

```python
def messages_to_text(item):
    parts = []
    for msg in item["messages"]:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    return {"text": "\n".join(parts)}

train_ds = Dataset.from_list(data).map(messages_to_text)
```

不要用 `tokenizer.apply_chat_template`，Qwen3 的模板在 Unsloth 环境下可能报 `dict object has no element 0`。

### 教训

- Unsloth 的 SFTTrainer 和原生 trl 的 API 不完全兼容
- 最稳妥的方式是直接给 dataset 加 `text` 字段，避免依赖 formatting_func

---

## 2026-04-07: Unsloth 内置 GGUF 导出在 Windows 上失败

### 现象

`model.save_pretrained_gguf()` 合并权重成功，但转 GGUF 阶段报错：

```
RuntimeError: llama.cpp folder 'C:\Users\Merlin\.unsloth\llama.cpp' does not exist
RuntimeError: [FAIL] Command `pip install gguf protobuf sentencepiece mistral_common` failed with error
  `WARNING: Failed to remove contents in a temporary directory '...\~umpy.libs'`
```

### 根因

Unsloth 的 GGUF 导出会尝试：
1. 用 winget 安装 cmake、VS Build Tools、OpenSSL
2. 克隆并编译 llama.cpp 到 `~/.unsloth/llama.cpp`
3. 安装 gguf 等 Python 包

在 Windows 上第 2 步 git clone 失败（目录不存在），第 3 步 pip install 因为临时目录锁定也失败。即使依赖都装上了，Windows 编译 llama.cpp 也经常出问题。

### 解决方案

**不用 Unsloth 的 GGUF 导出**。`save_pretrained_gguf` 在合并权重阶段已经成功，safetensors 文件已保存。用项目里现有的 llama.cpp 手动转：

```powershell
python tools/llama.cpp/convert_hf_to_gguf.py models/huchat-merged-v5 --outtype f16 --outfile models/huchat-merged-v5/huchatfun-v3.1-f16.gguf
```

### 教训

- Unsloth 的 `save_pretrained_gguf` 在 Linux 上很方便，但 **Windows 上不可靠**，不要依赖
- 好在合并权重（`save_pretrained_merged` 的逻辑）在 GGUF 转换之前执行，即使后面失败了权重也不会丢
- Windows 项目统一用 `tools/llama.cpp` 转 GGUF，不要每次都尝试编译新的
