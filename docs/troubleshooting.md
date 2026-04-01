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
