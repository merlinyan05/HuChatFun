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
