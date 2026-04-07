# Model V2.4

数据版本：V2（`data/v2/`）
基座：Qwen3-8B
框架：transformers + peft + trl 1.0.0（和 V2.3 相同，不用 Unsloth）

## 相对 V2.3 的变更

- **LoRA r=128**（V2.3 是 r=64）：翻倍微调容量
- **LoRA alpha=256**（V2.3 是 128）：保持 alpha/r=2 比例不变
- 其余参数完全一致

## 为什么不用 Unsloth

V3.1 尝试了 Unsloth，在 Windows 上遇到大量兼容性问题：
- trl 降级到 0.24 导致 messages 格式丢失 loss masking，模型角色边界模糊
- tokenizer 配置被修改（extra_special_tokens 清空），GGUF 推理严重退化
- GGUF 导出需要编译 llama.cpp，Windows 上失败
- 详见 `docs/troubleshooting.md` 2026-04-07 的记录

## 训练参数

- LoRA r=128, alpha=256, dropout=0.05
- target_modules: q_proj, k_proj, v_proj, o_proj
- QLoRA 4-bit NF4
- max_seq_len: 1024
- batch_size: 1, grad_accum: 16 (effective batch=16)
- epochs: 3
- lr: 1.5e-4, warmup: 3%
- NEFTune noise alpha: 5
- bf16, gradient_checkpointing
- optimizer: paged_adamw_8bit

## 显存预估

r=128 的 LoRA 参数量约为 r=64 的 2 倍（~1.5% trainable vs ~0.74%）。
预计显存 13-14 GB，RTX 5080 16GB 应该能撑住。

## 输出

- LoRA: `models/huchat-lora-v6/`
- 合并: `models/huchat-merged-v6/`
- 日志: `logs/run6/`
- Ollama: `huchatfunV8`
