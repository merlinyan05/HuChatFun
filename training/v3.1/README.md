# Model V3.1

数据版本：V2（`data/v2/`，2667 条，与 V2.3 相同）
基座：Qwen3-8B（本地 `models/Qwen3-8B/`）
框架：**Unsloth 2026.4.4**（替换原生 transformers + peft）

## 目的

V3.1 是 Unsloth 框架的**基线验证版**。训练参数与 V2.3 完全一致，用于确认框架切换不影响效果，同时测量速度和显存改善。

## 相对 V2.3 的变更

| 项目 | V2.3 | V3.1 |
|------|------|------|
| 框架 | transformers 5.5 + peft 0.18 + trl 1.0 | **Unsloth 2026.4.4** + trl 0.24 |
| 模型加载 | AutoModelForCausalLM + BitsAndBytesConfig | FastLanguageModel.from_pretrained |
| LoRA 配置 | get_peft_model | FastLanguageModel.get_peft_model |
| gradient_checkpointing | 原生 | **Unsloth 优化版**（额外省 ~30% 显存） |
| 优化器 | paged_adamw_8bit | **adamw_8bit**（Unsloth 推荐） |
| 数据预处理 | SFTTrainer 自动处理 messages | 手动 map 转 ChatML text 字段 |
| 显存峰值 | ~11-13 GB | **~6.4 GB**（dry run 实测） |

超参数（r/alpha/lr/epoch/seq_len/NEFTune）完全一致，不做任何调整。

## 训练参数

- LoRA r=64, alpha=128, dropout=0.05
- target_modules: q_proj, k_proj, v_proj, o_proj
- QLoRA 4-bit NF4（Unsloth 内置）
- max_seq_len: 1024
- batch_size: 1, grad_accum: 16 (effective batch=16)
- epochs: 3
- lr: 1.5e-4, warmup: 3%, cosine scheduler
- NEFTune noise alpha: 5
- bf16, gradient_checkpointing (unsloth)
- optimizer: adamw_8bit
- Unsloth 自动特性: padding-free, gradient offload

## 环境

- Python 3.11.15（conda env: huchat）
- Unsloth 2026.4.4 + unsloth_zoo 2026.4.3
- PyTorch 2.11.0+cu128（手动重装 CUDA 版，见 troubleshooting.md）
- Xformers 0.0.35（替代 Flash Attention 2）
- triton-windows 3.6.0
- NVIDIA GeForce RTX 5080 16GB, Driver 595.97
- Windows 11 Pro

## Windows 注意事项

1. **编码问题**：脚本顶部需 `sys.stdout.reconfigure(encoding="utf-8")`，否则 Unsloth 的 emoji 输出会崩
2. **PowerShell 环境变量**：用 `$env:PYTHONIOENCODING="utf-8"; python train.py`
3. **数据格式**：不能用 `tokenizer.apply_chat_template`，需手动拼 ChatML

详见 `docs/troubleshooting.md` 中 2026-04-07 的三条记录。

## 运行命令

```powershell
# 训练
$env:PYTHONIOENCODING="utf-8"; python training/v3.1/train.py

# 导出 GGUF
$env:PYTHONIOENCODING="utf-8"; python training/v3.1/export_gguf.py
```

## 输出

- LoRA: `models/huchat-lora-v5/`
- 日志: `logs/run5/`
- GGUF: `models/huchat-merged-v5/`（由 export_gguf.py 导出）

## 训练结果

训练中，待填写。

<!--
完成后补充：
- final loss / accuracy
- 训练总时长
- 与 V2.3 的对比（loss 2.10 / acc 0.61 / 显存 ~12GB）
-->
