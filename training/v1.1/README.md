# Model V1.1

数据版本：V1（`data/v1/`）
基座：Qwen3-8B

## 训练参数

- LoRA r=32, alpha=64, dropout=0.05
- target_modules: q_proj, k_proj, v_proj, o_proj
- QLoRA 4-bit NF4
- max_seq_len: 1024
- batch_size: 1, grad_accum: 16 (effective batch=16)
- epochs: 3
- lr: 2e-4, warmup: 3%
- bf16, gradient_checkpointing

## 结果

- loss: 3.80 → 2.33
- 风格学到了但严重重复循环
- 原因：数据太长（26轮/条）被 seq=1024 截断 + 3 epoch 过拟合

## Ollama 部署

- 名称：`huchatfun`
- 量化：Q4_K_M（5GB）
