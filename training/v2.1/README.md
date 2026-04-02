# Model V2.1

数据版本：V2（`data/v2/`）
基座：Qwen3-8B

## 训练参数

- LoRA r=32, alpha=64, dropout=0.05
- target_modules: q_proj, k_proj, v_proj, o_proj
- QLoRA 4-bit NF4
- max_seq_len: 2048
- batch_size: 1, grad_accum: 16 (effective batch=16)
- epochs: 1
- lr: 2e-4, warmup: 3%
- bf16, gradient_checkpointing
- eval_strategy: no（eval 在训练中 OOM）

## 结果

- loss: 2.24, acc: 60%
- 有改善但仍 ~70% 重复循环
- 原因：1 epoch 可能欠拟合

## Ollama 部署

- 名称：`huchatfunV2`
- 量化：Q4_K_M（5GB）
