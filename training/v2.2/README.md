# Model V2.2

数据版本：V2（`data/v2/`）
基座：Qwen3-8B

## 训练参数

- LoRA r=64, alpha=128, dropout=0.05
- target_modules: q_proj, k_proj, v_proj, o_proj
- QLoRA 4-bit NF4
- max_seq_len: 1024（91% 数据在 1024 内）
- batch_size: 1, grad_accum: 16 (effective batch=16)
- epochs: 2
- lr: 2e-4, warmup: 3%
- bf16, gradient_checkpointing
- eval_strategy: no

## 结果

- 训练最佳权重
- r=64 翻倍容量 + 2 epoch 平衡

## Ollama 部署

- 名称：`huchatfunV3`（f16, 16GB）
- Q4 量化严重降质，已放弃
- V2.2-deploy-a（`huchatfunV4`）和 V2.2-deploy-b（`huchatfunV5`，当前最佳）是同权重调推理参数
