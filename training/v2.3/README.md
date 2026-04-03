# Model V2.3

数据版本：V2（`data/v2/`）
基座：Qwen3-8B

## 相对 V2.2 的变更

- **NEFTune α=5**：训练时给 embedding 加噪声，抗重复
- **3 epoch**（V2.2 是 2）：配合低 lr 多跑一轮
- **lr 1.5e-4**（V2.2 是 2e-4）：降低学习率减少过拟合
- **seq=1024**（V2.2 也是 1024）：不变

## 训练参数

- LoRA r=64, alpha=128, dropout=0.05
- target_modules: q_proj, k_proj, v_proj, o_proj
- QLoRA 4-bit NF4
- max_seq_len: 1024
- batch_size: 1, grad_accum: 16 (effective batch=16)
- epochs: 3
- lr: 1.5e-4, warmup: 3%
- NEFTune noise alpha: 5
- bf16, gradient_checkpointing

## 输出

- LoRA: `models/huchat-lora-v4/`
- 日志: `logs/run4/`
