# Training V3.1

V3 数据（2023+2024+2025），训练参数同 V2.3。

## 参数

- 基座：Qwen3-8B
- QLoRA：4-bit NF4, r=64, α=128, dropout=0.05
- target_modules: q/k/v/o_proj
- epochs: 3, lr: 1.5e-4, cosine
- batch=1, grad_accum=16, seq=1024
- NEFTune α=5
- 输出：models/huchat-lora-v7

## 用法

```bash
python training/v3.1/train.py
python training/v3.1/merge_lora.py
```
