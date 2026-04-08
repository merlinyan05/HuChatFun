# V3 数据管线设计

## 目标

将语料范围从 2024+2025 扩展到 2023+2024+2025，增加训练数据量，其余不变。

## 变更范围

**唯一变更**：`pipeline/v3/step1_clean.py` 的年份 glob 从 `2024年*` + `2025年*` 改为 `2023年*` + `2024年*` + `2025年*`。

管线其余步骤（step0/step2-step6）逻辑不变，只是输入/输出路径改为 `data/v3/`。

## 语料统计

| 年份 | 文件数 |
|------|--------|
| 2023 | 168 |
| 2024 | 167 |
| 2025 | 184 |
| 合计 | 519 |

去除 INC 文件、非对话文件后，预计有效文件 ~490 个。
经 step1-step4 清洗后，预计产出 4000-5000 条训练对（V2 用 351 个文件产出 2667 条）。

## 目录结构

```
pipeline/v3/          从 v2 复制，改 step1 年份范围 + 所有路径指向 data/v3/
data/v3/              各步中间产物
  step1_cleaned/
  step2_segmented/
  step3_scored/
  step4_pairs/
  final/              train.json + eval.json
training/v3.1/        训练脚本（r=64, α=128, 3ep, 其余同 V2.3）
deploy/v3.1/          Modelfile
```

## 训练参数（V3.1，同 V2.3）

- 基座：Qwen3-8B
- QLoRA：4-bit NF4, r=64, α=128, dropout=0.05
- target_modules: q/k/v/o_proj
- epochs: 3, lr: 1.5e-4, cosine scheduler
- batch=1, grad_accum=16, seq=1024
- NEFTune α=5
- 输出：`models/huchat-lora-v7`

## 部署

- 合并 LoRA → GGUF f16 → Ollama `huchatfunV9`
- Modelfile 同 V2.3（temperature=0.7, repeat_penalty=1.3, num_predict=150）

## 版本对照

| 版本 | 数据 | 关键变更 | Ollama |
|------|------|----------|--------|
| V2.3 (V6) | V2: 2667条, 2024+2025 | 当前最佳 | huchatfunV6 |
| V2.4 (V8) | V2: 2667条, 2024+2025 | r=128, 无提升 | huchatfunV8 |
| V3.1 (V9) | V3: ~4500条, 2023+2024+2025 | 数据量翻倍 | huchatfunV9 |

## 风险

- 2023 年语料风格可能偏早期，"安卓"梗密度较低，可能稀释整体风格浓度
- 数据量翻倍但 seq=1024 不变，训练时间约 2.5-3 小时（V2.3 是 1.5h）
