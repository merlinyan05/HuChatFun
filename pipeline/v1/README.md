# Pipeline V1

产出数据：`data/v1/`

## 参数

- 语料范围：仅 2025 年
- step2 MAX_LINES：60（约 30 轮）
- 无质量评分（无 step3）
- 无训练对构造（无 step4/step6，V1 训练对构造方式已不可考）

## 步骤

1. `step0_explore.py` — 语料格式探查
2. `step1_clean.py` — 粗切去噪
3. `step2_segment.py` — 结构化切分

## 产出

- 1581 条训练 / 176 条验证
- 平均 26.3 轮/条
