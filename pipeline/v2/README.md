# Pipeline V2

产出数据：`data/v2/`

## 参数

- 语料范围：2024 + 2025 年
- step2 MAX_LINES：24（约 12 轮）
- step3 质量评分阈值：>= 40 分
- step4 MAX_TURNS：8（硬截到 8 轮）
- step4 去重：基于 assistant 内容 hash
- step6 train/eval 切分：9:1，seed=42

## 步骤

1. `step0_explore.py` — 语料格式探查
2. `step1_clean.py` — 粗切去噪
3. `step2_segment.py` — 结构化切分
4. `step3_score.py` — 质量评分过滤（口头禅密度/发言长度/互动质量/噪声惩罚）
5. `step4_pairs.py` — 构造 ChatML 训练对
6. `step6_export.py` — 输出 train.json / eval.json

## 产出

- 2667 条训练 / 297 条验证
- 平均 7.8 轮/条（≤8）
