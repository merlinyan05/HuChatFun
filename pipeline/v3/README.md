# Pipeline V3

产出数据：`data/v3/`

## 参数

- 语料范围：2023 + 2024 + 2025 年（V2 是 2024+2025）
- step2 MAX_LINES：24（约 12 轮）
- step3 质量评分阈值：>= 40 分
- step4 MAX_TURNS：8（硬截到 8 轮）
- step4 去重：基于 assistant 内容 hash
- step6 train/eval 切分：9:1，seed=42

## 步骤

1. `step1_clean.py` — 粗切去噪
2. `step2_segment.py` — 结构化切分
3. `step3_score.py` — 质量评分过滤
4. `step4_pairs.py` — 构造 ChatML 训练对
5. `step6_export.py` — 输出 train.json / eval.json

## 与 V2 的差异

唯一变更：step1 年份范围加入 2023 年（168 个文件）。
