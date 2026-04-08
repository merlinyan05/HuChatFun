# V3 数据管线实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将语料范围从 2024+2025 扩展到 2023+2024+2025，数据量翻倍，训练并部署 V3.1 模型。

**Architecture:** 从 pipeline/v2 复制 6 个脚本到 pipeline/v3，唯一逻辑变更是 step1 的年份 glob 加入 2023。所有中间产物输出到 data/v3/。训练脚本复制自 V2.3（r=64），路径指向 V3 数据。

**Tech Stack:** Python, transformers, peft, trl, llama.cpp, Ollama

---

## 文件结构

```
pipeline/v3/                  # 从 v2 复制，改路径
  README.md                   # 更新说明
  step0_explore.py            # 路径改为 v3（可选，不影响产出）
  step1_clean.py              # 年份 glob 加入 2023年*，输出路径改 data/v3/
  step2_segment.py            # 输入输出路径改 data/v3/
  step3_score.py              # 输入输出路径改 data/v3/
  step4_pairs.py              # 输入输出路径改 data/v3/
  step6_export.py             # 输入输出路径改 data/v3/

training/v3.1/                # 从 V2.3 复制，改数据路径和输出路径
  train.py                    # 数据 → data/v3/final/，输出 → models/huchat-lora-v7
  merge_lora.py               # LoRA → models/huchat-lora-v7，输出 → models/huchat-merged-v7

deploy/v3.1/                  # 从 V2.3 复制，改 GGUF 路径
  Modelfile                   # FROM 指向 models/huchat-v3.1-f16.gguf
```

---

### Task 1: 创建 pipeline/v3 管线脚本

**Files:**
- Create: `pipeline/v3/step1_clean.py`
- Create: `pipeline/v3/step2_segment.py`
- Create: `pipeline/v3/step3_score.py`
- Create: `pipeline/v3/step4_pairs.py`
- Create: `pipeline/v3/step6_export.py`
- Create: `pipeline/v3/README.md`

- [ ] **Step 1: 复制 V2 管线到 V3 并修改路径**

从 `pipeline/v2/` 复制以下文件到 `pipeline/v3/`：
- `step1_clean.py` — 修改两处：
  1. 年份 glob：加入 `2023年*`
     ```python
     # V2:
     target_dirs = sorted(CORPUS_DIR.glob("2024年*")) + sorted(CORPUS_DIR.glob("2025年*"))
     # V3:
     target_dirs = sorted(CORPUS_DIR.glob("2023年*")) + sorted(CORPUS_DIR.glob("2024年*")) + sorted(CORPUS_DIR.glob("2025年*"))
     ```
  2. OUTPUT_DIR 路径：`data/v2/step1_cleaned` → `data/v3/step1_cleaned`

- `step2_segment.py` — INPUT_DIR 和 OUTPUT_DIR 路径 `v2` → `v3`
- `step3_score.py` — INPUT_DIR 和 OUTPUT_DIR 路径 `v2` → `v3`
- `step4_pairs.py` — INPUT_FILE 和 OUTPUT_DIR 路径 `v2` → `v3`
- `step6_export.py` — INPUT_FILE 和 OUTPUT_DIR 路径 `v2` → `v3`

- [ ] **Step 2: 创建 README.md**

```markdown
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
```

- [ ] **Step 3: 验证管线脚本**

逐步运行并检查每步产出：
```bash
python pipeline/v3/step1_clean.py
python pipeline/v3/step2_segment.py
python pipeline/v3/step3_score.py
python pipeline/v3/step4_pairs.py
python pipeline/v3/step6_export.py
```

检查 `data/v3/final/train.json` 的条数（预期 4000-5000 条）。

- [ ] **Step 4: Commit**

```bash
git add pipeline/v3/ data/v3/final/
git commit -m "feat: V3 数据管线，加入 2023 年语料"
```

---

### Task 2: 创建训练脚本

**Files:**
- Create: `training/v3.1/train.py`
- Create: `training/v3.1/merge_lora.py`
- Create: `training/v3.1/README.md`

- [ ] **Step 1: 创建 train.py**

从 `training/v2.3/train.py` 复制，修改以下常量：
```python
TRAIN_DATA = str(ROOT / "data" / "v3" / "final" / "train.json")
EVAL_DATA = str(ROOT / "data" / "v3" / "final" / "eval.json")
OUTPUT_DIR = str(ROOT / "models" / "huchat-lora-v7")
LOG_DIR = str(ROOT / "logs" / "run7")
```

LoRA 参数保持 V2.3 的值：r=64, α=128。
脚本顶部注释改为 `V3.1`。

- [ ] **Step 2: 创建 merge_lora.py**

从 `training/v2.4/merge_lora.py` 复制（它已修复 tokenizer 从基座加载的问题），修改：
```python
LORA_ADAPTER = str(ROOT / "models" / "huchat-lora-v7")
OUTPUT_DIR = str(ROOT / "models" / "huchat-merged-v7")
```

- [ ] **Step 3: 创建 README.md**

```markdown
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
```

- [ ] **Step 4: Commit**

```bash
git add training/v3.1/
git commit -m "feat: V3.1 训练脚本（r=64, V3 数据）"
```

---

### Task 3: 创建部署配置

**Files:**
- Create: `deploy/v3.1/Modelfile`
- Create: `deploy/v3.1/README.md`

- [ ] **Step 1: 创建 Modelfile**

```
FROM ../../models/huchat-v3.1-f16.gguf

TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ end }}{{- if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}<|im_end|>
{{ end }}{{- end }}<|im_start|>assistant
"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.3
PARAMETER num_predict 150

SYSTEM """你是户晨风，B站著名直播博主。说话风格直接、犀利、有观点，喜欢在连麦对话中一针见血地点出对方逻辑漏洞。

核心特点：
- 爱用"安卓"作万能贬义词（安卓人/安卓逻辑/安卓学历）
- "纯纯"作强调词（纯纯废了/纯纯安卓）
- 用"购买力"衡量一切
- "前程似锦"是反讽，意思是没希望
- 口头禅：「知道吧」「能明白吗」「我告诉你」「你给我记住了」「咱有什么说什么」
- 判定对方没救时说「你废了」
- 不绕弯子，直接给结论，有时候怼完还补一句关怀"""
```

- [ ] **Step 2: Commit**

```bash
git add deploy/v3.1/
git commit -m "feat: V3.1 部署 Modelfile"
```

---

### Task 4: 运行训练（用户手动）

- [ ] **Step 1: 跑训练**

```bash
python training/v3.1/train.py
```

预计 2.5-3 小时。观察 loss（期望 < 2.10）和 accuracy。

- [ ] **Step 2: 合并 LoRA**

```bash
python training/v3.1/merge_lora.py
```

- [ ] **Step 3: 转 GGUF**

```bash
python tools/llama.cpp/convert_hf_to_gguf.py models/huchat-merged-v7 --outtype f16 --outfile models/huchat-v3.1-f16.gguf
```

- [ ] **Step 4: 导入 Ollama**

```bash
ollama create huchatfunV9 -f deploy/v3.1/Modelfile
```

- [ ] **Step 5: 测试**

```bash
ollama run huchatfunV9
```

测试问题：
- "你是谁"
- "我月薪3000，想买苹果"
- "如何评价华为"
- "我大专毕业，想进大厂"

对比 V6 的重复率和风格浓度。

---

### Task 5: 更新 CLAUDE.md

- [ ] **Step 1: 更新进度和版本表**

在 CLAUDE.md 中：
- 当前进度：加入 V2.4 完成记录 + V3.1 进度
- 版本对照表：加入 V2.4 和 V3.1
- 上次会话摘要：更新

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: 更新 CLAUDE.md V2.4 结果和 V3 进度"
```
