# HuChatFun — AI 户晨风

基于户晨风直播语料训练的 AI 对话模型。目标：口头禅比真人更密集，观点比真人更爆。

仅供娱乐，非官方项目。AI 生成内容不代表户晨风本人观点。

## 技术方案

| 项       | 选型                                                          |
| -------- | ------------------------------------------------------------- |
| 基座模型 | Qwen3-8B（原 Qwen3.5-9B，因 Windows 兼容性问题已更换）       |
| 微调方法 | QLoRA（4-bit NF4, LoRA r=32）                                 |
| 训练硬件 | RTX 5080 16GB                                                 |
| 推理部署 | Mac Mini M4 16GB → Ollama                                    |
| 最终接入 | OpenClaw                                                      |
| 语料     | 户晨风直播文字稿 2024-2025（V2 当前使用范围）                 |

## 快速开始

```bash
# 1. 数据清洗（按顺序跑）
python pipeline/step1_rough_clean.py
python pipeline/step2_segment.py
python pipeline/step3_score_filter.py
python pipeline/step4_build_pairs.py
python pipeline/step6_finalize.py

# 2. 训练
CUDA_VISIBLE_DEVICES=0 python training/train.py

# 3. 合并 + 量化
python training/merge_lora.py
bash training/convert_gguf.sh

# 4. 部署
ollama create huchatfun -f deploy/Modelfile
ollama run huchatfun
```

## 目录说明

```
corpus/          原始语料（只读）
pipeline/        数据处理脚本（step0-6）
data/
  v1/            第一版数据产物（已归档）
  v2/            第二版数据产物（当前）
training/        训练、合并脚本
eval/            评估脚本
deploy/          Ollama Modelfile
models/          模型文件（不入库，含 v1/v2 LoRA）
logs/            训练日志（run1, run2, ...）
docs/            项目文档
tools/           第三方工具（不入库）
```

详细目录结构见 `docs/project_structure.md`。

## 文档索引

- [完整实施指南](docs/implementation_guide.md)
- [数据清洗策略](docs/cleaning_strategy.md)
- [数据清洗工作流](docs/cleaning_workflow.md)
- [项目目录结构](docs/project_structure.md)
- [语料格式探查记录](docs/format_notes.md)
- [踩坑记录](docs/troubleshooting.md)
- [实验日志](logs/experiments.md)

## 免责声明

本项目仅供个人学习和娱乐，与户晨风本人无关。AI 生成的所有内容不代表户晨风的真实观点。请勿用于商业用途或冒充本人。
