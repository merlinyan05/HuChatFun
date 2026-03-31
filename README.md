# HuChatFun — AI 户晨风

基于户晨风直播语料训练的 AI 对话模型。目标：口头禅比真人更密集，观点比真人更爆。

仅供娱乐，非官方项目。AI 生成内容不代表户晨风本人观点。

## 技术方案

| 项       | 选型                                                          |
| -------- | ------------------------------------------------------------- |
| 基座模型 | Qwen3.5-9B                                                    |
| 微调方法 | QLoRA（4-bit NF4, LoRA r=64）                                 |
| 训练硬件 | RTX 5080 16GB                                                 |
| 推理部署 | Mac Mini M4 16GB → Ollama                                    |
| 最终接入 | OpenClaw                                                      |
| 语料     | 户晨风直播文字稿 2023-2025（400+ 场）（有可能只选取近几年的） |

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
corpus/      原始语料（只读）
pipeline/    数据处理脚本（step0-6）
data/        各阶段产物 → data/final/ 为最终训练集
training/    训练、合并、量化脚本
eval/        评估脚本
deploy/      Ollama + OpenClaw 配置
models/      模型文件（不入库）
docs/        项目文档
```

详细目录结构见 `docs/project_structure.md`。

## 文档索引

- [完整实施指南](docs/implementation_guide.md)
- [数据清洗策略](docs/cleaning_strategy.md)
- [数据清洗工作流](docs/cleaning_workflow.md)
- [项目目录结构](docs/project_structure.md)
- [语料格式探查记录](docs/format_notes.md)
- [实验日志](logs/experiments.md)

## 免责声明

本项目仅供个人学习和娱乐，与户晨风本人无关。AI 生成的所有内容不代表户晨风的真实观点。请勿用于商业用途或冒充本人。
