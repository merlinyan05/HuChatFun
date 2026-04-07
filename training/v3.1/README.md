# Model V3.1

数据版本：V2 或 V3（待定）
基座：待定（Qwen3-8B / Qwen4 / DeepSeek）
框架：Unsloth（替换原生 transformers + peft）

## 相对 V2.3 的变更

- 训练框架从 transformers+peft 换为 Unsloth（2x 提速 + 70% 显存节省）
- 可能换基座模型
- 可能加 DPO 二阶段

## V2 系列瓶颈（V3 需要解决的问题）

- loss/accuracy 在 V2.2 和 V2.3 之间持平（2.09-2.10 / 0.61-0.62）
- SFT + LoRA 无法有效解决重复循环
- V2 数据 2667 条已充分学习，同数据增加 epoch 无效

## 计划步骤

1. 在 Win 上装 Unsloth 环境
2. 用 Qwen3-8B + V2 数据跑通 Unsloth 流程
3. 对比 V2.3 效果（同测试问题）
4. （可选）换 Qwen4 / DeepSeek 基座
5. （可选）DPO 二阶段

详见 `docs/upgrade_plan.md`

## 训练参数

待定。

## 输出

- LoRA: `models/huchat-lora-v5/`（待定）
- 日志: `logs/run5/`（待定）
