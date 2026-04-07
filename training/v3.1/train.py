"""
HuChatFun V3.1 训练脚本（Unsloth 版）
在 RTX 5080 16GB 上微调 Qwen3-8B
变更：Unsloth 替换原生 transformers+peft，其余参数与 V2.3 保持一致作为基线
用法: PYTHONIOENCODING=utf-8 python training/v3.1/train.py
"""

import sys
if sys.platform == "win32":  # Unsloth 输出含 emoji，Windows GBK 编码会崩
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import json
from pathlib import Path
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# ==================== 可调参数 ====================
ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = str(ROOT / "models" / "Qwen3-8B")
TRAIN_DATA = str(ROOT / "data" / "v2" / "final" / "train.json")
EVAL_DATA = str(ROOT / "data" / "v2" / "final" / "eval.json")
OUTPUT_DIR = str(ROOT / "models" / "huchat-lora-v5")
LOG_DIR = str(ROOT / "logs" / "run5")

# LoRA 参数
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# 训练超参（与 V2.3 一致，作为 Unsloth 基线对比）
MAX_SEQ_LEN = 1024
BATCH_SIZE = 1
GRAD_ACCUM = 16            # 有效 batch = 16
LR = 1.5e-4
EPOCHS = 3
WARMUP_RATIO = 0.03
NEFTUNE_ALPHA = 5
SAVE_STEPS = 100
LOG_STEPS = 10
# ==================================================


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    print(f"模型路径: {MODEL_PATH}")
    print(f"训练数据: {TRAIN_DATA}")
    print(f"输出目录: {OUTPUT_DIR}")

    # ── Unsloth 加载模型（自带 4-bit 量化） ──
    print("加载模型（Unsloth 4-bit）...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,  # 自动检测，RTX 5080 会用 bfloat16
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Unsloth LoRA 配置 ──
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth 优化版，比原生省 30% 显存
    )
    model.print_trainable_parameters()

    # ── 加载数据集并转为文本 ──
    def messages_to_text(item):
        """将 messages 格式手动拼成 ChatML 文本"""
        parts = []
        for msg in item["messages"]:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        return {"text": "\n".join(parts)}

    train_raw = load_json(TRAIN_DATA)
    eval_raw = load_json(EVAL_DATA)
    train_ds = Dataset.from_list(train_raw).map(messages_to_text)
    eval_ds = Dataset.from_list(eval_raw).map(messages_to_text)
    print(f"训练集: {len(train_ds)} 条 | 验证集: {len(eval_ds)} 条")

    # ── 训练配置 ──
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_dir=LOG_DIR,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=False,
        report_to="none",
        save_total_limit=3,
        optim="adamw_8bit",  # Unsloth 推荐用 adamw_8bit
        max_length=MAX_SEQ_LEN,
        seed=42,
        neftune_noise_alpha=NEFTUNE_ALPHA,
    )

    # ── 开始训练 ──
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("=" * 50)
    print("开始训练！V3.1 (Unsloth)")
    print(f"  有效 batch size: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  总 epoch: {EPOCHS}")
    print(f"  最大序列长度: {MAX_SEQ_LEN}")
    print(f"  LoRA rank: {LORA_R}")
    print(f"  学习率: {LR}")
    print(f"  NEFTune alpha: {NEFTUNE_ALPHA}")
    print(f"  优化器: adamw_8bit (Unsloth)")
    print("=" * 50)

    trainer.train()

    # ── 保存 LoRA adapter ──
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nLoRA adapter 已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
