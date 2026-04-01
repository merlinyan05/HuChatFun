"""
HuChatFun QLoRA 训练脚本
在 RTX 5080 16GB 上微调 Qwen3-8B
用法: python train.py
"""

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# ==================== 可调参数 ====================
# 路径（相对于 training/ 目录运行）
MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "Qwen3-8B")
TRAIN_DATA = str(Path(__file__).resolve().parent.parent / "data" / "final" / "train.json")
EVAL_DATA = str(Path(__file__).resolve().parent.parent / "data" / "final" / "eval.json")
OUTPUT_DIR = str(Path(__file__).resolve().parent.parent / "models" / "huchat-lora")
LOG_DIR = str(Path(__file__).resolve().parent.parent / "logs" / "run1")

# LoRA 参数
LORA_R = 32               # LoRA 秩，32 对 9B 模型够用，显存友好
LORA_ALPHA = 64            # 通常设为 2×r，控制 LoRA 缩放
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # 注意力层

# 训练超参
MAX_SEQ_LEN = 1024         # Qwen3-8B 标准 Transformer，16GB 跑 1024 没问题
BATCH_SIZE = 1             # 单卡 batch，5080 16GB 只能放 1
GRAD_ACCUM = 16            # 梯度累积 → 有效 batch = 16
LR = 2e-4                  # QLoRA 标准学习率
EPOCHS = 3
WARMUP_RATIO = 0.03
SAVE_STEPS = 100
EVAL_STEPS = 100
LOG_STEPS = 10
# ==================================================


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    print(f"模型路径: {MODEL_PATH}")
    print(f"训练数据: {TRAIN_DATA}")
    print(f"输出目录: {OUTPUT_DIR}")

    # ── 4-bit NF4 量化配置 ──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # 二次量化，再省一点显存
    )

    # ── 加载 Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_SEQ_LEN

    # ── 加载模型（4-bit 量化） ──
    print("加载模型（4-bit 量化）...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ── LoRA 配置 ──
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 加载数据集 ──
    # 数据格式: [{"messages": [{"role": "system", ...}, {"role": "user", ...}, ...]}]
    # SFTTrainer 会自动检测 messages 列并应用 chat template
    train_ds = Dataset.from_list(load_json(TRAIN_DATA))
    eval_ds = Dataset.from_list(load_json(EVAL_DATA))
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
        eval_strategy="no",  # eval 在训练中 OOM（logits.float() 爆显存），训练完单独评估
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=False,
        report_to="none",
        save_total_limit=3,
        optim="paged_adamw_8bit",
        max_length=MAX_SEQ_LEN,
        seed=42,
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
    print("开始训练！")
    print(f"  有效 batch size: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  总 epoch: {EPOCHS}")
    print(f"  最大序列长度: {MAX_SEQ_LEN}")
    print(f"  LoRA rank: {LORA_R}")
    print("=" * 50)

    trainer.train()

    # ── 保存 LoRA adapter ──
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nLoRA adapter 已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
