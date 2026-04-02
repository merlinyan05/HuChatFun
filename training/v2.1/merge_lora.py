"""
合并 LoRA adapter 到基座模型，输出完整 fp16 权重
用法: python merge_lora.py
输出: models/huchat-merged/ (约 16GB safetensors)
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = str(ROOT / "models" / "Qwen3-8B")
LORA_ADAPTER = str(ROOT / "models" / "huchat-lora")
OUTPUT_DIR = str(ROOT / "models" / "huchat-merged")


def main():
    print(f"基座模型: {BASE_MODEL}")
    print(f"LoRA adapter: {LORA_ADAPTER}")
    print(f"输出目录: {OUTPUT_DIR}")

    # ── 加载 tokenizer ──
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER, trust_remote_code=True)

    # ── 加载基座模型（fp16，CPU）──
    # 合并需要完整精度，不能用 4-bit
    print("加载基座模型（fp16，CPU）... 需要约 16GB 内存")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # ── 加载并合并 LoRA ──
    print("加载 LoRA adapter 并合并...")
    model = PeftModel.from_pretrained(model, LORA_ADAPTER)
    model = model.merge_and_unload()
    print("合并完成")

    # ── 保存合并后的模型 ──
    print(f"保存到 {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("完成！")


if __name__ == "__main__":
    main()
