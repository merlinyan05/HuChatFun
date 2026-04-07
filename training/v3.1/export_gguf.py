"""
HuChatFun V3.1 导出脚本（Unsloth 版）
直接从 LoRA adapter 导出 f16 GGUF，不需要中间步骤
用法: PYTHONIOENCODING=utf-8 python training/v3.1/export_gguf.py
输出: models/huchat-merged-v5/
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
from unsloth import FastLanguageModel

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = str(ROOT / "models" / "Qwen3-8B")
LORA_ADAPTER = str(ROOT / "models" / "huchat-lora-v5")
OUTPUT_DIR = str(ROOT / "models" / "huchat-merged-v5")
GGUF_OUTPUT = str(ROOT / "models" / "huchat-merged-v5")

MAX_SEQ_LEN = 1024


def main():
    print(f"基座模型: {MODEL_PATH}")
    print(f"LoRA adapter: {LORA_ADAPTER}")

    # ── 加载模型 + LoRA ──
    print("加载模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LORA_ADAPTER,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )

    # ── 方式1：直接导出 GGUF（推荐，一步到位） ──
    print(f"导出 f16 GGUF 到 {GGUF_OUTPUT}/ ...")
    model.save_pretrained_gguf(
        GGUF_OUTPUT,
        tokenizer,
        quantization_method="f16",
    )
    print("GGUF 导出完成！")

    # ── 方式2（备用）：导出 merged safetensors ──
    # 如果 Ollama 加载 GGUF 有问题，可以改用这种方式再手动转
    # print(f"导出 merged 权重到 {OUTPUT_DIR}/ ...")
    # model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="merged_16bit")
    # print("merged 导出完成！")


if __name__ == "__main__":
    main()
