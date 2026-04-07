"""
HuChatFun V3.1 导出脚本
步骤1: Unsloth 合并 LoRA 到基座（输出 safetensors）
步骤2: llama.cpp 转 GGUF（f16）

用法:
  $env:PYTHONIOENCODING="utf-8"; python training/v3.1/export_gguf.py
输出:
  models/huchat-merged-v5/           (合并后的 safetensors)
  models/huchat-merged-v5/*.gguf     (f16 GGUF)
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import subprocess
from pathlib import Path
from unsloth import FastLanguageModel

ROOT = Path(__file__).resolve().parent.parent.parent
LORA_ADAPTER = str(ROOT / "models" / "huchat-lora-v5")
OUTPUT_DIR = str(ROOT / "models" / "huchat-merged-v5")
GGUF_FILE = str(ROOT / "models" / "huchat-merged-v5" / "huchatfun-v3.1-f16.gguf")
LLAMA_CPP_CONVERT = str(ROOT / "tools" / "llama.cpp" / "convert_hf_to_gguf.py")

MAX_SEQ_LEN = 1024


def main():
    print(f"LoRA adapter: {LORA_ADAPTER}")
    print(f"输出目录: {OUTPUT_DIR}")

    # ── 步骤1: Unsloth 合并 LoRA 权重 ──
    print("\n[步骤1] 加载模型并合并 LoRA...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LORA_ADAPTER,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="merged_16bit")
    print(f"合并完成: {OUTPUT_DIR}")

    # ── 步骤2: llama.cpp 转 GGUF ──
    print(f"\n[步骤2] 转换 GGUF: {GGUF_FILE}")
    cmd = [
        sys.executable, LLAMA_CPP_CONVERT,
        OUTPUT_DIR,
        "--outtype", "f16",
        "--outfile", GGUF_FILE,
    ]
    print(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nGGUF 转换失败（exit code {result.returncode}）")
        print("可以手动运行:")
        print(f"  python {LLAMA_CPP_CONVERT} {OUTPUT_DIR} --outtype f16 --outfile {GGUF_FILE}")
        sys.exit(1)

    print(f"\n导出完成: {GGUF_FILE}")


if __name__ == "__main__":
    main()
