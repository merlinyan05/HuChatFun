"""
step6_export.py — 输出 train.json / eval.json

输入：data/step4_pairs/train_pairs.jsonl
输出：data/final/train.json（90%）
      data/final/eval.json（10%）

随机打乱后按 9:1 切分，固定 seed 保证可复现。
"""

import json
import random
from pathlib import Path

INPUT_FILE = Path(__file__).parent.parent / "data" / "step4_pairs" / "train_pairs.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "final"

SEED = 42
EVAL_RATIO = 0.1


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in INPUT_FILE.read_text(encoding="utf-8").splitlines()]

    random.seed(SEED)
    random.shuffle(records)

    split = int(len(records) * (1 - EVAL_RATIO))
    train = records[:split]
    eval_ = records[split:]

    train_path = OUTPUT_DIR / "train.json"
    eval_path = OUTPUT_DIR / "eval.json"

    train_path.write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    eval_path.write_text(json.dumps(eval_, ensure_ascii=False, indent=2), encoding="utf-8")

    # 统计
    train_turns = sum(len(r["messages"]) - 1 for r in train)
    eval_turns = sum(len(r["messages"]) - 1 for r in eval_)

    print(f"总样本：{len(records):,}")
    print(f"train：{len(train):,} 条（{train_turns:,} 轮）")
    print(f"eval： {len(eval_):,} 条（{eval_turns:,} 轮）")
    print(f"输出：{train_path}")
    print(f"输出：{eval_path}")


if __name__ == "__main__":
    main()
