"""
step4_pairs.py — 构造训练对

输入：data/step3_scored/passed.jsonl（1,757 个高质量 segment）
输出：data/step4_pairs/train_pairs.jsonl

格式：Qwen ChatML 多轮对话
  {"messages": [
      {"role": "system", "content": "..."},
      {"role": "user",   "content": "..."},
      {"role": "assistant", "content": "..."},
      ...
  ]}

构造逻辑：
  1. 将 segment 的行按说话人分组，合并连续同说话人的行
  2. 要求序列以"某网友"开头，且至少有 2 个完整回合（user+assistant）
  3. 去掉开头不是某网友的行，直到第一个某网友出现
  4. 截断到最后一个完整的 assistant 回合（保证结尾是户晨风）
"""

import json
from pathlib import Path

INPUT_FILE = Path(__file__).parent.parent / "data" / "step3_scored" / "passed.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "step4_pairs"

SYSTEM_PROMPT = """你是户晨风，B站著名直播博主。说话风格直接、犀利、有观点，喜欢在连麦对话中一针见血地点出对方逻辑漏洞。

核心特点：
- 爱用"安卓"作万能贬义词（安卓人/安卓逻辑/安卓学历）
- "纯纯"作强调词（纯纯废了/纯纯安卓）
- 用"购买力"衡量一切
- "前程似锦"是反讽，意思是没希望
- 口头禅：「知道吧」「能明白吗」「我告诉你」「你给我记住了」「咱有什么说什么」
- 判定对方没救时说「你废了」
- 不绕弯子，直接给结论，有时候怼完还补一句关怀"""


def merge_turns(lines: list[str]) -> list[tuple[str, str]]:
    """
    将行列表合并为 (speaker, content) 的交替序列。
    连续同说话人的行合并为一条。
    返回格式：[("某网友", "内容"), ("户晨风", "内容"), ...]
    """
    turns = []
    for line in lines:
        if line.startswith("户晨风："):
            content = line[4:].strip()
            speaker = "户晨风"
        elif line.startswith("某网友："):
            content = line[4:].strip()
            speaker = "某网友"
        else:
            continue

        if not content:
            continue

        # 合并连续同说话人
        if turns and turns[-1][0] == speaker:
            turns[-1] = (speaker, turns[-1][1] + "　" + content)
        else:
            turns.append([speaker, content])

    return [(s, c) for s, c in turns]


def to_messages(turns: list[tuple[str, str]]) -> list[dict] | None:
    """
    将交替 turn 序列转为 ChatML messages 列表。
    要求：以某网友开头，至少 2 个完整回合，结尾是户晨风。
    """
    # 去掉开头不是某网友的 turn
    while turns and turns[0][0] != "某网友":
        turns = turns[1:]

    if not turns:
        return None

    # 截断到最后一个户晨风 turn 结尾
    while turns and turns[-1][0] != "户晨风":
        turns = turns[:-1]

    # 至少 2 个完整回合（4 条 turn）
    if len(turns) < 4:
        return None

    # 构造 messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for speaker, content in turns:
        role = "user" if speaker == "某网友" else "assistant"
        messages.append({"role": role, "content": content})

    return messages


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in INPUT_FILE.read_text(encoding="utf-8").splitlines()]

    total = len(records)
    kept = 0
    skipped = 0

    out_path = OUTPUT_DIR / "train_pairs.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for rec in records:
            turns = merge_turns(rec["lines"])
            messages = to_messages(turns)
            if messages is None:
                skipped += 1
                continue
            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            kept += 1

    print(f"输入 segment：{total:,}")
    print(f"输出训练样本：{kept:,}（跳过 {skipped} 个结构不合格的）")

    # 展示样本
    samples = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()]
    print(f"\n=== 样本展示（第1条）===")
    s = samples[0]
    for msg in s["messages"]:
        role = msg["role"]
        content = msg["content"][:120]
        print(f"  [{role}] {content}")

    # 统计轮数分布
    turn_counts = [len(json.loads(l)["messages"]) - 1 for l in out_path.read_text(encoding="utf-8").splitlines()]
    avg_turns = sum(turn_counts) / len(turn_counts)
    print(f"\n平均对话轮数（不含 system）：{avg_turns:.1f}")
    print(f"总 user/assistant 轮次：{sum(turn_counts):,}")


if __name__ == "__main__":
    main()
