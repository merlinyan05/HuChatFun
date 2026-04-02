"""
step2_segment.py — 结构化切分

输入：data/step1_cleaned/*.txt
输出：data/step2_segmented/*.jsonl
  每行是一个 JSON 对象：{"source": "2025-01-09", "seg_id": 3, "lines": [...]}

切分逻辑：
  按连麦边界切分。每次户晨风送走一个网友（说"下一个"/"再见"/"拜拜"/"生活愉快"
  /"前程似锦.*再见"等），当前 segment 结束，下一行开始新 segment。

过滤：
  - segment 内户晨风发言 < 3 轮 → 丢弃（太短，没有实质内容）
  - segment 内只有一个说话人 → 丢弃（缺少互动）
  - segment 总行数 > 60 → 截断到前 60 行（避免超长段混入多话题）
"""

import json
import re
from pathlib import Path

INPUT_DIR = Path(__file__).parent.parent / "data" / "step1_cleaned"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "step2_segmented"

# 送客信号：户晨风说这些词代表一段连麦结束
FAREWELL_PATTERN = re.compile(
    r"(下一个|再见|拜拜|生活愉快|前程似锦|挂了|不聊了|下一位|换下一个|好，谢谢你)"
)

MIN_HU_TURNS = 3   # segment 内户晨风最少发言轮数
MAX_LINES = 60     # segment 最大行数


def is_farewell(line: str) -> bool:
    """户晨风的这行是否含有送客信号"""
    if not line.startswith("户晨风："):
        return False
    return bool(FAREWELL_PATTERN.search(line[4:]))


def segment_file(lines: list[str]) -> list[list[str]]:
    """把一个文件的行列表切成多个 segment"""
    segments = []
    current = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        current.append(line)

        if is_farewell(line):
            segments.append(current)
            current = []

    # 最后一段（没有送客信号结尾）
    if current:
        segments.append(current)

    return segments


def is_valid(segment: list[str]) -> bool:
    """过滤低质量 segment"""
    hu_turns = sum(1 for l in segment if l.startswith("户晨风："))
    wangyou_turns = sum(1 for l in segment if l.startswith("某网友："))
    if hu_turns < MIN_HU_TURNS:
        return False
    if wangyou_turns == 0:
        return False
    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(INPUT_DIR.glob("*.txt"))
    print(f"输入文件：{len(txt_files)} 个")

    total_segs = 0
    kept_segs = 0

    for f in txt_files:
        lines = f.read_text(encoding="utf-8").splitlines()
        segments = segment_file(lines)
        source = f.stem  # e.g. "2025-01-09"

        valid_segments = []
        for seg in segments:
            total_segs += 1
            if not is_valid(seg):
                continue
            # 截断超长段
            seg = seg[:MAX_LINES]
            valid_segments.append(seg)
            kept_segs += 1

        if not valid_segments:
            continue

        out_path = OUTPUT_DIR / f.name.replace(".txt", ".jsonl")
        with out_path.open("w", encoding="utf-8") as out:
            for i, seg in enumerate(valid_segments):
                record = {"source": source, "seg_id": i, "lines": seg}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    ratio = kept_segs / total_segs * 100 if total_segs else 0
    print(f"总 segment：{total_segs:,}")
    print(f"保留 segment：{kept_segs:,}（{ratio:.1f}%）")
    print(f"输出文件：{len(list(OUTPUT_DIR.glob('*.jsonl')))} 个")


if __name__ == "__main__":
    main()
