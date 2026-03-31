"""
step0_explore.py — 语料格式探查

扫描 corpus/HuChenFeng-1.1/ 下所有 .md 文件，输出：
- 文件总数、INC 文件数
- 说话人标记的格式变体
- 每个文件的行数、对话轮数
- 各年份的文件数量分布
- 前几行样本（用于人工确认格式）
"""

import re
import os
from pathlib import Path
from collections import defaultdict

CORPUS_DIR = Path(__file__).parent.parent / "corpus" / "HuChenFeng-1.1"


def scan_file(filepath: Path) -> dict:
    text = filepath.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # 统计说话人标记
    speaker_lines = [l for l in lines if re.match(r"^.{1,10}：", l)]
    speakers = set()
    for l in speaker_lines:
        m = re.match(r"^(.{1,10})：", l)
        if m:
            speakers.add(m.group(1))

    hu_lines = [l for l in lines if l.startswith("户晨风：")]
    wangyou_lines = [l for l in lines if l.startswith("某网友：")]

    # 是否有时间戳（常见格式：[00:00:00] 或 00:00）
    has_timestamp = bool(re.search(r"\[\d{2}:\d{2}", text) or re.search(r"^\d{2}:\d{2}", text, re.MULTILINE))

    # 是否有 markdown 格式（标题/加粗）
    has_markdown = bool(re.search(r"^#{1,6} ", text, re.MULTILINE) or "**" in text)

    return {
        "total_lines": len(lines),
        "total_chars": len(text),
        "hu_turns": len(hu_lines),
        "wangyou_turns": len(wangyou_lines),
        "speakers": speakers,
        "has_timestamp": has_timestamp,
        "has_markdown": has_markdown,
        "first_line": lines[0][:80] if lines else "",
    }


def main():
    md_files = sorted(CORPUS_DIR.rglob("*.md"))
    # 跳过 README.md
    md_files = [f for f in md_files if f.name != "README.md"]

    inc_files = [f for f in md_files if "INC" in f.name]
    normal_files = [f for f in md_files if "INC" not in f.name]

    print(f"{'='*60}")
    print(f"语料概览")
    print(f"{'='*60}")
    print(f"总文件数:     {len(md_files)}")
    print(f"正常文件:     {len(normal_files)}")
    print(f"INC 文件:     {len(inc_files)}")

    # 按年份统计
    year_counts = defaultdict(int)
    for f in md_files:
        # 路径形如 2023年03月/2023-03-10.md
        year = f.parent.name[:4]
        year_counts[year] += 1

    print(f"\n{'='*60}")
    print("各年份文件数")
    print(f"{'='*60}")
    for year in sorted(year_counts):
        print(f"  {year}: {year_counts[year]} 个文件")

    # 扫描所有文件
    all_speakers = set()
    total_hu = 0
    total_wangyou = 0
    has_ts_count = 0
    has_md_count = 0
    results = []

    print(f"\n{'='*60}")
    print("逐文件扫描（进度）...")
    print(f"{'='*60}")

    for f in md_files:
        info = scan_file(f)
        results.append((f, info))
        all_speakers.update(info["speakers"])
        total_hu += info["hu_turns"]
        total_wangyou += info["wangyou_turns"]
        if info["has_timestamp"]:
            has_ts_count += 1
        if info["has_markdown"]:
            has_md_count += 1

    print(f"\n{'='*60}")
    print("说话人标记")
    print(f"{'='*60}")
    for s in sorted(all_speakers):
        print(f"  「{s}」")

    print(f"\n{'='*60}")
    print("内容统计")
    print(f"{'='*60}")
    print(f"户晨风发言总轮数: {total_hu}")
    print(f"某网友发言总轮数: {total_wangyou}")
    print(f"含时间戳的文件:   {has_ts_count} / {len(md_files)}")
    print(f"含 Markdown 的文件: {has_md_count} / {len(md_files)}")

    # 输出异常文件（说话人不是这两种的）
    unexpected = [(f, info) for f, info in results
                  if info["speakers"] - {"户晨风", "某网友"}]
    if unexpected:
        print(f"\n{'='*60}")
        print(f"异常说话人（非标准标记）— {len(unexpected)} 个文件")
        print(f"{'='*60}")
        for f, info in unexpected[:20]:
            extra = info["speakers"] - {"户晨风", "某网友"}
            print(f"  {f.name}: {extra}")

    # INC 文件样本
    print(f"\n{'='*60}")
    print("INC 文件样本（前3个，各取首行）")
    print(f"{'='*60}")
    for f in inc_files[:3]:
        info = dict(filter(lambda x: x[0] == "first_line",
                           scan_file(f).items()))
        first = f.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        print(f"  {f.name}: {first[:80]}")

    # 最短 / 最长文件
    results_sorted = sorted(results, key=lambda x: x[1]["hu_turns"])
    print(f"\n{'='*60}")
    print("极端文件（户晨风发言轮数）")
    print(f"{'='*60}")
    print("最少发言的5个文件:")
    for f, info in results_sorted[:5]:
        print(f"  {f.name}: {info['hu_turns']} 轮, {info['total_chars']} 字符")
    print("最多发言的5个文件:")
    for f, info in results_sorted[-5:]:
        print(f"  {f.name}: {info['hu_turns']} 轮, {info['total_chars']} 字符")

    print(f"\n{'='*60}")
    print("探查完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
