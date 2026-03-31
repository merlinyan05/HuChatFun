"""
step3_score.py — 质量评分过滤

输入：data/step2_segmented/*.jsonl
输出：data/step3_scored/scored.jsonl（所有 segment 打分后合并）
      data/step3_scored/passed.jsonl（过线的 segment）

评分维度（总分 100）：
  1. 口头禅密度（40分）：命中户晨风核心口头禅词越多越高
  2. 户晨风平均发言长度（30分）：平均字数越长，观点越实在
  3. 互动质量（20分）：某网友发言 / 户晨风发言比值，接近 1:1 最好
  4. 噪声惩罚（-10分上限）：仍含感谢/礼物类内容扣分

过线阈值：>= 50 分
"""

import json
import re
from pathlib import Path

INPUT_DIR = Path(__file__).parent.parent / "data" / "step2_segmented"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "step3_scored"

# 口头禅词表（权重：高频核心词 2 分，普通口头禅 1 分）
CATCHPHRASES = {
    "安卓": 2, "纯纯": 2, "购买力": 2, "前程似锦": 2,
    "你废了": 2, "废了": 1,
    "知道吧": 1, "能明白吗": 1, "明白吗": 1,
    "我告诉你": 1, "你给我记住": 1,
    "咱有什么说什么": 1, "实话实说": 1,
    "安卓逻辑": 2, "安卓人": 2, "安卓学历": 2,
    "低论": 1, "纯纯低论": 2,
    "爱莫能助": 1, "匪夷所思": 1,
    "生活愉快": 1, "不寒碜": 1,
}

# 噪声模式（扣分）
NOISE_PATTERN = re.compile(r"感谢.{0,15}(舰长|总|礼物)|谢谢.{0,10}(礼物|总)|读.*SC|超级SC")


def score_segment(lines: list[str]) -> dict:
    hu_lines = [l for l in lines if l.startswith("户晨风：")]
    wy_lines = [l for l in lines if l.startswith("某网友：")]

    hu_contents = [l[4:] for l in hu_lines]
    full_text = " ".join(lines)

    # 1. 口头禅密度（满分 40）
    phrase_score = 0
    phrase_hits = []
    for phrase, weight in CATCHPHRASES.items():
        count = full_text.count(phrase)
        if count > 0:
            phrase_score += min(count * weight, weight * 3)  # 单词最多算 3 次
            phrase_hits.append(phrase)
    phrase_score = min(phrase_score, 40)

    # 2. 户晨风平均发言长度（满分 30）
    if hu_contents:
        avg_len = sum(len(c) for c in hu_contents) / len(hu_contents)
        # 50字以上满分，线性插值
        length_score = min(avg_len / 50 * 30, 30)
    else:
        length_score = 0

    # 3. 互动质量（满分 20）
    if hu_lines and wy_lines:
        ratio = min(len(wy_lines), len(hu_lines)) / max(len(wy_lines), len(hu_lines))
        interaction_score = ratio * 20
    else:
        interaction_score = 0

    # 4. 噪声惩罚（最多扣 10 分）
    noise_count = len(NOISE_PATTERN.findall(full_text))
    noise_penalty = min(noise_count * 2, 10)

    total = phrase_score + length_score + interaction_score - noise_penalty

    return {
        "phrase_score": round(phrase_score, 1),
        "length_score": round(length_score, 1),
        "interaction_score": round(interaction_score, 1),
        "noise_penalty": round(noise_penalty, 1),
        "total": round(total, 1),
        "phrase_hits": phrase_hits,
        "hu_turns": len(hu_lines),
        "wy_turns": len(wy_lines),
        "avg_hu_len": round(sum(len(c) for c in hu_contents) / len(hu_contents), 1) if hu_contents else 0,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    for f in sorted(INPUT_DIR.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            scores = score_segment(rec["lines"])
            rec.update(scores)
            all_records.append(rec)

    # 写全量打分结果
    scored_path = OUTPUT_DIR / "scored.jsonl"
    with scored_path.open("w", encoding="utf-8") as out:
        for rec in all_records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 过线
    passed = [r for r in all_records if r["total"] >= 40]

    passed_path = OUTPUT_DIR / "passed.jsonl"  # 阈值 40
    with passed_path.open("w", encoding="utf-8") as out:
        for rec in passed:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 统计
    scores = [r["total"] for r in all_records]
    print(f"总 segment：{len(all_records):,}")
    print(f"过线（≥50）：{len(passed):,}（{len(passed)/len(all_records)*100:.1f}%）")
    print(f"分数分布：")
    for threshold in [30, 40, 50, 60, 70, 80]:
        count = sum(1 for s in scores if s >= threshold)
        print(f"  ≥{threshold}: {count:,} ({count/len(scores)*100:.1f}%)")
    print(f"\n平均分：{sum(scores)/len(scores):.1f}")
    print(f"最高分：{max(scores):.1f}")

    # 展示几个高分样本
    top = sorted(all_records, key=lambda x: x["total"], reverse=True)[:3]
    print(f"\n=== 高分样本 TOP 3 ===")
    for r in top:
        print(f"\n[{r['source']} seg{r['seg_id']}] 总分={r['total']} 口头禅={r['phrase_hits']}")
        for l in r["lines"][:6]:
            print(f"  {l[:80]}")


if __name__ == "__main__":
    main()
