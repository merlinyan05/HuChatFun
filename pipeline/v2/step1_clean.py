"""
step1_clean.py — 粗切去噪

处理范围：corpus/HuChenFeng-1.1/2025年*/ 下的非 INC .md 文件
输出到：data/step1_cleaned/，每个源文件对应一个同名 .txt

去噪规则（激进模式）：
1. 跳过非对话文件（Preface / Acknowledgements / SUMMARY / videos / README）
2. 删除转录错误行（含 【API调用失败 等）
3. 删除纯礼物感谢行（`感谢xxx` / `谢谢xxx` 且行内无其他实质内容）
4. 删除户晨风发言中字数 < 15 的短行（"嗯" "对" "好的" "谢谢" 等填充语）
5. 某网友发言保留所有行（作为 user 侧，长短无所谓）
6. strip markdown 格式（# 标题、** 加粗）
7. 修复异常说话人标记（截断到第一个 ：）
"""

import re
from pathlib import Path

CORPUS_DIR = Path(__file__).parent.parent.parent / "corpus" / "HuChenFeng-1.1"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "v2" / "step1_cleaned"

# 非对话文件，直接跳过
SKIP_FILES = {"Preface.md", "Acknowledgements.md", "SUMMARY.md", "videos.md", "README.md"}

# 礼物感谢句：以感谢/谢谢开头，后面跟的是人名/ID（非汉字观点内容）
# 特征：感谢后面跟的是昵称/数字/英文，且行尾没有标点延伸的观点
THANKS_PATTERN = re.compile(
    r"^(户晨风：)?(感谢|谢谢)[^\u4e00-\u9fff，。！？,.!?]{0,20}$"
)

# 转录错误行
ERROR_PATTERN = re.compile(r"【API调用失败|语音转文字脚本|GitHub仓库地址")

# markdown strip
MD_HEADING = re.compile(r"^#{1,6}\s+")
MD_BOLD = re.compile(r"\*\*(.+?)\*\*")

# 标准说话人前缀
VALID_SPEAKERS = {"户晨风", "某网友"}


def fix_speaker(line: str) -> str:
    """修复说话人标记里混入了内容的异常行，如「某网友：我找到了：内容」"""
    m = re.match(r"^(.{1,10})：", line)
    if m:
        speaker = m.group(1)
        if speaker not in VALID_SPEAKERS:
            # 尝试截取到第一个合法说话人
            for s in VALID_SPEAKERS:
                if line.startswith(s):
                    return line
            # 说话人不在白名单，标记为待删除
            return ""
    return line


def is_noise(line: str) -> bool:
    """判断一行是否为噪声，应删除"""
    # 转录错误
    if ERROR_PATTERN.search(line):
        return True

    # 纯感谢行（整行）
    if THANKS_PATTERN.match(line):
        return True

    # 户晨风短行（去掉前缀后实际内容 < 15 字）
    if line.startswith("户晨风："):
        content = line[4:]
        if len(content) < 15:
            return True

    return False


def clean_line(line: str) -> str:
    """清洗单行文本"""
    # strip markdown
    line = MD_HEADING.sub("", line)
    line = MD_BOLD.sub(r"\1", line)
    line = line.strip()
    return line


def process_file(filepath: Path) -> list[str]:
    """处理单个文件，返回清洗后的行列表"""
    text = filepath.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = clean_line(line)
        if not line:
            continue

        # 修复异常说话人
        line = fix_speaker(line)
        if not line:
            continue

        # 去噪
        if is_noise(line):
            continue

        cleaned.append(line)

    return cleaned


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # V2: 取 2024+2025 年数据
    target_dirs = sorted(CORPUS_DIR.glob("2024年*")) + sorted(CORPUS_DIR.glob("2025年*"))
    if not target_dirs:
        print("未找到 2024年*/2025年* 目录，请检查路径")
        return

    all_files = []
    for d in target_dirs:
        for f in sorted(d.glob("*.md")):
            if f.name in SKIP_FILES:
                continue
            if "INC" in f.name:
                continue
            all_files.append(f)

    print(f"待处理文件：{len(all_files)} 个")

    total_in, total_out = 0, 0
    empty_files = []

    for f in all_files:
        lines_in = len([l for l in f.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()])
        cleaned = process_file(f)
        lines_out = len(cleaned)

        total_in += lines_in
        total_out += lines_out

        if lines_out == 0:
            empty_files.append(f.name)
            continue

        out_path = OUTPUT_DIR / f.name.replace(".md", ".txt")
        out_path.write_text("\n".join(cleaned), encoding="utf-8")

    ratio = total_out / total_in * 100 if total_in else 0
    print(f"输入行数：{total_in:,}")
    print(f"输出行数：{total_out:,}（保留 {ratio:.1f}%）")
    print(f"输出文件：{len(all_files) - len(empty_files)} 个")
    if empty_files:
        print(f"清洗后为空（跳过）：{empty_files}")


if __name__ == "__main__":
    main()
