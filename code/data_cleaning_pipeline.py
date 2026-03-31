"""
HuChatFun 语料清洗管线
=====================
专门针对户晨风直播文字稿设计。

核心策略：不是"还原"户晨风，而是做一个"浓缩版"。
- 砍掉 80% 的废话噪声（感谢礼物、等人、技术问题）
- 保留 20% 的金句、观点输出、连麦精华
- 通过 prompt 和数据双管齐下，让口头禅密度 > 真人

整个管线分 4 步：
1. 粗切：按说话人分段，去除系统噪声
2. 细筛：分类打标，只保留"有观点"的段落
3. 浓缩：提取金句，构造高密度训练对
4. 风格增强：用 LLM 改写，把口头禅密度拉满
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# 第一步：粗切 — 去噪声
# ============================================================

# 户晨风直播中的"废话模式"，这些占了 60-70% 的文字量
NOISE_PATTERNS = [
    # 1. 感谢礼物（最大噪声源）
    r'感谢[我]?.*?[总种].*?(?:感谢|十分感谢|再次感谢)',
    r'感谢[我]?.*?(?:的|啊).*?(?:舰长|提督|见长|SC|介绍|热气球|告白气球|飞天|探索|风之吟唱|爱神|梦奇|红衣|急速|灯火|爱的乐章|心愿)',
    r'(?:又有|有个?)(?:舰长|提督|见长)',
    r'哎呀.*?(?:总|x总).*?(?:破费|花钱|消费|厉害|太感谢)',

    # 2. 等人 / 开场废话
    r'好[，,]?不着急啊[，,]?不着急',
    r'稍安勿躁啊[，,]?稍安勿躁',
    r'咱们稍微等一下人啊[，,]?稍微等一下人',
    r'好[，,]?不急啊[，,]?不急',
    r'马上开始连[线麦]啊[，,]?马上开始连[线麦]',
    r'我现在把连麦[钮按].*?打开',
    r'大家重新申请一下',

    # 3. SC 读取过渡语
    r'好[，,]?读一下SC啊',
    r'好[，,]?再[读对]一下.*?SC',
    r'我先[去就]?看[微位私]',
    r'这个叫?xxxx总说',
    r'xxxx总说',

    # 4. 技术问题 / 卡顿
    r'怎么[有又]?卡[了呢]',
    r'安卓网.*?卡',
    r'网[速有]?[点太]?[卡慢]',
    r'先给[对他]方闭[个一]?麦',
    r'你太卡了.*?知道吧',

    # 5. 连麦操作指导
    r'点[最]?[左右上下].*?角.*?(?:加号|按钮|直播|关注)',
    r'你[能会知]道怎么开直播吗',
    r'输入我的名字户晨风',
    r'我给你点关注了',

    # 6. 重复性结尾感谢名单（xxxx总 x N）
    r'(?:xxxx总[、，,]?){3,}',

    # 7. 排队 / 催促
    r'要排队啊[，,]?[要]?排队',
    r'你别急[啊]?[，,]?要排队',
    r'好[，,]?下一个啊[，,]?下一个',
]

# 户晨风的"口头禅词库"——训练时要保留甚至强化的
CATCHPHRASES = {
    # 核心梗
    "安卓": "用于贬义形容一切低端/不好的事物",
    "苹果": "用于褒义形容一切高端/优质的事物",
    "购买力": "衡量产品/人的核心指标",
    "纯纯": "强调程度，如'纯纯低论'、'纯纯安卓'",
    "前程似锦": "反讽，实际意思是没什么希望",
    "爆赞": "表示认可",

    # 连麦常用
    "直接表达观点": "让连麦者直说",
    "速度": "催促对方",
    "别废话": "不耐烦时的催促",
    "你废了": "对连麦者的否定判断",
    "日本大专生": "贬低对方学历",
    "脱产学生": "批评不了解社会的学生",

    # 评价体系
    "低论": "没有道理的观点",
    "智商税": "不值得买的产品",
    "安卓网": "网络差",
    "安卓水": "不健康的饮料",
    "安卓显示器": "低端显示器",
    "安卓家庭": "家里用安卓产品的家庭",
    "安卓逻辑": "不讲道理的逻辑",

    # 过渡语（适量保留，不要全删）
    "不着急啊不着急": "安抚观众",
    "稍安勿躁": "安抚观众",
    "咱有什么说什么": "表示自己实话实说",
    "我跟你这么讲": "引出观点",
    "你给我记住了": "强调重点",
    "能明白吗": "确认对方理解",
    "知道吧": "口头禅收尾",
}


def rough_clean(text: str) -> str:
    """第一步粗切：去除明显噪声"""
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, '', text)
    # 去除连续空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 去除行首尾空白
    text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
    return text


# ============================================================
# 第二步：按说话人切分 + 分类
# ============================================================

@dataclass
class Utterance:
    speaker: str          # "户晨风" 或 "某网友"
    text: str
    category: str = ""    # 分类标签

@dataclass
class Conversation:
    """一次完整的连麦对话"""
    utterances: list[Utterance] = field(default_factory=list)
    topic: str = ""       # 话题标签
    quality: int = 0      # 质量评分 1-5

def split_by_speaker(text: str) -> list[Utterance]:
    """按说话人切分"""
    pattern = r'^(户晨风|某网友)：(.+?)(?=^(?:户晨风|某网友)：|\Z)'
    matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
    utterances = []
    for speaker, content in matches:
        content = content.strip()
        if content:
            utterances.append(Utterance(speaker=speaker, text=content))
    return utterances

def split_into_conversations(utterances: list[Utterance]) -> list[Conversation]:
    """
    将连续的户晨风+网友对话分成独立的连麦段落。
    判断依据：
    - 户晨风说"你好，请讲" / "直接表达观点" = 新对话开始
    - 户晨风说"好，再见" / "拜拜" = 对话结束
    - 连续多条户晨风独白（读SC）= 非对话段落
    """
    conversations = []
    current = Conversation()
    NEW_CONV_MARKERS = ['请讲', '你好', '说话', '直接表达观点', '喂']
    END_CONV_MARKERS = ['再见', '拜拜', '下一个', '还有什么想说的']

    for u in utterances:
        # 检测新对话开始
        if u.speaker == '户晨风' and any(m in u.text[:20] for m in NEW_CONV_MARKERS):
            if current.utterances:
                conversations.append(current)
            current = Conversation()

        current.utterances.append(u)

        # 检测对话结束
        if u.speaker == '户晨风' and any(m in u.text[-20:] for m in END_CONV_MARKERS):
            if len(current.utterances) >= 4:  # 至少 4 轮才算有效对话
                conversations.append(current)
            current = Conversation()

    if current.utterances:
        conversations.append(current)
    return conversations


# ============================================================
# 第三步：质量评估与分类
# ============================================================

# 话题分类关键词
TOPIC_KEYWORDS = {
    "数码测评": ["iPhone", "苹果", "安卓", "手机", "电脑", "耳机", "MacBook", "特斯拉",
                "华为", "OPPO", "小米", "三星", "单反", "相机", "路由器"],
    "人生建议": ["学历", "工作", "挣钱", "收入", "大专", "本科", "985", "211",
                "减肥", "体检", "血压", "医院"],
    "整理收纳": ["扔掉", "扔", "收拾", "整理", "房间", "架子", "玩偶"],
    "消费观": ["购买力", "山姆", "洗碗机", "德国厨具", "不锈钢", "筷子"],
    "职业规划": ["雅思", "留学", "英语", "电焊", "创业", "公司", "offer"],
    "价值观辩论": ["贩卖焦虑", "废了", "钱", "底线", "离婚", "出轨"],
    "房产": ["买房", "自建房", "租房", "学区房", "别墅"],
}

# 高价值内容的信号词（出现这些词的段落优先保留）
HIGH_VALUE_SIGNALS = [
    "我跟你这么讲", "我告诉你", "你给我记住了",
    "纯纯", "安卓", "购买力", "前程似锦",
    "你废了", "低论", "智商税",
    "不要去", "一定要", "绝对",
    "这是常识", "这是事实", "能明白吗",
    "我就跟你这样讲", "我就把话给你撂在这",
    "有一个算一个", "你不信咱们走着瞧",
]

def classify_conversation(conv: Conversation) -> Conversation:
    """给对话打标签和评分"""
    full_text = ' '.join(u.text for u in conv.utterances)

    # 话题分类
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in full_text for kw in keywords):
            conv.topic = topic
            break
    if not conv.topic:
        conv.topic = "其他"

    # 质量评分
    score = 0
    # 长度加分（太短没价值，太长可能是废话）
    hu_text = ' '.join(u.text for u in conv.utterances if u.speaker == '户晨风')
    if 100 < len(hu_text) < 2000:
        score += 1
    if 200 < len(hu_text) < 1000:
        score += 1  # 最佳长度额外加分

    # 高价值信号词加分
    signal_count = sum(1 for s in HIGH_VALUE_SIGNALS if s in full_text)
    score += min(signal_count, 3)  # 最多加 3 分

    # 有实质性观点输出加分
    if any(phrase in hu_text for phrase in ["我觉得", "我认为", "我建议", "我跟你讲"]):
        score += 1

    # 扣分项
    # 纯催促/等待/技术问题
    if hu_text.count("卡") > 3 or hu_text.count("等") > 5:
        score -= 2
    # 对方没说实质内容就挂掉的
    if len(conv.utterances) < 6:
        score -= 1

    conv.quality = max(1, min(5, score))
    return conv


# ============================================================
# 第四步：构造训练数据
# ============================================================

SYSTEM_PROMPT = """你是户晨风（户子），B站数码科技直播博主。你的风格特征：

【说话方式】
- 极其直接，从不绕弯子，上来就问关键信息：学历、收入、年龄、城市
- 口头禅密集：把"安卓"当万能贬义词，"苹果"当万能褒义词
- 喜欢说"纯纯XX""我告诉你""你给我记住了""能明白吗""知道吧"
- 用"购买力"衡量一切，用"前程似锦"反讽没希望的情况

【核心观点】
- 苹果生态 > 一切安卓，特斯拉 > 一切国产新能源
- 学历和收入高度关联，低学历不努力的人"纯废"
- 山姆 > 其他超市，德国厨具 > 一切，不锈钢 > 木头
- 洗碗机是必需品，不是奢侈品
- 年轻人要去大城市，不要在小地方待
- 不借钱给任何人，能拒绝别人是一种能力

【连麦风格】
- 快速连珠炮式提问：干什么的？多大？挣多少？几本？什么专业？
- 根据对方回答迅速给出判断，判断很绝对
- 对方磨叽立刻催促，废话多直接挂掉
- 对真正需要帮助的人很有耐心，给具体可执行的建议"""


def conversation_to_training_pair(conv: Conversation) -> Optional[dict]:
    """将一段连麦对话转换为训练数据"""
    if conv.quality < 3:
        return None

    hu_parts = []
    user_parts = []

    for u in conv.utterances:
        if u.speaker == '户晨风':
            hu_parts.append(u.text)
        else:
            user_parts.append(u.text)

    if not user_parts or not hu_parts:
        return None

    # 策略：把网友的发言合并为"用户输入"，户晨风的发言合并为"回答"
    # 但要按对话轮次组织，不能简单拼接
    turns = []
    current_role = None
    current_text = []

    for u in conv.utterances:
        role = "user" if u.speaker == "某网友" else "assistant"
        if role == current_role:
            current_text.append(u.text)
        else:
            if current_text:
                turns.append({"role": current_role, "content": '\n'.join(current_text)})
            current_role = role
            current_text = [u.text]
    if current_text:
        turns.append({"role": current_role, "content": '\n'.join(current_text)})

    # 确保以 user 开头
    if turns and turns[0]["role"] != "user":
        turns = turns[1:]
    # 确保 user/assistant 交替
    cleaned_turns = []
    last_role = None
    for t in turns:
        if t["role"] != last_role:
            cleaned_turns.append(t)
            last_role = t["role"]

    if len(cleaned_turns) < 2:
        return None

    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            *cleaned_turns
        ],
        "metadata": {
            "topic": conv.topic,
            "quality": conv.quality,
        }
    }


def extract_monologue_opinions(text: str) -> list[dict]:
    """
    从户晨风的独白段（读SC回复、非连麦观点输出）中提取训练数据。
    这些往往是最精华的观点输出。
    """
    results = []

    # 用高价值信号词定位观点段落
    sentences = re.split(r'[。！？\n]', text)
    buffer = []
    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 10:
            continue

        has_signal = any(s in sent for s in HIGH_VALUE_SIGNALS)
        if has_signal:
            buffer.append(sent)
        elif buffer:
            # 信号词后面的 1-2 句也保留（通常是论证）
            buffer.append(sent)
            if len(buffer) >= 3:
                opinion_text = '。'.join(buffer) + '。'
                # 反向构造一个合理的用户提问
                results.append({
                    "opinion": opinion_text,
                    "needs_question": True,  # 标记需要 LLM 生成配对问题
                })
                buffer = []
        else:
            buffer = []

    return results


# ============================================================
# 第五步：风格增强（让口头禅密度超过真人）
# ============================================================

STYLE_ENHANCEMENT_PROMPT = """你是一个数据标注助手。给你一段户晨风的原始发言，请改写成"浓缩版"。

改写规则：
1. 保留原文的核心观点和信息，不要添加编造的内容
2. 大幅增加以下口头禅的使用频率：
   - "安卓"作为贬义词（安卓逻辑、安卓人、安卓水平、安卓学历）
   - "纯纯"加强语气（纯纯低论、纯纯废了、纯纯安卓）
   - "购买力"衡量价值
   - "我告诉你"引出观点
   - "知道吧""能明白吗"收尾
   - "前程似锦"反讽
3. 让语气更加斩钉截铁，去掉犹豫和客套
4. 保持口语化，像直播聊天而不是写文章
5. 适当加入"苹果 vs 安卓"的类比

示例：
原文：你这个工作收入不高，建议你考虑换一个行业。
改写：你一个月挣这点钱，纯纯安卓收入，我告诉你这个行业你待下去就是纯废。你趁年轻赶紧换，别搁这耗着了，耗到最后前程似锦，知道吧？

输出改写后的文本，不要加任何解释。"""


def generate_enhanced_qa(original_opinion: str, question: str = None) -> dict:
    """
    用 LLM 生成风格增强后的问答对。
    
    这个函数需要调用 LLM API，这里只给出模板。
    实际使用时接入 Claude API 或 Qwen API。
    """
    # 如果没有配对问题，先生成一个
    if question is None:
        gen_question_prompt = f"""给你一段户晨风的观点输出，请生成一个自然的观众提问。
要求：像真实观众在直播间会问的问题，简短口语化。

户晨风的观点：{original_opinion}

只输出问题，不要其他内容。"""
        # question = call_llm(gen_question_prompt)
        question = "[需要 LLM 生成]"

    # 风格增强
    enhance_prompt = f"""{STYLE_ENHANCEMENT_PROMPT}

原文：{original_opinion}"""
    # enhanced = call_llm(enhance_prompt)
    enhanced = "[需要 LLM 增强]"

    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": enhanced},
        ]
    }


# ============================================================
# 主流程
# ============================================================

def process_transcript(filepath: str) -> dict:
    """处理单个文字稿文件的完整流程"""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    stats = {"raw_chars": len(raw_text)}

    # 1. 粗切去噪
    cleaned = rough_clean(raw_text)
    stats["after_rough_clean"] = len(cleaned)
    stats["noise_removed_pct"] = f"{(1 - len(cleaned)/len(raw_text))*100:.1f}%"

    # 2. 按说话人切分
    utterances = split_by_speaker(cleaned)
    stats["total_utterances"] = len(utterances)
    stats["hu_utterances"] = sum(1 for u in utterances if u.speaker == '户晨风')

    # 3. 分成独立对话
    conversations = split_into_conversations(utterances)
    stats["total_conversations"] = len(conversations)

    # 4. 分类打分
    conversations = [classify_conversation(c) for c in conversations]
    stats["high_quality"] = sum(1 for c in conversations if c.quality >= 3)
    stats["quality_distribution"] = {
        q: sum(1 for c in conversations if c.quality == q)
        for q in range(1, 6)
    }

    # 5. 生成训练数据
    training_data = []
    for conv in conversations:
        pair = conversation_to_training_pair(conv)
        if pair:
            training_data.append(pair)
    stats["training_pairs"] = len(training_data)

    # 6. 提取独白观点（户晨风自己讲的长段落）
    hu_monologues = [u.text for u in utterances
                     if u.speaker == '户晨风' and len(u.text) > 200]
    opinions = []
    for mono in hu_monologues:
        opinions.extend(extract_monologue_opinions(mono))
    stats["extracted_opinions"] = len(opinions)

    return {
        "stats": stats,
        "training_data": training_data,
        "opinions_for_enhancement": opinions,
        "conversations": conversations,
    }


# ============================================================
# 手动标注的"金句库"模板
# ============================================================

GOLDEN_QUOTES = """
以下是从语料中手动挑选的高价值金句，直接作为训练数据的 assistant 回答使用。
每条金句配一个合理的用户提问。

注意：这个列表需要你手动从语料中挑选并填充。
下面是从你提供的两条语料中我识别到的高价值片段示例：

---

Q: 户子，苹果手机拍照到底好在哪？
A: 我最后再说一次这个拍照的问题。所有的手机的拍照都在跟iPhone比，只有iPhone的拍照在跟肉眼比，你记住这句话了。iPhone是真的在拿自己手机拍的照在跟真实世界比，其他家都在跟iPhone比，这就是区别，这就是苹果 designed by Apple in California。

---

Q: 家里筷子用什么材质的好？
A: 把你们家所有的木筷子全部扔掉，我告诉你，我说完这个你还得谢我。木筷子中间那个缝里面长时间潮湿会发霉，黄曲霉素，吃到胃里面引发胃癌。这不是我给人瞎说，这是医学常识。扔掉之后用304不锈钢的筷子，又卫生又容易清洁。有人说不锈钢筷子烫嘴，这是好事，是在提醒你不要吃烫的食物，保护你的食道，你应该心怀感恩知道吗？

---

Q: 电焊工怎么提高收入？
A: 你今天来对地方了。电焊加雅思6.5大于等于985。你给我记住了。你会电焊有手艺，再加上雅思6.5，直接起飞我告诉你，985都不如你。另外你是靠电焊吃饭的，一定要买好的防护用品，去买3M的或者霍尼维尔的，别的不买。你是靠这个养家糊口的，别怕花钱。

---

Q: ASML和中兴国际的offer怎么选？
A: 你犹豫都是对ASML这个世界上最强光刻机制造企业的不尊重。在差不多情况下一定是ASML。你在行业排名第一的企业干过，去其他公司一定是升职加薪。但你在非头部公司干，想去头部公司很难。就算你想去中兴国际，也先去ASML干两年再去，比你直接去中兴国际发展更好、工资更高。这个叫曲线去中兴国际，今天你来我这不白来。

---

Q: 怎么看年轻人去菜市场买菜？
A: 年轻人不要去菜市场买菜。第一，见人下菜，卖菜的看你是年轻人，菜价立马比卖给别人高50%甚至一倍。第二，缺斤少两，太普遍了。你去一次吃亏一次，买的菜又贵又垃圾。你就去连锁超市，哪怕是永辉大润发都行，不给你来那种见人下菜缺斤少两。经常买菜的都知道，菜市场的菜反而贵，超市的反而便宜。

---

Q: 要不要在农村老家建自建房？
A: 不要在农村建自建房，除非父母在农村常住。建了房子没人住你放在那干什么呢？又不会保值升值，纯纯浪费钱。你存银行还能有个利息呢。你买房去省会去大城市，小县城的房子没有任何价值，年轻人都走了。大城市会持续人口流入，小地方纯纯人口流失。
"""


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python data_cleaning_pipeline.py <transcript_file>")
        print("\n示例:")
        print("  python data_cleaning_pipeline.py data/raw/2025-09-01_直播.txt")
        print("\n也可以直接导入使用:")
        print("  from data_cleaning_pipeline import process_transcript")
        sys.exit(0)

    filepath = sys.argv[1]
    result = process_transcript(filepath)

    print("\n" + "="*60)
    print("清洗统计")
    print("="*60)
    for k, v in result["stats"].items():
        print(f"  {k}: {v}")

    # 保存训练数据
    output_path = filepath.replace('.txt', '_training.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result["training_data"], f, ensure_ascii=False, indent=2)
    print(f"\n训练数据已保存到: {output_path}")
    print(f"共 {len(result['training_data'])} 条训练样本")
