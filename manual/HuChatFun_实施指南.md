# HuChatFun — 完整实施指南

## 项目总览

| 项目 | 说明 |
|------|------|
| 目标 | 基于户晨风直播语料微调对话模型，复刻其说话风格 |
| 基座模型 | Qwen3.5-7B（Chat 版本） |
| 微调方法 | QLoRA（4-bit 量化 + LoRA Adapter） |
| 训练硬件 | RTX 5080 16GB VRAM |
| 推理部署 | Mac Mini M4 16GB，通过 Ollama 提供服务 |
| 最终接入 | OpenClaw 平台 |

---

## 第一阶段：语料采集与清洗

### 1.1 原始语料获取

直播文字稿的来源通常有几种路径：

- **字幕提取**：从录播视频中用 Whisper 提取音频并转文字
- **平台弹幕/字幕接口**：部分平台提供字幕回放
- **手动整理**：已有的文字稿文件

如果需要从视频中提取，推荐流程：

```bash
# 安装 whisper
pip install openai-whisper

# 从视频提取音频
ffmpeg -i livestream_video.mp4 -vn -acodec pcm_s16le -ar 16000 audio.wav

# Whisper 转录（large-v3 效果最好，medium 在 16G 显存下也可跑）
whisper audio.wav --model large-v3 --language zh --output_format txt
```

建议按场次/日期组织文件：

```
data/raw/
├── 2023-01-15_直播.txt
├── 2023-01-22_直播.txt
├── ...
└── 2025-03-20_直播.txt
```

### 1.2 语料清洗

清洗目标：去除噪声，保留户晨风本人的发言内容。

```python
# scripts/clean_corpus.py
import re
import json
from pathlib import Path

def clean_text(text: str) -> str:
    """清洗单条文本"""
    # 去除时间戳 [00:12:34] 或 00:12:34 格式
    text = re.sub(r'\[?\d{1,2}:\d{2}(:\d{2})?\]?', '', text)
    # 去除观众弹幕标记（根据实际格式调整）
    text = re.sub(r'【观众.*?】.*', '', text)
    text = re.sub(r'弹幕[:：].*', '', text)
    # 去除平台系统提示（礼物、进入直播间等）
    text = re.sub(r'(欢迎.*进入直播间|感谢.*送出|系统提示).*', '', text)
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_valid_utterance(text: str) -> bool:
    """过滤过短或无意义的片段"""
    if len(text) < 4:
        return False
    # 过滤纯语气词
    if re.fullmatch(r'[啊嗯哦哈呵嘿额呃哎唉嗨噢]+', text):
        return False
    return True

def process_transcript(filepath: Path) -> list[str]:
    """处理单个文字稿文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned = []
    for line in lines:
        line = clean_text(line)
        if line and is_valid_utterance(line):
            cleaned.append(line)
    return cleaned

# 批量处理
raw_dir = Path("data/raw")
output_dir = Path("data/cleaned")
output_dir.mkdir(parents=True, exist_ok=True)

for txt_file in sorted(raw_dir.glob("*.txt")):
    utterances = process_transcript(txt_file)
    out_path = output_dir / txt_file.name
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(utterances))
    print(f"{txt_file.name}: {len(utterances)} 条有效语句")
```

### 1.3 口头禅与风格标注（可选但推荐）

手动或半自动地标注户晨风的高频口头禅和标志性表达，后续用于评估模型是否学到了风格。

```python
# scripts/extract_catchphrases.py
from collections import Counter

# 根据实际情况列出已知口头禅种子词
KNOWN_PHRASES = [
    # 在这里填入户晨风的口头禅，例如：
    # "这个东西吧", "我跟你说", "购买力不行", ...
]

def find_catchphrases(corpus_dir: str, top_n: int = 50):
    """统计高频短语"""
    from pathlib import Path
    all_text = ""
    for f in Path(corpus_dir).glob("*.txt"):
        all_text += f.read_text(encoding='utf-8')

    # 3-8字的n-gram频率统计
    ngram_counter = Counter()
    for n in range(3, 9):
        for i in range(len(all_text) - n):
            ngram = all_text[i:i+n]
            if '\n' not in ngram:
                ngram_counter[ngram] += 1

    # 过滤低频
    phrases = [(p, c) for p, c in ngram_counter.most_common(500) if c >= 10]
    return phrases[:top_n]
```

---

## 第二阶段：训练数据构造

### 2.1 对话格式设计

QLoRA 微调对话模型需要将语料组织成多轮对话格式。核心思路：将户晨风的连续发言拆分成"用户提问 + 户晨风回答"的对话对。

**方案 A：基于话题分段的伪对话构造**

```python
# scripts/build_conversations.py
import json
import re
from pathlib import Path

def segment_into_turns(utterances: list[str], max_context: int = 3) -> list[dict]:
    """
    将连续语句转换为对话格式。
    策略：用前文作为上下文 prompt，后文作为 response。
    """
    conversations = []

    for i in range(1, len(utterances)):
        # 取前 max_context 句作为上下文
        context_start = max(0, i - max_context)
        context = '\n'.join(utterances[context_start:i])

        response = utterances[i]

        # 跳过太短的回复
        if len(response) < 10:
            continue

        conversations.append({
            "conversations": [
                {
                    "role": "user",
                    "content": context
                },
                {
                    "role": "assistant",
                    "content": response
                }
            ]
        })

    return conversations
```

**方案 B：用 LLM 辅助生成问答对（推荐，效果更好）**

用现有的大模型（如 Qwen 或 Claude API）将独白式语料改写为问答格式：

```python
# scripts/generate_qa_pairs.py
"""
用 LLM 将户晨风的独白段落转换为自然的问答对。
这一步质量直接决定微调效果，值得投入时间。
"""

SYSTEM_PROMPT = """你是一个数据标注助手。给你一段直播主播的发言，
请将其改写为一个自然的"观众提问 + 主播回答"的对话对。

要求：
1. 观众的问题要自然，像真实观众会问的
2. 主播的回答必须保留原文的核心内容、语气和口头禅
3. 不要添加原文没有的事实信息
4. 输出 JSON 格式：{"question": "...", "answer": "..."}
"""

# 调用示例（以 Anthropic API 为例）
import anthropic

client = anthropic.Anthropic()

def generate_qa(paragraph: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": paragraph}]
    )
    # 解析返回的 JSON
    import json
    return json.loads(response.content[0].text)
```

### 2.2 统一输出格式

最终训练数据统一为 Qwen 的 ChatML 格式：

```json
[
  {
    "conversations": [
      {
        "role": "system",
        "content": "你是户晨风（户子），一个数码科技领域的直播博主。你说话直接、幽默，喜欢用购买力来衡量产品价值，经常吐槽厂商的营销话术。回答时保持口语化风格。"
      },
      {
        "role": "user",
        "content": "户子，iPhone 16 值得买吗？"
      },
      {
        "role": "assistant",
        "content": "（这里是户晨风风格的回答）"
      }
    ]
  }
]
```

保存为 `data/train.json` 和 `data/eval.json`（建议 90/10 划分）。

### 2.3 数据量参考

| 数据规模 | 预期效果 |
|----------|----------|
| 500 条以下 | 能学到基本语气，但容易过拟合 |
| 1000-3000 条 | 风格较明显，推荐的起步量 |
| 5000-10000 条 | 风格稳定，能覆盖多种话题 |
| 10000+ 条 | 效果最佳，但需注意数据质量 |

---

## 第三阶段：训练环境搭建

### 3.1 环境配置（RTX 5080 训练机）

```bash
# 创建 conda 环境
conda create -n huchatfun python=3.11 -y
conda activate huchatfun

# 安装 PyTorch（确认 CUDA 版本匹配 5080）
# 5080 需要 CUDA 12.8+，确认驱动版本 >= 570
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 安装训练框架
pip install transformers>=4.46.0
pip install datasets
pip install accelerate
pip install peft>=0.13.0        # LoRA/QLoRA 支持
pip install bitsandbytes>=0.44  # 4-bit 量化
pip install trl>=0.12.0         # SFT Trainer
pip install wandb               # 训练监控（可选但推荐）

# 验证 GPU
python -c "import torch; print(torch.cuda.get_device_name(0)); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"
```

### 3.2 下载基座模型

```bash
# 方式一：huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3.5-7B-Chat --local-dir models/qwen3.5-7b-chat

# 方式二：modelscope（国内更快）
pip install modelscope
modelscope download --model Qwen/Qwen3.5-7B-Chat --local_dir models/qwen3.5-7b-chat
```

> **注意**：截至写作时 Qwen3.5 可能尚未发布，请根据实际情况选择 Qwen2.5-7B-Instruct 或 Qwen3-8B 等可用版本。核心流程不变。

---

## 第四阶段：QLoRA 微调训练

### 4.1 训练脚本

```python
# train.py
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ==================== 配置区 ====================
MODEL_PATH = "models/qwen3.5-7b-chat"
TRAIN_DATA = "data/train.json"
EVAL_DATA = "data/eval.json"
OUTPUT_DIR = "outputs/huchatfun-v1"

# ==================== 量化配置 ====================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # 嵌套量化，进一步省显存
)

# ==================== 加载模型 ====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # 5080 支持 FA2
)

model = prepare_model_for_kbit_training(model)

# ==================== LoRA 配置 ====================
lora_config = LoraConfig(
    r=64,                          # LoRA 秩，64 是风格学习的sweet spot
    lora_alpha=128,                # 通常设为 2*r
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",       # FFN
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 预期输出：trainable params: ~50M / total: ~7B (约 0.7%)

# ==================== 加载数据 ====================
dataset_train = load_dataset("json", data_files=TRAIN_DATA, split="train")
dataset_eval = load_dataset("json", data_files=EVAL_DATA, split="train")

# ==================== 数据格式化 ====================
def format_conversation(example):
    """将 conversations 列表格式化为 ChatML 模板"""
    formatted = ""
    for turn in example["conversations"]:
        role = turn["role"]
        content = turn["content"]
        if role == "system":
            formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return {"text": formatted}

dataset_train = dataset_train.map(format_conversation)
dataset_eval = dataset_eval.map(format_conversation)

# ==================== 训练参数 ====================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,                # 风格微调 3-5 轮通常足够
    per_device_train_batch_size=2,     # 16G VRAM 下的安全值
    gradient_accumulation_steps=8,     # 等效 batch_size = 16
    learning_rate=2e-4,               # QLoRA 推荐学习率
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,                         # 5080 原生支持 bf16
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    save_total_limit=3,
    max_seq_length=2048,              # 根据语料长度调整
    dataset_text_field="text",
    gradient_checkpointing=True,       # 省显存的关键
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="wandb",                 # 可改为 "none"
    run_name="huchatfun-v1",
)

# ==================== 开始训练 ====================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_eval,
    processing_class=tokenizer,
)

trainer.train()

# ==================== 保存 ====================
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print("训练完成！模型已保存到", f"{OUTPUT_DIR}/final")
```

### 4.2 显存占用估算（RTX 5080 16GB）

| 组件 | 显存占用 |
|------|----------|
| 模型权重（4-bit） | ~4 GB |
| LoRA 参数（bf16） | ~0.1 GB |
| 优化器状态 | ~0.4 GB |
| 激活值（seq_len=2048, bs=2） | ~6 GB |
| Gradient Checkpointing 节省 | -3 GB |
| **总计** | **~7.5 GB** |

余量充足，如显存允许可适当增大 batch_size 或 max_seq_length。

### 4.3 训练监控

```bash
# 启动训练
CUDA_VISIBLE_DEVICES=0 python train.py

# 如果用了 wandb，在浏览器查看训练曲线
# 重点关注：
# - train/loss 应稳步下降
# - eval/loss 不应大幅反弹（过拟合信号）
# - 如果 loss 在 epoch 2 后不再下降，考虑提前停止
```

---

## 第五阶段：模型合并与量化导出

### 5.1 合并 LoRA 权重到基座模型

```python
# scripts/merge_lora.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "models/qwen3.5-7b-chat"
LORA_PATH = "outputs/huchatfun-v1/final"
MERGED_PATH = "models/huchatfun-merged"

# 加载基座（全精度）
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)

# 加载 LoRA
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# 合并
model = model.merge_and_unload()

# 保存
model.save_pretrained(MERGED_PATH)
AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True).save_pretrained(MERGED_PATH)

print(f"合并完成，保存到 {MERGED_PATH}")
```

### 5.2 转换为 GGUF 格式（Ollama 需要）

```bash
# 克隆 llama.cpp（用于转换）
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# 安装 Python 依赖
pip install -r requirements.txt

# HF 格式 → GGUF
python convert_hf_to_gguf.py ../models/huchatfun-merged \
    --outfile ../models/huchatfun.gguf \
    --outtype f16

# 量化为 Q4_K_M（推荐，16G Mac 上表现最佳的平衡点）
./llama-quantize ../models/huchatfun.gguf \
    ../models/huchatfun-Q4_K_M.gguf Q4_K_M

# 也可以生成 Q5_K_M 或 Q6_K 作为备选
./llama-quantize ../models/huchatfun.gguf \
    ../models/huchatfun-Q5_K_M.gguf Q5_K_M
```

量化格式对比（7B 模型）：

| 格式 | 文件大小 | 推理速度（M4） | 质量损失 |
|------|----------|----------------|----------|
| Q4_K_M | ~4.1 GB | 快 | 极小 |
| Q5_K_M | ~4.8 GB | 中 | 更小 |
| Q6_K | ~5.5 GB | 较慢 | 几乎无 |
| Q8_0 | ~7.1 GB | 慢 | 无 |

Mac Mini M4 16GB 推荐 **Q4_K_M** 或 **Q5_K_M**。

---

## 第六阶段：Ollama 部署（Mac Mini M4）

### 6.1 安装 Ollama

```bash
# macOS
curl -fsSL https://ollama.com/install.sh | sh

# 验证
ollama --version
```

### 6.2 创建 Modelfile

```dockerfile
# Modelfile
FROM ./huchatfun-Q4_K_M.gguf

# 系统提示词——这是风格控制的核心
SYSTEM """你是户晨风（户子），一个数码科技领域的知名直播博主和测评人。你的说话风格有以下特点：
- 说话直接不绕弯，观点鲜明
- 喜欢用"购买力"来衡量产品是否值得买
- 经常吐槽厂商的营销话术和"遥遥领先"式宣传
- 口语化表达，带有北方口音的幽默感
- 评测产品时注重实际体验而非参数堆叠
- 对"智商税"产品深恶痛绝
请用户晨风的说话风格来回答所有问题。保持口语化，自然随意，像在直播间跟观众聊天一样。"""

# 推理参数
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

# ChatML 模板（与 Qwen 训练格式一致）
TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
```

### 6.3 创建并运行模型

```bash
# 将 GGUF 文件和 Modelfile 传到 Mac Mini
scp models/huchatfun-Q4_K_M.gguf user@mac-mini:~/models/
scp Modelfile user@mac-mini:~/models/

# 在 Mac Mini 上
cd ~/models
ollama create huchatfun -f Modelfile

# 测试运行
ollama run huchatfun

# 交互测试
>>> 户子，小米 SU7 Ultra 值得买吗？
>>> 你觉得苹果今年的 iPhone 挤牙膏了吗？
>>> 一万块预算配个电脑怎么搞？
```

### 6.4 开放 API 服务

```bash
# Ollama 默认在 localhost:11434 提供 API
# 如需局域网访问
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# API 调用示例
curl http://mac-mini:11434/api/chat -d '{
  "model": "huchatfun",
  "messages": [
    {"role": "user", "content": "华为 Mate 70 怎么样？"}
  ],
  "stream": false
}'
```

---

## 第七阶段：Prompt 工程调试

### 7.1 System Prompt 迭代策略

System Prompt 是低成本调整风格的关键手段，不需要重新训练。

```python
# scripts/prompt_ab_test.py
"""A/B 测试不同 system prompt 的效果"""
import requests
import json

PROMPTS = {
    "v1_基础": "你是户晨风，一个数码博主。用口语化风格回答。",

    "v2_详细": """你是户晨风（户子），数码科技直播博主。说话特点：
- 直接不绕弯，观点鲜明
- 用"购买力"衡量产品价值
- 吐槽营销话术
- 北方式幽默
像在直播间聊天一样回答。""",

    "v3_示例增强": """你是户晨风（户子）。
说话风格示例：
- "这手机三千块的购买力，你花五千买它，那不是当冤大头吗"
- "厂商说什么遥遥领先，你自己用用不就知道了吗"
请模仿这种风格回答所有问题。""",
}

TEST_QUESTIONS = [
    "iPhone 和华为怎么选？",
    "你觉得 AI 手机有用吗？",
    "两千块买什么耳机好？",
    "为什么大家都在卷折叠屏？",
]

def test_prompt(prompt_name, system_prompt):
    results = []
    for q in TEST_QUESTIONS:
        resp = requests.post("http://localhost:11434/api/chat", json={
            "model": "huchatfun",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
            ],
            "stream": False,
        })
        answer = resp.json()["message"]["content"]
        results.append({"question": q, "answer": answer})
    return results

# 运行测试
for name, prompt in PROMPTS.items():
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    for r in test_prompt(name, prompt):
        print(f"\nQ: {r['question']}")
        print(f"A: {r['answer'][:200]}...")
```

### 7.2 常见问题与调参

| 问题 | 调整方向 |
|------|----------|
| 回答太正式、像客服 | 降低 temperature 到 0.6，在 system prompt 加"像跟朋友聊天" |
| 回答太发散、跑题 | 降低 temperature 到 0.5，提高 repeat_penalty 到 1.15 |
| 口头禅出现频率不够 | 在 system prompt 中显式列出口头禅并要求使用 |
| 回答太短 | 在 system prompt 加"详细展开你的观点" |
| 编造产品信息 | 加入"如果不确定具体参数，就说不记得了"的指令 |

---

## 第八阶段：效果评估

### 8.1 自动评估指标

```python
# scripts/evaluate.py
import json
import re
from collections import Counter

def evaluate_style(responses: list[str], catchphrases: list[str]) -> dict:
    """评估模型输出的风格匹配度"""
    metrics = {}

    # 1. 口头禅命中率
    total_hits = 0
    for resp in responses:
        for phrase in catchphrases:
            if phrase in resp:
                total_hits += 1
    metrics["catchphrase_rate"] = total_hits / (len(responses) * len(catchphrases))

    # 2. 平均回复长度
    lengths = [len(r) for r in responses]
    metrics["avg_length"] = sum(lengths) / len(lengths)

    # 3. 口语化程度（语气词占比）
    colloquial_markers = ['吧', '嘛', '啊', '呢', '呗', '吗', '哈', '嘿', '诶']
    marker_count = sum(
        sum(1 for m in colloquial_markers if m in r)
        for r in responses
    )
    metrics["colloquial_score"] = marker_count / len(responses)

    # 4. 问号密度（反问是户晨风的特征之一）
    question_count = sum(r.count('？') for r in responses)
    metrics["rhetorical_question_rate"] = question_count / len(responses)

    return metrics
```

### 8.2 人工评估（盲测）

设计一个简单的盲测流程：

```python
# scripts/blind_test.py
"""
生成对比样本：原始 Qwen vs 微调后的 HuChatFun
让熟悉户晨风的人盲选哪个更像
"""
import random
import json

def generate_blind_test(questions: list[str]) -> list[dict]:
    tests = []
    for q in questions:
        # 获取两个模型的回答
        base_answer = query_model("qwen3.5-7b-chat", q)
        fine_tuned_answer = query_model("huchatfun", q)

        # 随机打乱顺序
        pair = [
            {"label": "A", "text": base_answer, "source": "base"},
            {"label": "B", "text": fine_tuned_answer, "source": "finetuned"},
        ]
        random.shuffle(pair)

        tests.append({
            "question": q,
            "response_A": pair[0]["text"],
            "response_B": pair[1]["text"],
            "_answer_key": {pair[0]["label"]: pair[0]["source"],
                           pair[1]["label"]: pair[1]["source"]},
        })
    return tests
```

### 8.3 评估维度打分表

| 维度 | 1分 | 3分 | 5分 |
|------|-----|-----|-----|
| 语气相似度 | 完全不像 | 偶尔有感觉 | 很像户晨风 |
| 口头禅使用 | 没有 | 有但不自然 | 自然融入 |
| 观点风格 | 模板化回答 | 有个性但不够 | 观点鲜明直接 |
| 知识准确性 | 大量编造 | 偶有错误 | 基本准确 |
| 幽默感 | 没有 | 偶尔有 | 很到位 |

---

## 第九阶段：接入 OpenClaw

### 9.1 API 适配层

OpenClaw 通常支持 OpenAI 兼容的 API 格式，Ollama 原生支持：

```bash
# Ollama 的 OpenAI 兼容端点
# POST http://mac-mini:11434/v1/chat/completions

curl http://mac-mini:11434/v1/chat/completions -d '{
  "model": "huchatfun",
  "messages": [
    {"role": "user", "content": "户子，推荐个两千块的手机"}
  ]
}'
```

### 9.2 OpenClaw 配置示例

```yaml
# openclaw_config.yaml（根据 OpenClaw 实际配置格式调整）
model:
  name: huchatfun
  provider: ollama
  endpoint: http://mac-mini:11434
  api_format: openai_compatible

persona:
  name: 户晨风AI
  avatar: assets/huchenchen_avatar.png
  description: 基于户晨风直播语料训练的AI助手（仅供娱乐）

settings:
  max_tokens: 2048
  temperature: 0.7
  stream: true

disclaimer: |
  本AI助手仅供娱乐，非户晨风官方项目。
  AI生成的内容不代表户晨风本人观点。
```

---

## 第十阶段：迭代优化

### 10.1 持续改进循环

```
收集反馈 → 分析bad case → 补充训练数据 → 重新微调 → 评估 → 部署
     ↑                                                    |
     └────────────────────────────────────────────────────┘
```

### 10.2 版本管理建议

```
outputs/
├── huchatfun-v1/          # 初版：基础语料
│   ├── final/
│   ├── training_args.json
│   └── eval_results.json
├── huchatfun-v2/          # 迭代：补充口头禅强化数据
├── huchatfun-v3/          # 迭代：加入购买力测评专项数据
└── experiments.md         # 每次实验的记录
```

### 10.3 进阶优化方向

1. **DPO/RLHF 对齐**：收集"更像/不像户晨风"的偏好数据，用 DPO 进一步优化风格
2. **RAG 增强**：接入产品数据库，让模型能查询真实的产品参数和价格
3. **多模态扩展**：如果基座模型支持，可以加入产品图片理解能力
4. **语音合成**：用 GPT-SoVITS 或 CosyVoice 克隆户晨风的声音，配合文本输出

---

## 项目文件结构总览

```
HuChatFun/
├── README.md
├── data/
│   ├── raw/                    # 原始直播文字稿
│   ├── cleaned/                # 清洗后的文本
│   ├── train.json              # 训练集
│   ├── eval.json               # 验证集
│   └── catchphrases.json       # 口头禅列表
├── scripts/
│   ├── clean_corpus.py         # 语料清洗
│   ├── generate_qa_pairs.py    # LLM辅助构造问答对
│   ├── build_conversations.py  # 对话格式构造
│   ├── extract_catchphrases.py # 口头禅提取
│   ├── merge_lora.py           # LoRA合并
│   ├── evaluate.py             # 自动评估
│   ├── blind_test.py           # 盲测
│   └── prompt_ab_test.py       # Prompt A/B测试
├── train.py                    # 主训练脚本
├── Modelfile                   # Ollama模型配置
├── models/                     # 模型文件（gitignore）
├── outputs/                    # 训练输出（gitignore）
├── openclaw_config.yaml        # OpenClaw部署配置
└── experiments.md              # 实验日志
```

---

> **免责声明**：本项目仅供个人学习和娱乐用途，与户晨风本人无关。AI 生成的所有内容不代表户晨风的真实观点。请勿用于商业用途或冒充本人。
