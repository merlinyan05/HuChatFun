# V3 升级计划

> 回到 Win 主机（RTX 5080 16GB）时照着跑。
> 基于 2026-04-04 调研，当前最佳效果为 V2.2-deploy-b。

---

## 一、Unsloth 替换训练框架（优先级最高）

现有流程用的是原生 transformers + peft + trl，改用 Unsloth 可以 **2x 提速 + 70% 显存节省**，代码改动很小。

### 1.1 安装

```bash
# 创建新环境，避免污染现有的
conda create -n huchat-v3 python=3.11 -y
conda activate huchat-v3

# Unsloth 安装（RTX 5080 用 CUDA 12.x）
pip install unsloth
# 如果上面装不上，用官方指定源：
# pip install --upgrade --no-cache-dir "unsloth[cu124-ampere-torch250] @ https://unsloth.ai/whl/0.2/cu124-ampere-torch250"

# 验证
python -c "from unsloth import FastLanguageModel; print('OK')"
```

### 1.2 训练脚本改动

对比 `training/v2.3/train.py`，核心改动就两块：

**模型加载（替换）**：
```python
# ---- 旧 ----
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, ...)
tokenizer = AutoTokenizer.from_pretrained(model_name, ...)

# ---- 新 ----
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B-bnb-4bit",  # Unsloth 预量化版，下载更快
    max_seq_length=1024,
    load_in_4bit=True,
)
```

**LoRA 配置（替换）**：
```python
# ---- 旧 ----
from peft import LoraConfig, get_peft_model
peft_config = LoraConfig(r=64, lora_alpha=128, target_modules=[...], ...)
model = get_peft_model(model, peft_config)

# ---- 新 ----
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

**SFTTrainer 部分不用改**，trl 的 SFTTrainer 照用。

**推理前加一行**：
```python
FastLanguageModel.for_inference(model)  # 推理模式，速度翻倍
```

### 1.3 合并 & 导出

Unsloth 内置 GGUF 导出，不用再手动跑 llama.cpp：

```python
# 方式1：直接存 GGUF（推荐）
model.save_pretrained_gguf("models/huchat-v3", tokenizer, quantization_method="f16")

# 方式2：先存 merged 再转（和以前一样）
model.save_pretrained_merged("models/huchat-merged-v5", tokenizer)
```

---

## 二、试新基座模型（优先级中）

先用 Unsloth + Qwen3-8B 跑通，确认流程没问题后再换基座。

### 候选模型

| 模型 | Unsloth 名称 | 说明 |
|------|-------------|------|
| Qwen3-8B（当前） | `unsloth/Qwen3-8B-bnb-4bit` | 保底，先用这个跑通 |
| Qwen3.5-9B | `unsloth/Qwen3.5-9B-bnb-4bit` | 之前因混合注意力炸显存，Unsloth 可能已修复，值得重试 |
| Qwen4-7B | 待查 | 如果已发 stable 版，优先试 |
| DeepSeek-V3-7B 蒸馏 | 待查 | MIT 协议，中文强，备选 |

### 测试方法

每个模型用**相同数据**（V2 的 2667 条）训练 1 epoch，对比：
1. 训练 loss 收敛情况
2. 同一组测试问题的回答质量（口头禅密度 + 是否重复）
3. 显存占用和训练时间

---

## 三、DPO 二阶段（优先级低，等 SFT 效果稳定后再做）

SFT 只教模型"怎么说话"，DPO 教模型"哪种说法更好"。

### 流程

1. 用 SFT 模型对 eval 集的每个问题生成 5 条回答
2. 人工标注：选出"最像户晨风的"（chosen）和"最不像的"（rejected）
3. 用 trl 的 DPOTrainer 训练一轮

```python
from trl import DPOTrainer, DPOConfig

training_args = DPOConfig(
    output_dir="models/huchat-dpo-v1",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-5,  # DPO 用更低的 lr
    bf16=True,
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dpo_dataset,  # 含 prompt/chosen/rejected 字段
    tokenizer=tokenizer,
)
```

### 标注量

eval 集 297 条 × 每条选 1 chosen + 1 rejected = **297 对**，标注量不大，手动可控。

---

## 四、操作顺序

```
Step 1  在 Win 上装 Unsloth 环境
Step 2  把 training/v2.3/train.py 改成 Unsloth 版，存为 training/v3/train.py
Step 3  用 Qwen3-8B + V2 数据跑一版，确认 Unsloth 流程跑通
Step 4  对比 V2.2 效果（同样的测试问题）
Step 5  （可选）换 Qwen3.5 / Qwen4 基座重跑
Step 6  （可选）DPO 二阶段
```

---

## 五、注意事项

- Unsloth 的 GGUF 导出需要确认 Ollama 能正常加载，先小规模测试
- Qwen3.5-9B 之前在 Win 上炸过（见 `docs/troubleshooting.md`），如果 Unsloth 版还是炸就果断放弃，不要浪费时间
- 换基座后 system prompt 和推理参数（temperature/repeat_penalty 等）可能需要重新调
- V2.2-deploy-b 的 Modelfile 参数还没入库，先从 Win 机器导出再开始新实验
