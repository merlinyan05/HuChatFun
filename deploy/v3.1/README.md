# Deploy V3.1

模型版本：V3.1（V3 数据 + r=64，原框架 transformers+peft+trl）

## 部署步骤

```bash
# 1. 合并 LoRA
python training/v3.1/merge_lora.py

# 2. 转 GGUF f16
python tools/llama.cpp/convert_hf_to_gguf.py models/huchat-merged-v7 --outtype f16 --outfile models/huchat-v3.1-f16.gguf

# 3. 导入 Ollama
ollama create huchatfunV9 -f deploy/v3.1/Modelfile
```

## 推理参数

- temperature: 0.7
- top_p: 0.9
- repeat_penalty: 1.3
- num_predict: 150
