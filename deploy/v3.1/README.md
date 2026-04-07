# Deploy V3.1

模型版本：V3.1（待训练）

## 状态

待 V3.1 训练完成后配置 Modelfile。

## 注意事项

- 如果换了基座模型，system prompt 和推理参数需要重新调
- Unsloth 内置 GGUF 导出，不再需要手动跑 llama.cpp
- 先用 f16 验证效果，确认好再考虑量化
