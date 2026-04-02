# Deploy V2.2

模型权重：V2.2（`huchat-merged-v3/huchatfun-v3-f16.gguf`）

## 部署变体

### Modelfile（当前文件，对应旧 V3/V4）

```
temperature 0.7 | top_p 0.9 | repeat_penalty 1.3 | num_predict 150
```

### V2.2-deploy-b（旧 V5，当前最佳）

> **TODO**: 参数未入库，需要从 Win 训练机的 Ollama 导出后补录

### Modelfile.v1-mirostat（最早的参数，效果差已废弃）

```
mirostat 2 | mirostat_eta 0.1 | mirostat_tau 4.0 | repeat_penalty 1.8 | repeat_last_n 256 | num_predict 256
```
