# 0729-2025, 每个服务都应该配备一个操作指南
# 项目简介
这是为AILab设计的单轮对话-多语料库检索服务，rag_fusion_ctrl.py
需要使用uv启动，uvicorn rag_fusion_ctrl:app --host 0.0.0.0 --port 8848

网页端每次提问，会由Qwen3-32b模型进行意图识别，当需要检索知识库的时候，调用此服务

## 运行此服务的容器
debian 7的轻量化版本
需要连入milvus-standalone的网络milvus
同一network下，使用http://milvus-standalone:19530访问数据库

跨服务器时，需要内网，访问容器

## 运行逻辑
