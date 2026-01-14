import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from llama_index.core import Settings
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import DashScopeEmbedding 

# ✅ 引入新的 Router
# 对应文件路径: /mnt/omicshub/rag/api/query.py
from api.query import router as query_router
from api.upload import router as documents_router 


# 引入日志配置 (假设您有这个工具，如果没有请改为 import logging)
from utils.logger import logger

# ==========================================
# 1. 全局模型初始化 (Global Init)
# ==========================================
# 这一步非常关键，query_service.py 中依赖 Settings.llm 和 Settings.embed_model
# ==========================================

API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 1.1 配置 LLM (通义千问)
try:
    Settings.llm = DashScope(
        model_name="qwen-plus", 
        api_key=API_KEY
    )
    logger.info("✅ [Global] LLM 配置成功: qwen-plus")
except Exception as e:
    logger.error(f"❌ [Global] LLM 配置失败: {e}")

# 1.2 配置 Embedding (文本向量)
try:
    Settings.embed_model = DashScopeEmbedding(
        model_name="text-embedding-v4",
        text_type="query",
        api_key=API_KEY
    )
    logger.info("✅ [Global] Embedding 配置成功: text-embedding-v4")
except Exception as e:
    logger.error(f"❌ [Global] Embedding 配置失败: {e}")


# ==========================================
# 2. 生命周期管理 (Lifespan)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 RAG 服务启动中...")
    # 这里可以添加 Milvus 连接预热等逻辑
    yield
    logger.info("🛑 RAG 服务关闭中...")


# ==========================================
# 3. App 定义与路由挂载
# ==========================================
app = FastAPI(
    title="RAG Search Service", 
    version="1.0.0",
    lifespan=lifespan
)

# ✅ 挂载路由
# prefix="/query" 配合 api/query.py 中的 router
# 最终访问地址: POST http://localhost:8088/query
app.include_router(query_router, prefix="/query")
app.include_router(documents_router, prefix="/documents", tags=["Documents"])

# 健康检查 (Ops 常用)
@app.get("/health")
async def health_check():
    return {"status": "ok", "module": "rag-service"}


if __name__ == "__main__":
    import uvicorn
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8088)