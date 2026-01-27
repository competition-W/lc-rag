import os
import uuid
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Settings
from llm.inference import DashScopeLLM
from llm.get_embedding import get_embed_model 
from starlette.websockets import WebSocketState

# ✅ 引入新的 Router
# 对应文件路径: /mnt/omicshub/rag/api/query.py
from api.query import router as query_router
from api.upload import router as documents_router 


# 引入日志配置 (假设您有这个工具，如果没有请改为 import logging)
from utils.logger import logger

# 从utils导入WebSocket连接管理器
from utils.websocket_manager import manager

# ==========================================
# 1. 全局模型初始化 (Global Init)
# ==========================================
# 这一步非常关键，query_service.py 中依赖 Settings.llm 和 Settings.embed_model
# ==========================================

# 从config中获取设置，已经包含了从.env加载的环境变量
from config import settings

# 1.1 配置 LLM (通义千问)
try:
    logger.info(f"🔍 从settings获取DASHSCOPE_API_KEY: {settings.DASHSCOPE_API_KEY}")
    Settings.llm = DashScopeLLM(
        model_name="qwen-plus", 
        api_key=settings.DASHSCOPE_API_KEY,
        max_tokens=4096  # 设置更大的max_tokens防止内容被截断
    )
    logger.info("✅ [Global] LLM 配置成功: qwen-plus")
except Exception as e:
    logger.error(f"❌ [Global] LLM 配置失败: {e}")
    import traceback
    logger.error(f"❌ LLM 配置失败堆栈信息: {traceback.format_exc()}")

# 1.2 配置 Embedding (文本向量)
try:
    Settings.embed_model = get_embed_model(model_name="text-embedding-v4")
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

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应更严格
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# WebSocket路由
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket 连接端点
    """
    # 为每个连接生成唯一ID
    connection_id = str(uuid.uuid4())
    logger.info(f"✅ 新的WebSocket连接请求 (ID: {connection_id})")
    await manager.connect(websocket, connection_id)
    
    # 建立连接后立即发送 connection_id
    await websocket.send_json({
        "type": "connection_info",
        "connection_id": connection_id
    })
    
    try:
        # 保持连接，处理客户端消息
        while websocket.client_state == WebSocketState.CONNECTED:
            # 接收客户端消息
            data = await websocket.receive_json()
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket断开 (ID: {connection_id})")
    except Exception as e:
        logger.error(f"WebSocket错误 (ID: {connection_id}): {str(e)}")
    finally:
        await manager.disconnect(connection_id)
        logger.info(f"❌ WebSocket连接已关闭 (ID: {connection_id})")


if __name__ == "__main__":
    import uvicorn
    # 启动服务，使用8090端口，避免与已占用的8088端口冲突
    uvicorn.run(app, host="0.0.0.0", port=8090)