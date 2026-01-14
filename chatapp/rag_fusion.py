#!/mnt/chatchat/.venv/bin python3
# -*- coding: utf-8 -*-

import os
import asyncio
import logging
from typing import List, Optional, Literal, AsyncIterable, Dict
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import uvicorn
from pymilvus import connections, utility, Collection

from llama_index.core import (
    Settings,
    load_index_from_storage,
    StorageContext,
    get_response_synthesizer,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank

# 业务配置导入
from llm.get_embedding import EMBED_MODEL
from llm.get_inference import get_dashscope_qwen_turbo
from db_connection.name_persist import map_reduce

# =========================
# ✅ 1. 高并发配置 (从压测版本移植)
# =========================
class Config:
    # 基础配置
    EMBED_DIM = 1024
    DEFAULT_URI = "http://110.1.122.1:30530"
    DEFAULT_TOP_K = 15
    DEFAULT_COLLECTION_NAME = "TOOL_CN_0628_dir5"
    DEFAULT_RERANK_TOP_K = 6
    
    # 模型配置
    DEFAULT_EMBED_MODEL = EMBED_MODEL
    DEFAULT_LLM = get_dashscope_qwen_turbo(model_name="qwen-plus")
    
    # 🚀 性能优化配置
    # 引擎池大小：建议设置为 CPU核心数 * 2 ~ 4。如果内存紧张，适当调小 (例如 16 或 32)
    ENGINE_POOL_SIZE = int(os.getenv("ENGINE_POOL_SIZE", "32"))
    
    # 最大排队数：超过此数量的请求将快速失败或等待，保护服务不被 OOM
    MAX_INFLIGHT = int(os.getenv("MAX_INFLIGHT", "200"))

# 初始化全局 LlamaIndex 设置
Settings.embed_model = Config.DEFAULT_EMBED_MODEL
Settings.llm = Config.DEFAULT_LLM

# =========================
# ✅ 2. 日志配置
# =========================
log_filename = f"./log_station/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
handlers = [logging.StreamHandler()]
handlers.append(logging.FileHandler(log_filename, mode='a')) # 如需写文件取消注释

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger("rag-service")

# =========================
# ✅ 3. 全局状态管理 (核心修改)
# =========================
# 检索器列表（只加载一次，所有 Engine 共享，节省内存）
retrievers = []

# 引擎池队列
engine_pool: asyncio.Queue = asyncio.Queue()

# 信号量（控制总并发）
rag_semaphore = asyncio.Semaphore(Config.MAX_INFLIGHT)

# 服务就绪状态
ready = False
shutting_down = False

# =========================
# ✅ 4. Milvus & Index 加载逻辑
# =========================
def connect_milvus_db(URI: str = Config.DEFAULT_URI):
    try:
        connections.connect(alias="default", uri=URI)
        logger.info("✅ Milvus Connected")
    except Exception as e:
        logger.error(f"❌ Milvus Connection Failed: {e}")

def load_storage_context(
        persist_dir: str, 
        collection_name: str = Config.DEFAULT_COLLECTION_NAME, 
        uri: str = Config.DEFAULT_URI, 
        dim: int = Config.EMBED_DIM):
    
    vector_store = MilvusVectorStore(
        uri=uri, 
        dim=dim, 
        overwrite=False, 
        collection_name=collection_name
    )
    
    try:
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.load() # 强制加载到内存
            logger.info(f"✅ [Milvus] 集合 {collection_name} 加载完成")
        else:
            logger.warning(f"⚠️ [Milvus] 集合 {collection_name} 不存在")
    except Exception as e:
        logger.error(f"❌ [Milvus] 加载集合 {collection_name} 失败: {str(e)}")

    return StorageContext.from_defaults(
        vector_store=vector_store, 
        persist_dir=persist_dir
    )

def build_retrievers(map_reduce_config: dict):
    """构建所有检索器，只执行一次"""
    global retrievers
    loaded_retrievers = []
    
    for collection_name, persist_dir in map_reduce_config.items():
        try:
            logger.info(f"📂 Loading: {collection_name} from {persist_dir}")
            storage_context = load_storage_context(
                persist_dir=persist_dir,
                collection_name=collection_name
            )
            index = load_index_from_storage(storage_context)
            loaded_retrievers.append(index.as_retriever())
        except Exception as e:
            logger.exception(f"❌ Failed to load {collection_name}: {e}")
    
    retrievers = loaded_retrievers
    logger.info(f"✅ Total Retrievers Loaded: {len(retrievers)}")

# =========================
# ✅ 5. QueryEngine 工厂 (核心优化)
# =========================
def create_query_engine_instance():
    """
    创建一个新的 QueryEngine 实例。
    每个请求从池子拿一个独立的 Engine，互不干扰。
    """
    global retrievers

    if not retrievers:
        # 防崩溃兜底
        logger.error("❌ No retrievers available!")
        class DummyRetriever:
            async def aretrieve(self, q): return []
        current_retrievers = [DummyRetriever()]
    else:
        current_retrievers = retrievers

    # ------------------------------------------------
    # 提示词定义 (保留原逻辑)
    # ------------------------------------------------
    how_to_gen_query = (
        "原始查询：{query}\n"
        "生成2个语义相同的变体查询，使用同义词或不同表达方式。"
    )
    
    query_send_template = PromptTemplate(
        "你是一个智能助手。以下是检索到的上下文信息：\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "用户问题: {query_str}\n\n"
        "请严格按照以下逻辑进行处理（这是最重要的指令）：\n\n"
        "【核心判断逻辑】\n"
        "1. **判断相关性**：首先判断上下文是否包含回答用户问题所需的信息。\n"
        "2. **如果相关**：基于上下文回答，不要编造。\n"
        "3. **如果不相关/上下文为空**（关键步骤）：\n"
        "   - **彻底忽略上下文**：不要在【回答摘要】和【详细说明】中提到“上下文没有相关信息”、“文档主要讨论了XX”或“无法评估”等内容。\n"
        "   - **直接回答问题**：直接调用你的通用知识库，针对用户问题本身，给出一个专业的、完整的答案（就像没有检索步骤一样）。\n"
        "   - **仅在最后说明**：只允许在【补充信息】栏目中说明知识库缺失的情况。\n\n"
        "【输出格式规范】\n"
        "1. 必须使用清晰的三段式结构（结论先行 → 关键要点 → 补充说明）。\n"
        "2. 格式要求：标题用【】，重点用◆，分项用●。严禁使用Markdown（如**粗体**）。\n"
        "3. 语气：专业、客观、直接。\n\n"
        "请按以下模板输出：\n"
        "【回答摘要】\n"
        "◆ （如果是通用回答，直接给出该领域的专业结论，不要说“无法回答”；如果是基于文档，概括文档结论）\n\n"
        "【详细说明】\n"
        "● （如果是通用回答，详细展开该领域的知识点，分点论述）\n"
        "● ...\n\n"
        "【补充信息】\n"
        "（仅在此处揭示来源：如果使用了通用知识，必须注明‘注：知识库中未找到关于[用户问题关键词]的文档，以上内容基于通用医学/科学常识生成，仅供参考。’）\n\n"
        "Answer: "
    )

    # ------------------------------------------------
    # 构建组件
    # ------------------------------------------------
    fusion_retriever = QueryFusionRetriever(
        retrievers=current_retrievers,
        similarity_top_k=5,     
        num_queries=2,           
        mode="simple",           
        use_async=True,          
        verbose=False,
        query_gen_prompt=how_to_gen_query,
    )

    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        streaming=True,
        text_qa_template=query_send_template
    )
    
    node_postprocessors = [
        # 1. 粗排过滤
        SimilarityPostprocessor(similarity_cutoff=0.1),
        # 2. 精排 (DashScope)
        DashScopeRerank(
            top_n=Config.DEFAULT_RERANK_TOP_K,
            model="gte-rerank-v2",
            return_documents=True
        )
    ]

    return RetrieverQueryEngine.from_args(
        retriever=fusion_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors,
        streaming=True,
    )

# =========================
# ✅ 6. 引擎池管理
# =========================
async def init_engine_pool():
    """初始化引擎池，填充多个 QueryEngine 实例"""
    logger.info(f"🔧 Initializing Engine Pool (Size: {Config.ENGINE_POOL_SIZE})...")
    
    # 清空旧数据
    while not engine_pool.empty():
        engine_pool.get_nowait()
        
    for i in range(Config.ENGINE_POOL_SIZE):
        engine = create_query_engine_instance()
        engine_pool.put_nowait(engine)
        if (i + 1) % 5 == 0:
            logger.info(f"   ... created {i + 1} engines")
            
    logger.info(f"✅ Engine pool ready: {engine_pool.qsize()}")

@asynccontextmanager
async def acquire_engine():
    """从池中获取 Engine 的上下文管理器"""
    engine = await engine_pool.get()
    try:
        yield engine
    finally:
        engine_pool.put_nowait(engine)

# =========================
# ✅ 7. FastAPI App & Lifespan
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ready, shutting_down

    logger.info("🚀 Service starting...")
    connect_milvus_db()

    # 1. 构建检索器 (只做一次)
    if not map_reduce:
        logger.warning("⚠️ map_reduce config is empty!")
    
    build_retrievers(map_reduce)
    
    if not retrievers:
        logger.error("❌ No Retrievers built! Service may not function correctly.")

    # 2. 初始化引擎池
    await init_engine_pool()
    
    ready = True
    logger.info("🎉 Service is READY to accept traffic")

    yield

    # 关闭逻辑
    shutting_down = True
    ready = False
    logger.info("🛑 Shutdown signal received.")
    try:
        connections.disconnect("default")
        logger.info("🔌 Milvus disconnected")
    except:
        pass

app = FastAPI(lifespan=lifespan)
router = APIRouter()

# =========================
# ✅ 8. API 模型与端点
# =========================
class BaseResponse(BaseModel):
    status: Literal["success", "error", "skipped"]
    data: Optional[dict] = None
    message: Optional[str] = None

@router.get("/health", response_model=BaseResponse)
async def health():
    if not ready:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return BaseResponse(
        status="success",
        data={
            "pool_size": Config.ENGINE_POOL_SIZE,
            "available_engines": engine_pool.qsize(),
            "retrievers_count": len(retrievers)
        }
    )

@router.post("/get_nodes", response_model=BaseResponse)
async def get_nodes(request: dict = Body(...)):
    """非流式检索节点接口"""
    try:
        query_text = request.get("query", "")
        user_id = request.get("user_id", "")
        
        if not query_text:
            return BaseResponse(status="error", message="No query provided")

        # 使用信号量和引擎池
        async with rag_semaphore:
            async with acquire_engine() as qe:
                nodes = await qe.aretrieve(query_text)
                
        return BaseResponse(
            status="success",
            data={
                "source_nodes": [node.model_dump() for node in nodes],
                "user_id": user_id
            }
        )
    except Exception as e:
        logger.exception("❌ get_nodes failed")
        return BaseResponse(status="error", message=str(e))

@router.post("/stream_chat", response_model=BaseResponse)
async def stream_chat_session(request: dict = Body(...)):
    """
    ✅ 核心接口：流式对话
    使用 Engine Pool + Semaphore 实现高并发
    """
    query_text = request.get("query", "")
    user_id = request.get("user_id", "unknown")
    
    if not query_text:
        return BaseResponse(status="error", message="No query provided")

    async def generate_responses() -> AsyncIterable[str]:
        if shutting_down:
            yield "Service is shutting down."
            return

        try:
            logger.info(f"[{user_id}] ⏳ Queueing request...")
            
            # 1. 限制并发数
            async with rag_semaphore:
                # 2. 获取独立引擎
                async with acquire_engine() as qe:
                    logger.info(f"[{user_id}] 🚀 Processing: {query_text[:30]}...")
                    
                    # 3. 执行查询
                    response = await qe.aquery(query_text)
                    
                    # 4. 流式输出
                    if hasattr(response, "response_gen"):
                        async for chunk in response.response_gen:
                            yield chunk
                    elif hasattr(response, "achat_stream"):
                        async for part in streaming_response.achat_stream:
                            yield part.delta
                    else:
                        yield str(response)
                        
            logger.info(f"[{user_id}] ✅ Done.")

        except asyncio.TimeoutError:
            yield "Server busy, please try again."
        except Exception as e:
            logger.exception(f"[{user_id}] ❌ Error")
            yield f"Error: {str(e)}"

    return StreamingResponse(
        generate_responses(),
        media_type="text/plain",
        headers={"X-Accel-Buffering": "no"} # 禁止 Nginx 缓冲
    )

app.include_router(router)

if __name__ == "__main__":
    # 注意：在 K8s 中 workers 由 deployment 控制，这里仅作为本地调试入口
    uvicorn.run(app, host="0.0.0.0", port=8848)
