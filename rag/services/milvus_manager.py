#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milvus管理器（V6 K8s 专用稳定版）
修复: 超时问题 (Timeout) & 参数冲突
"""
import logging
import sys
import asyncio
import jieba 

from typing import List, Optional

from pymilvus import connections, utility
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings

from llm.get_embedding import get_embed_model
from config import settings, get_collection_name

# 1. 配置日志
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. 全局设置
Settings.tokenizer = lambda text: list(jieba.cut(text))

class MilvusManager:
    """Milvus管理器"""
    
    def __init__(self):
        try:
            logger.info("🔧 [Init] 初始化 Embedding 模型...")
            self.embed_model = get_embed_model(
                model_name="text-embedding-v4",
                dim=settings.MILVUS_EMBEDDING_DIM,
                output_type="dense"
            )
            Settings.embed_model = self.embed_model
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            raise

    def _prepare_environment(self):
        """
        🚑 环境准备：修复 Asyncio Loop + 清理僵尸连接
        """
        # A. 修复 Celery Event Loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # B. 清理旧连接
        try:
            if connections.has_connection("default"):
                connections.disconnect("default")
        except Exception:
            pass 

    def get_vector_store(self, department: str, overwrite: bool = False) -> MilvusVectorStore:
        collection_name = get_collection_name(department)
        
        # 1. 准备环境
        self._prepare_environment()
        
        logger.info(f"🔍 [Store] 准备连接 Milvus (K8s Mode)...")
        logger.info(f"   Host: {settings.MILVUS_HOST}")
        logger.info(f"   Port: {settings.MILVUS_PORT}")
        logger.info(f"   Collection: {collection_name}")

        try:
            # ✅ V6 核心修正：
            # 1. 放弃 uri="http://...", 回归 host/port 模式，避免协议解析错误
            # 2. 增加 timeout=60，解决 K8s 10秒超时问题
            # 3. 显式 secure=False
            
            vector_store = MilvusVectorStore(
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
                uri="", # 显式留空，强制使用 Host/Port
                token=None, # 显式为 None
                collection_name=collection_name,
                dim=settings.MILVUS_EMBEDDING_DIM,
                overwrite=overwrite,
                enable_sparse=False,
                # 关键：增加超时设置
                timeout=60, 
                secure=False,
                index_config={
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 1024}
                },
            )
            logger.info("✅ MilvusVectorStore 初始化成功")
            
            # 显式加载集合，避免查询时出现 collection not loaded 错误
            logger.info(f"📥 正在加载集合: {collection_name}")
            from pymilvus import connections, Collection
            try:
                # 连接 Milvus
                connections.connect(
                    alias="load_collection",
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT,
                    timeout=10
                )
                # 获取 Collection 对象并加载
                coll = Collection(collection_name)
                coll.load()
                logger.info(f"✅ 集合 {collection_name} 加载成功")
            except Exception as e:
                logger.warning(f"⚠️ 集合加载失败 (将在查询时自动加载): {e}")
            finally:
                # 断开临时连接
                try:
                    connections.disconnect("load_collection")
                except:
                    pass
            
            return vector_store
        except Exception as e:
            logger.error(f"❌ MilvusVectorStore 初始化失败: {e}")
            logger.error("💡 提示: 请检查 K8s NodePort 防火墙或网络连通性")
            raise e
    
    def create_index_from_nodes(self, department: str, nodes: List, show_progress: bool = True) -> VectorStoreIndex:
        logger.info(f"🚀 [Index] 开始处理 {len(nodes)} 个节点...")
        
        self._prepare_environment()

        try:
            vector_store = self.get_vector_store(department)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            logger.info("⏳ [Index] 正在写入向量库 (Embedding)...")
            
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embed_model, 
                show_progress=show_progress,
            )
            
            logger.info(f"✅ 索引创建完成! 数据已存入 {get_collection_name(department)}")
            return index
            
        except Exception as e:
            logger.exception(f"❌ 创建索引失败: {e}")
            raise e

    def get_existing_index(self, department: str) -> Optional[VectorStoreIndex]:
        try:
            self._prepare_environment()
            
            # 手动检查集合 (使用 host/port 模式)
            try:
                connections.connect(
                    alias="check", 
                    host=settings.MILVUS_HOST, 
                    port=settings.MILVUS_PORT,
                    timeout=10
                )
                exists = utility.has_collection(get_collection_name(department), using="check")
                connections.disconnect("check")
                if not exists:
                    return None
            except:
                return None

            vector_store = self.get_vector_store(department)
            return VectorStoreIndex.from_vector_store(
                vector_store=vector_store, 
                embed_model=self.embed_model
            )
        except Exception:
            return None
    
    def list_departments(self) -> List[str]:
        self._prepare_environment()
        try:
            connections.connect(alias="list", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
            collections = utility.list_collections(using="list")
            connections.disconnect("list")
            return [col.replace("dept_", "") for col in collections if col.startswith("dept_")]
        except:
            return []
        
    def insert_nodes_with_embeddings(
        self, 
        department: str, 
        nodes: List,
        show_progress: bool = True
    ) -> VectorStoreIndex:
        """
        直接插入已有 embedding 的 nodes（不再重新生成）
        
        Args:
            department: 部门名称
            nodes: 已包含 embedding 的 TextNode 列表
            show_progress: 是否显示进度
        
        Returns:
            VectorStoreIndex
        """
        logger.info(f"💾 [Milvus] 开始写入 {len(nodes)} 个节点到 {department}")
        logger.info(f"   ⚡ 跳过 Embedding 生成（已预先计算）")
        
        self._prepare_environment()
        
        try:
            vector_store = self.get_vector_store(department)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # 因为 nodes 已有 embedding，VectorStoreIndex 不会重新生成
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                show_progress=show_progress,
            )
            
            logger.info(f"✅ [Milvus] 写入完成: {get_collection_name(department)}")
            return index
            
        except Exception as e:
            logger.exception(f"❌ 写入失败: {e}")
            raise e

try:
    milvus_manager = MilvusManager()
except Exception as e:
    logger.critical(f"MilvusManager 启动失败: {e}")
    milvus_manager = None