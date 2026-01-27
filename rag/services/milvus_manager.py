#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milvus管理器（V6 K8s 专用稳定版）
修复: 超时问题 (Timeout) & 参数冲突
"""
import sys
import asyncio
import jieba 

from typing import List, Optional, Dict, Any

from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings

from llm.get_embedding import get_embed_model
from config import settings, get_collection_name
from utils.logger import logger

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
    
    def _connect_milvus(self, alias: str = "default"):
        """
        辅助函数：连接Milvus
        
        Args:
            alias: 连接别名
        """
        try:
            connections.connect(
                alias=alias,
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
                timeout=30
            )
            logger.info(f"✅ 成功连接到Milvus (alias: {alias})")
        except Exception as e:
            logger.error(f"❌ 连接Milvus失败 (alias: {alias}): {e}")
            raise
    
    def _disconnect_milvus(self, alias: str = "default"):
        """
        辅助函数：断开Milvus连接
        
        Args:
            alias: 连接别名
        """
        try:
            if connections.has_connection(alias):
                connections.disconnect(alias)
                logger.info(f"✅ 成功断开Milvus连接 (alias: {alias})")
        except Exception as e:
            logger.warning(f"⚠️ 断开Milvus连接失败 (alias: {alias}): {e}")
    
    def _get_milvus_schema_fields(self) -> List[FieldSchema]:
        """
        定义Milvus Collection的完整字段列表
        只包含核心RAG字段和公共元数据字段作为非动态字段
        两张表各自的字段通过动态字段处理，避免出现大量空字段
        """
        return [
            # 核心RAG字段 (所有数据共享，Milvus必需)
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=65535, is_primary=True, auto_id=False),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=65535, is_primary=False, nullable=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, is_primary=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.MILVUS_EMBEDDING_DIM, is_primary=False),
            
            # 公共元数据字段 (用于数据管理和筛选，Milvus标准实践)
            FieldSchema(name="source_table", dtype=DataType.VARCHAR, max_length=128, is_primary=False, nullable=True),
            FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=False, nullable=True),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=256, is_primary=False, nullable=True),
            FieldSchema(name="department", dtype=DataType.VARCHAR, max_length=128, is_primary=False, nullable=True),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=128, is_primary=False, nullable=True),
            FieldSchema(name="owner", dtype=DataType.VARCHAR, max_length=128, is_primary=False, nullable=True),
            FieldSchema(name="uploader", dtype=DataType.VARCHAR, max_length=128, is_primary=False, nullable=True),
            FieldSchema(name="sheet_name", dtype=DataType.VARCHAR, max_length=128, is_primary=False, nullable=True),
            FieldSchema(name="row_index", dtype=DataType.INT64, is_primary=False, nullable=True),
            # 注意：两张表各自的字段已移除，改为通过动态字段处理
            # 动态字段在创建Schema时已启用：enable_dynamic_field=True
        ]
    
    def _create_or_verify_collection_schema(self, collection_name: str):
        """
        确保Milvus Collection已经存在，并且其Schema与我们定义的Schema匹配
        
        Args:
            collection_name: 集合名称
        """
        logger.info(f"🔍 正在验证集合 {collection_name} 的Schema...")
        
        # 连接Milvus
        self._connect_milvus(alias="schema_verify")
        
        try:
            # 检查集合是否存在
            if not utility.has_collection(collection_name, using="schema_verify"):
                logger.info(f"📋 集合 {collection_name} 不存在，正在创建...")
                
                # 获取Schema字段
                fields = self._get_milvus_schema_fields()
                
                # 创建CollectionSchema，启用动态字段支持
                schema = CollectionSchema(fields=fields, description=f"Collection for {collection_name}", enable_dynamic_field=True)
                
                # 创建Collection
                collection = Collection(name=collection_name, schema=schema, using="schema_verify")
                logger.info(f"✅ 集合 {collection_name} 创建成功")
                
                # 立即为embedding字段创建向量索引
                logger.info(f"🔧 正在为embedding字段创建向量索引...")
                embedding_index_params = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 128}
                }
                collection.create_index(
                    field_name="embedding",
                    index_params=embedding_index_params,
                    sync=True
                )
                logger.info(f"✅ 向量索引创建成功")
                
                # 加载Collection
                collection.load()
                logger.info(f"✅ 集合 {collection_name} 加载成功")
            else:
                logger.info(f"📋 集合 {collection_name} 已存在，正在验证Schema...")
                
                # 获取现有集合
                existing_col = Collection(name=collection_name, using="schema_verify")
                existing_schema = existing_col.schema
                
                # 获取我们定义的Schema字段
                expected_fields = self._get_milvus_schema_fields()
                expected_field_dict = {field.name: field for field in expected_fields}
                existing_field_dict = {field.name: field for field in existing_schema.fields}
                
                # 检查字段差异
                missing_fields = set(expected_field_dict.keys()) - set(existing_field_dict.keys())
                extra_fields = set(existing_field_dict.keys()) - set(expected_field_dict.keys())
                type_mismatch_fields = []
                
                # 检查数据类型匹配
                for field_name, expected_field in expected_field_dict.items():
                    if field_name in existing_field_dict:
                        existing_field = existing_field_dict[field_name]
                        if expected_field.dtype != existing_field.dtype:
                            type_mismatch_fields.append((field_name, expected_field.dtype.name, existing_field.dtype.name))
                
                # 显示Schema验证结果
                if missing_fields or extra_fields or type_mismatch_fields:
                    logger.warning(f"⚠️ 集合 {collection_name} 的Schema与预期不符:")
                    if missing_fields:
                        logger.warning(f"   缺少字段: {missing_fields}")
                    if extra_fields:
                        logger.warning(f"   多余字段: {extra_fields}")
                    if type_mismatch_fields:
                        for field_name, expected_type, actual_type in type_mismatch_fields:
                            logger.warning(f"   类型不匹配 - {field_name}: 预期 {expected_type}, 实际 {actual_type}")
                    logger.warning("💡 建议: 请重建集合或更新Schema")
                else:
                    logger.info(f"✅ 集合 {collection_name} 的Schema验证通过")
        finally:
            # 断开连接
            self._disconnect_milvus(alias="schema_verify")
    
    def get_vector_store(self, department: str, overwrite: bool = False) -> MilvusVectorStore:
        collection_name = get_collection_name(department)
        
        # 1. 准备环境
        self._prepare_environment()
        
        logger.info(f"🔍 [Store] 准备连接 Milvus (K8s Mode)...")
        logger.info(f"   Host: {settings.MILVUS_HOST}")
        logger.info(f"   Port: {settings.MILVUS_PORT}")
        logger.info(f"   Collection: {collection_name}")

        try:
            # 1. 验证或创建集合Schema
            self._create_or_verify_collection_schema(collection_name)
            
            # ✅ V6 核心修正：
            # 1. 放弃 uri="http://...", 回归 host/port 模式，避免协议解析错误
            # 2. 增加 timeout=60，解决 K8s 10秒超时问题
            # 3. 显式 secure=False
            # 4. 根据 process.md 优化索引配置和字段定义
            # 5. 移除 overwrite 参数，因为我们手动管理Schema
            
            vector_store = MilvusVectorStore(
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
                uri="", # 显式留空，强制使用 Host/Port
                token=None, # 显式为 None
                collection_name=collection_name,
                dim=settings.MILVUS_EMBEDDING_DIM,
                enable_sparse=False,
                # 关键：增加超时设置
                timeout=60, 
                secure=False,
                # 优化索引配置，根据数据量调整
                index_config={
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 128}  # 优化nlist参数，根据数据量调整
                }
            )
            logger.info("✅ MilvusVectorStore 初始化成功")
            
            # 显式加载集合，避免查询时出现 collection not loaded 错误
            logger.info(f"📥 正在加载集合: {collection_name}")
            
            # 连接 Milvus
            self._connect_milvus(alias="load_collection")
            
            try:
                # 获取 Collection 对象
                coll = Collection(collection_name, using="load_collection")
                
                # 检查索引状态
                try:
                    indexes = coll.indexes
                    logger.info(f"🔍 集合索引信息: {[str(idx) for idx in indexes]}")
                except Exception as e:
                    logger.warning(f"⚠️ 获取索引信息失败: {e}")
                
                # 加载集合
                coll.load()
                logger.info(f"✅ 集合 {collection_name} 加载成功")
                
                # 为重要字段创建索引，加速过滤查询
                self._create_additional_indexes(coll)
            
            except Exception as e:
                logger.warning(f"⚠️ 集合加载失败 (将在查询时自动加载): {e}")
            finally:
                # 断开临时连接
                self._disconnect_milvus(alias="load_collection")
            
            return vector_store
        except Exception as e:
            logger.error(f"❌ MilvusVectorStore 初始化失败: {e}")
            logger.error("💡 提示: 请检查 K8s NodePort 防火墙或网络连通性")
            raise e
    
    def _create_additional_indexes(self, collection):
        """
        为重要字段创建索引，加速过滤查询
        根据 process.md 方案优化索引策略
        
        Args:
            collection: Milvus Collection 对象
        """
        logger.info(f"🔧 正在为集合 {collection.name} 创建附加索引...")
        
        try:
            # 获取所有字段
            all_fields = self._get_milvus_schema_fields()
            
            # 分离向量字段和标量字段
            vector_fields = []
            scalar_fields = []
            
            for field in all_fields:
                if field.dtype == DataType.FLOAT_VECTOR:
                    vector_fields.append(field)
                else:
                    scalar_fields.append(field)
            
            logger.info(f"📋 正在处理 {len(scalar_fields)} 个标量字段和 {len(vector_fields)} 个向量字段...")
            
            # 只处理标量字段，向量字段已在集合创建时处理
            for field in scalar_fields:
                logger.info(f"🔍 正在检查字段 {field.name} 的索引状态...")
                
                # 检查该字段是否已有索引
                has_index = False
                try:
                    # 获取集合的所有索引
                    indexes = collection.indexes
                    for idx in indexes:
                        if idx.field_name == field.name:
                            has_index = True
                            logger.info(f"   字段 {field.name} 已有索引，跳过创建")
                            break
                except Exception as e:
                    logger.warning(f"   检查索引状态失败: {e}")
                
                # 只有当字段没有索引时才创建
                if not has_index:
                    try:
                        # 根据字段类型选择合适的索引类型
                        if field.dtype in [DataType.VARCHAR, DataType.JSON, DataType.ARRAY]:
                            # 字符串、JSON和数组字段使用倒排索引
                            index_type = "INVERTED"
                        elif field.dtype in [DataType.INT64, DataType.FLOAT, DataType.BOOL]:
                            # 数值和布尔字段使用HASH索引，适合等值查询
                            index_type = "HASH"
                        else:
                            # 默认使用倒排索引
                            index_type = "INVERTED"
                        
                        # 创建索引参数
                        index_params = {
                            "index_type": index_type,
                            "params": {}
                        }
                        
                        # 创建索引
                        logger.info(f"   正在为字段 {field.name} 创建 {index_type} 索引...")
                        collection.create_index(
                            field_name=field.name,
                            index_params=index_params,
                            sync=True
                        )
                        logger.info(f"✅ 成功为字段 {field.name} 创建索引")
                    except Exception as e:
                        # 记录错误但不中断，继续处理其他字段
                        logger.warning(f"⚠️ 为字段 {field.name} 创建索引失败: {e}")
            
            logger.info("✅ 附加索引创建完成")
        
        except Exception as e:
            logger.error(f"❌ 创建附加索引失败: {e}")
    
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
                self._connect_milvus(alias="check")
                exists = utility.has_collection(get_collection_name(department), using="check")
                self._disconnect_milvus(alias="check")
                if not exists:
                    return None
            except Exception as e:
                logger.error(f"❌ 检查集合存在性失败: {e}")
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
            self._connect_milvus(alias="list")
            collections = utility.list_collections(using="list")
            self._disconnect_milvus(alias="list")
            return [col.replace("dept_", "") for col in collections if col.startswith("dept_")]
        except Exception as e:
            logger.error(f"❌ 获取部门列表失败: {e}")
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
            
            # 因为 nodes 已有 embedding，明确告知 VectorStoreIndex 不需要重新生成
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=None,  # 明确传递None，告知已包含embedding
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