#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置管理
"""
import os
from pydantic import Field
from enum import Enum
from typing import Dict, List, Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict



class DocumentType(str, Enum):
    """支持的文档类型"""
    EXCEL = "excel"
    PDF = "pdf"
    MARKDOWN = "markdown"
    WORD = "word"

# ==================== 权限控制配置 ====================
class PermissionMode(str, Enum):
    """权限模式"""
    DISABLED = "disabled"    # 禁用权限（开发阶段）
    BASIC = "basic"          # 基础模式（信任请求头）
    STRICT = "strict"        # 严格模式（验证Token）

class Settings(BaseSettings):
    """全局配置"""
    
    # ==================== 服务配置 ====================
    APP_NAME: str = "RAG Document Service"
    APP_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8088
    DEBUG: bool = False

    # ==================== 日志配置 ====================
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"  # 日志保存目录
    LOG_ROTATION: str = "500 MB"  # 日志轮转大小
    LOG_RETENTION: str = "30 days"  # 日志保留时间

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # ==================== LLM 配置====================
    DASHSCOPE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    OPENAI_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    LLM_PROVIDER: Literal["dashscope", "openai", "custom"] = "dashscope"
    EMBEDDING_PROVIDER: Literal["dashscope", "openai", "custom"] = "dashscope"
    
    # Embedding配置
    EMBEDDING_MODEL: str = "text-embedding-v4"  # dashscope
    EMBEDDING_DIM: int = 1024
    
    # LLM 配置
    LLM_MODEL: str = "qwen-plus"  # dashscope
    
    # Rerank配置
    RERANK_MODEL: str = "gte-rerank-v2"
    RERANK_TOP_N: int = 6
    
    # ==================== MinIO 配置 ====================
    MINIO_ENDPOINT: str = "110.1.122.1:30900"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    
    MINIO_BUCKET_DOCUMENTS: str = "rag-documents"
    MINIO_BUCKET_PROCESSED: str = "rag-processed"
    
    # ==================== Milvus 配置 ====================
    MILVUS_HOST: str = "110.1.122.1"
    MILVUS_PORT: int = 30530
    MILVUS_USER: Optional[str] = None
    MILVUS_PASSWORD: Optional[str] = None
    MILVUS_EMBEDDING_DIM: int = 1024
    
    # ==================== Redis 配置 ====================
    REDIS_HOST: str = "redis" 
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    
    # ==================== Celery 配置 ====================
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/1"
    
    # ==================== 文档处理配置 ====================
    # 文件类型映射（扩展名 -> 文档类型）
    FILE_TYPE_MAPPING: Dict[str, str] = {
        # Excel
        ".xlsx": DocumentType.EXCEL,
        ".xls": DocumentType.EXCEL,
        ".xlsm": DocumentType.EXCEL,
        
        # PDF
        ".pdf": DocumentType.PDF,
        
        # Markdown
        ".md": DocumentType.MARKDOWN,
        ".markdown": DocumentType.MARKDOWN,
        
        # Word
        ".docx": DocumentType.WORD,
        ".doc": DocumentType.WORD,
    }
    
    # 当前启用的文档类型（第一阶段只启用Excel）
    ENABLED_DOCUMENT_TYPES: List[str] = [DocumentType.EXCEL]
    
    # 文件大小限制
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # ==================== Excel处理配置 ====================
    EXCEL_SKIP_HEADER_ROWS: int = 1  # 跳过分组标题行
    EXCEL_MAX_ROWS_PER_BATCH: int = 1000
    EXCEL_SHEET_NAME_ALL: str = "all"  # 处理所有Sheet
    
    # 列名映射配置（用于Scalar Field）
    EXCEL_COLUMN_PREFIX: str = "col_"  # 动态列前缀
    EXCEL_MAX_COLUMNS: int = 100  # 最大列数限制
    
    # ==================== PDF处理配置（预留）====================
    PDF_USE_OCR: bool = True  # 是否对扫描PDF使用OCR
    PDF_OCR_LANGUAGE: str = "chi_sim+eng"  # Tesseract语言包
    PDF_CHUNK_SIZE: int = 500  # 文本分块大小
    PDF_CHUNK_OVERLAP: int = 50
    
    # ==================== Markdown处理配置（预留）====================
    MD_PRESERVE_STRUCTURE: bool = True  # 保留标题结构
    MD_CHUNK_BY_HEADER: bool = True  # 按标题分块
    
    # ==================== Word处理配置（预留）====================
    WORD_EXTRACT_TABLES: bool = True  # 提取表格
    WORD_EXTRACT_IMAGES: bool = False  # 暂不提取图片
    
    # ==================== 检索配置 ====================
    DEFAULT_TOP_K: int = 15
    DEFAULT_RERANK_TOP_K: int = 6
    SIMILARITY_CUTOFF: float = 0.1
    
    # ==================== 引擎池配置 ====================
    ENGINE_POOL_SIZE: int = int(os.getenv("ENGINE_POOL_SIZE", "32"))
    MAX_INFLIGHT: int = int(os.getenv("MAX_INFLIGHT", "200"))

    # 权限模式（第一阶段设为basic，后续改为strict）
    PERMISSION_MODE: str = PermissionMode.BASIC
    
    # 是否启用跨部门查询（管理员功能）
    ENABLE_CROSS_DEPARTMENT: bool = False
    
    # 管理员角色标识（从请求头获取）
    ADMIN_ROLE: str = "admin"
    
    # 请求头字段名（Java网关设置）
    HEADER_USER_ID: str = "X-User-Id"
    HEADER_DEPARTMENT: str = "X-Department"
    HEADER_USER_ROLE: str = "X-User-Role"
    
    # Token验证配置（预留给strict模式）
    JWT_SECRET_KEY: Optional[str] = None
    JWT_ALGORITHM: str = "HS256"
    
    # 用户-部门映射缓存时间（秒）
    USER_DEPT_CACHE_TTL: int = 3600
    


# 全局配置实例
settings = Settings()


# 部门Collection映射
DEPARTMENT_COLLECTIONS = {
    "market": "dept_market",
}

def get_embedding_function():
    """
    获取Embedding函数（复用您的 llm/get_embedding.py）
    """
    from llm.get_embedding import get_embed_model
    return get_embed_model


def get_llm_function():
    """
    获取LLM函数（复用您的 llm/get_inference.py）
    """
    from llm.get_inference import get_dashscope_qwen
    return get_dashscope_qwen


def get_collection_name(department: str) -> str:
    """获取部门Collection名称"""
    return DEPARTMENT_COLLECTIONS.get(department, f"dept_{department}")


def get_document_type(filename: str) -> Optional[str]:
    """根据文件名获取文档类型"""
    import os
    ext = os.path.splitext(filename)[1].lower()
    return settings.FILE_TYPE_MAPPING.get(ext)


def is_document_type_enabled(doc_type: str) -> bool:
    """检查文档类型是否启用"""
    return doc_type in settings.ENABLED_DOCUMENT_TYPES


# 部门Collection映射（后续可改为数据库动态加载）
DEPARTMENT_COLLECTIONS = {
    "market": "dept_market",
    # 动态添加其他部门...
}


def get_collection_name(department: str) -> str:
    """获取部门Collection名称"""
    return DEPARTMENT_COLLECTIONS.get(department, f"dept_{department}")


def add_department_collection(department: str, collection_name: str = None):
    """动态添加部门Collection映射"""
    if not collection_name:
        collection_name = f"dept_{department}"
    DEPARTMENT_COLLECTIONS[department] = collection_name


def get_user_departments(user_id: str) -> list:
    """
    获取用户可访问的部门列表
    
    TODO: 后续对接Java服务或数据库
    现阶段返回默认值
    """
    # 开发阶段：直接返回所有部门
    if settings.PERMISSION_MODE == PermissionMode.DISABLED:
        return list(DEPARTMENT_COLLECTIONS.keys())
    
    # 基础模式：信任请求头传递的department
    # 严格模式：需要查询数据库或调用Java API
    return []


def is_admin_user(user_role: str) -> bool:
    """判断是否为管理员"""
    return user_role == settings.ADMIN_ROLE