#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档处理器抽象基类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProcessedChunk:
    """处理后的文档块"""
    chunk_id: str                    # 唯一ID
    content: str                     # 文本内容（用于向量化）
    metadata: Dict[str, Any]         # 元数据
    
    # 可选字段
    page_number: Optional[int] = None       # 页码（PDF/Word）
    table_data: Optional[Dict] = None       # 表格数据（Excel/Word）
    header_level: Optional[int] = None      # 标题层级（Markdown）
    

@dataclass
class ProcessResult:
    """文档处理结果"""
    success: bool
    chunks: List[ProcessedChunk]
    total_chunks: int
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None  # 文档级元数据


class BaseDocumentProcessor(ABC):
    """文档处理器基类"""
    
    def __init__(self, department: str):
        """
        Args:
            department: 所属部门
        """
        self.department = department
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def process(
        self,
        file_data: bytes,
        filename: str,
        user_id: str,
        **kwargs
    ) -> ProcessResult:
        """
        处理文档（异步）
        
        Args:
            file_data: 文件二进制数据
            filename: 文件名
            user_id: 上传用户ID
            **kwargs: 额外参数
            
        Returns:
            ProcessResult: 处理结果
        """
        pass
    
    @abstractmethod
    def validate_file(self, file_data: bytes, filename: str) -> bool:
        """
        验证文件有效性
        
        Args:
            file_data: 文件数据
            filename: 文件名
            
        Returns:
            bool: 是否有效
        """
        pass
    
    def _generate_chunk_id(self, base_id: str, index: int) -> str:
        """生成chunk ID"""
        return f"{base_id}_chunk_{index}"
    
    def _build_base_metadata(
        self,
        filename: str,
        user_id: str,
        department: str
    ) -> Dict[str, Any]:
        """构建基础元数据"""
        return {
            "department": department,
            "filename": filename,
            "user_id": user_id,
            "processor_type": self.__class__.__name__,
        }