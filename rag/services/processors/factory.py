# 文件: /mnt/omicshub/rag/services/processors/factory.py

from typing import Optional
import logging
import os

# 确保这些文件存在，或者暂时注释掉未实现的 Processor
from .base import BaseDocumentProcessor
from .excel_processor import ExcelProcessor
# 暂时注释掉可能未实现的文件，防止再次报错
# from .pdf_processor import PDFProcessor
# from .markdown_processor import MarkdownProcessor
# from .word_processor import WordProcessor

from config import DocumentType, is_document_type_enabled, settings, get_document_type

logger = logging.getLogger(__name__)

class DocumentProcessorFactory:
    """
    文档处理器工厂
    (已重命名为 DocumentProcessorFactory 以匹配 tasks 中的引用)
    """
    
    _processors = {
        DocumentType.EXCEL: ExcelProcessor,
        # DocumentType.PDF: PDFProcessor,
        # DocumentType.MARKDOWN: MarkdownProcessor,
        # DocumentType.WORD: WordProcessor,
    }
    
    @classmethod
    def get_processor(cls, filename: str, department: str = "default") -> Optional[BaseDocumentProcessor]:
        """
        根据文件名自动获取处理器 (适配 tasks 调用)
        """
        # 1. 根据扩展名获取文档类型
        doc_type = get_document_type(filename)
        if not doc_type:
            logger.warning(f"无法识别的文件类型: {filename}")
            return None
            
        # 2. 调用内部创建方法
        return cls.create_processor(doc_type, department)

    @classmethod
    def create_processor(
        cls,
        doc_type: str,
        department: str
    ) -> Optional[BaseDocumentProcessor]:
        """
        创建文档处理器实例
        """
        # 检查类型是否启用
        if not is_document_type_enabled(doc_type):
            logger.warning(f"文档类型 {doc_type} 未启用")
            return None
        
        # 获取处理器类
        processor_class = cls._processors.get(doc_type)
        if not processor_class:
            logger.error(f"未找到对应的处理器类: {doc_type}")
            return None
        
        # 实例化
        try:
            return processor_class(department=department)
        except Exception as e:
            logger.exception(f"处理器实例化失败: {doc_type}")
            return None
    
    @classmethod
    def get_supported_types(cls) -> list:
        """获取所有支持的文档类型"""
        return list(cls._processors.keys())