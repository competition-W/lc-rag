#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Word文档处理器（预留接口）
"""
from .base import BaseDocumentProcessor, ProcessResult


class WordProcessor(BaseDocumentProcessor):
    """Word处理器"""
    
    def validate_file(self, file_data: bytes, filename: str) -> bool:
        return filename.endswith(('.docx', '.doc'))
    
    async def process(
        self,
        file_data: bytes,
        filename: str,
        user_id: str,
        **kwargs
    ) -> ProcessResult:
        """
        TODO: 实现以下功能
        1. 文本提取（python-docx）
        2. 表格提取
        3. 样式保留（标题/列表）
        """
        raise NotImplementedError("Word处理器尚未实现")