#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown文档处理器（预留接口）
"""
from .base import BaseDocumentProcessor, ProcessResult


class MarkdownProcessor(BaseDocumentProcessor):
    """Markdown处理器"""
    
    def validate_file(self, file_data: bytes, filename: str) -> bool:
        return filename.endswith(('.md', '.markdown'))
    
    async def process(
        self,
        file_data: bytes,
        filename: str,
        user_id: str,
        **kwargs
    ) -> ProcessResult:
        """
        TODO: 实现以下功能
        1. Markdown解析（mistune/markdown-it-py）
        2. 按标题层级分块
        3. 保留代码块和表格
        """
        raise NotImplementedError("Markdown处理器尚未实现")