#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokens消耗统计工具
支持文本模型、嵌入模型和重排序模型的tokens统计
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class TokenCounter:
    """
    Tokens消耗统计类
    支持统计文本模型、嵌入模型和重排序模型的tokens消耗
    """
    
    def __init__(self):
        self.token_stats = {
            "llm": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_calls": 0,
                "calls": []
            },
            "embedding": {
                "total_tokens": 0,
                "total_calls": 0,
                "calls": []
            },
            "rerank": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_calls": 0,
                "calls": []
            }
        }
    
    def count_llm_tokens(self, input_text: str, output_text: str, model_name: str = "") -> Dict[str, Any]:
        """
        统计文本模型的tokens消耗
        
        Args:
            input_text: 输入文本
            output_text: 输出文本
            model_name: 模型名称
            
        Returns:
            本次调用的tokens统计结果
        """
        # 这里使用简化的tokens计算方法，实际应用中应使用对应模型的tokenizer
        # 例如：from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # input_tokens = len(tokenizer.encode(input_text))
        # output_tokens = len(tokenizer.encode(output_text))
        
        # 简化的tokens计算：按字符数的1/4估算
        input_tokens = max(1, len(input_text) // 4)
        output_tokens = max(1, len(output_text) // 4)
        
        call_stats = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
        
        # 更新统计数据
        self.token_stats["llm"]["total_input_tokens"] += input_tokens
        self.token_stats["llm"]["total_output_tokens"] += output_tokens
        self.token_stats["llm"]["total_calls"] += 1
        self.token_stats["llm"]["calls"].append(call_stats)
        
        logger.info(f"💬 LLM调用 - 输入: {input_tokens} tokens, 输出: {output_tokens} tokens, 总计: {input_tokens + output_tokens} tokens")
        return call_stats
    
    def count_embedding_tokens(self, text: str, model_name: str = "") -> Dict[str, Any]:
        """
        统计嵌入模型的tokens消耗
        
        Args:
            text: 输入文本
            model_name: 模型名称
            
        Returns:
            本次调用的tokens统计结果
        """
        # 简化的tokens计算：按字符数的1/4估算
        tokens = max(1, len(text) // 4)
        
        call_stats = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "tokens": tokens
        }
        
        # 更新统计数据
        self.token_stats["embedding"]["total_tokens"] += tokens
        self.token_stats["embedding"]["total_calls"] += 1
        self.token_stats["embedding"]["calls"].append(call_stats)
        
        logger.info(f"🔤 Embedding调用 - 总计: {tokens} tokens")
        return call_stats
    
    def count_rerank_tokens(self, query: str, documents: list, model_name: str = "") -> Dict[str, Any]:
        """
        统计重排序模型的tokens消耗
        
        Args:
            query: 查询文本
            documents: 文档列表
            model_name: 模型名称
            
        Returns:
            本次调用的tokens统计结果
        """
        # 简化的tokens计算：按字符数的1/4估算
        input_tokens = max(1, len(query) // 4) + sum(max(1, len(doc) // 4) for doc in documents)
        output_tokens = max(1, len(documents))  # 输出tokens通常较少，按文档数估算
        
        call_stats = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
        
        # 更新统计数据
        self.token_stats["rerank"]["total_input_tokens"] += input_tokens
        self.token_stats["rerank"]["total_output_tokens"] += output_tokens
        self.token_stats["rerank"]["total_calls"] += 1
        self.token_stats["rerank"]["calls"].append(call_stats)
        
        logger.info(f"🔍 Rerank调用 - 输入: {input_tokens} tokens, 输出: {output_tokens} tokens, 总计: {input_tokens + output_tokens} tokens")
        return call_stats
    
    def get_total_stats(self) -> Dict[str, Any]:
        """
        获取所有模型的tokens消耗统计
        
        Returns:
            所有模型的tokens消耗统计
        """
        return self.token_stats.copy()
    
    def get_llm_stats(self) -> Dict[str, Any]:
        """
        获取文本模型的tokens消耗统计
        
        Returns:
            文本模型的tokens消耗统计
        """
        return self.token_stats["llm"].copy()
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        获取嵌入模型的tokens消耗统计
        
        Returns:
            嵌入模型的tokens消耗统计
        """
        return self.token_stats["embedding"].copy()
    
    def get_rerank_stats(self) -> Dict[str, Any]:
        """
        获取重排序模型的tokens消耗统计
        
        Returns:
            重排序模型的tokens消耗统计
        """
        return self.token_stats["rerank"].copy()
    
    def reset_stats(self) -> None:
        """
        重置所有统计数据
        """
        self.__init__()

# 创建全局实例
token_counter = TokenCounter()
