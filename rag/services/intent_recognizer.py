#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图识别器
用于识别用户查询的意图类型
"""

from typing import Dict, Any, Optional
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
import logging

logger = logging.getLogger(__name__)


class IntentRecognizer:
    """
    意图识别器
    负责识别用户查询的意图类型
    """
    
    def __init__(self):
        self.intent_map = {
            "sample_query": ["样本", "组织", "物种", "实验", "平台", "冻存", "裂红", "去死", "核酸质量"],
            "project_query": ["项目", "经验", "方案", "流程", "步骤", "案例", "历史", "记录"]
        }
    
    async def recognize_intent(self, user_query: str) -> str:
        """
        识别用户查询的意图
        
        Args:
            user_query: 用户的自然语言查询
            
        Returns:
            意图类型字符串
        """
        # 1. 基于关键词的简单匹配（快速通道）
        intent = self._keyword_based_recognition(user_query)
        if intent:
            logger.info(f"🔍 基于关键词识别意图: {intent}")
            return intent
        
        # 2. 基于LLM的意图识别（精确通道）
        intent = await self._llm_based_recognition(user_query)
        logger.info(f"🧠 基于LLM识别意图: {intent}")
        return intent
    
    def _keyword_based_recognition(self, user_query: str) -> Optional[str]:
        """
        基于关键词的意图识别
        
        Args:
            user_query: 用户的自然语言查询
            
        Returns:
            意图类型字符串，或None
        """
        user_query_lower = user_query.lower()
        
        # 计算每个意图的关键词匹配数量
        intent_scores = {}
        for intent, keywords in self.intent_map.items():
            score = sum(1 for keyword in keywords if keyword in user_query_lower)
            intent_scores[intent] = score
        
        # 找出得分最高的意图
        max_score = max(intent_scores.values())
        if max_score > 0:
            return max(intent_scores, key=intent_scores.get)
        
        # 默认返回样本查询意图
        return "sample_query"
    
    async def _llm_based_recognition(self, user_query: str) -> str:
        """
        基于LLM的意图识别
        
        Args:
            user_query: 用户的自然语言查询
            
        Returns:
            意图类型字符串
        """
        try:
            system_prompt = """你是一个意图识别专家，需要识别用户的查询意图。

请根据用户的查询，从以下意图中选择一个最匹配的：
1. sample_query: 样本准备查询，涉及样本的物种、组织、实验平台等信息
2. project_query: 项目经验查询，涉及项目的历史、方案、流程等信息
3. other: 其他类型的查询

请只输出意图名称，不要包含任何其他内容。例如：
用户：帮我找小鼠心脏的数据
输出：sample_query

用户：告诉我项目A的流程
输出：project_query

用户：你好
输出：other"""
            
            response = await Settings.llm.achat(
                messages=[
                    ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                    ChatMessage(role=MessageRole.USER, content=user_query)
                ]
            )
            
            intent = response.message.content.strip()
            
            # 验证意图是否有效
            valid_intents = list(self.intent_map.keys()) + ["other"]
            if intent not in valid_intents:
                logger.warning(f"⚠️ 无效的意图识别结果: {intent}，使用默认意图")
                return "sample_query"
            
            return intent
        except Exception as e:
            logger.error(f"❌ LLM意图识别失败: {e}，使用默认意图")
            return "sample_query"
    
    def add_intent(self, intent: str, keywords: list) -> None:
        """
        添加新的意图类型
        
        Args:
            intent: 意图名称
            keywords: 关键词列表
        """
        self.intent_map[intent] = keywords
        logger.info(f"📝 添加了新的意图类型: {intent}")
    
    def get_all_intents(self) -> list:
        """
        获取所有支持的意图类型
        
        Returns:
            意图类型列表
        """
        return list(self.intent_map.keys())


# 全局意图识别器实例
intent_recognizer = IntentRecognizer()
