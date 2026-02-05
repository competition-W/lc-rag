#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取意图识别模型
使用DashScope的tongyi-intent-detect-v3模型
"""

import os
import json
from dashscope import Generation
from utils.logger import logger

# 尝试加载.env文件
try:
    from dotenv import load_dotenv
    # 尝试从多个位置加载.env文件
    load_dotenv()  # 当前目录
    load_dotenv(dotenv_path=".env")  # 当前目录
    load_dotenv(dotenv_path="../.env")  # 上级目录
    load_dotenv(dotenv_path="../../.env")  # 上上级目录
    logger.info("✅ 尝试加载.env文件")
except ImportError:
    logger.warning("⚠️ 未安装python-dotenv，无法自动加载.env文件")


def get_intent_model(api_key: str = None):
    """
    获取意图识别模型
    
    Args:
        api_key: DashScope API密钥，如果为None则从环境变量获取
    
    Returns:
        一个可以调用的模型函数
    """
    if not api_key:
        api_key = os.getenv("DASHSCOPE_API_KEY")
    
    # 检查API密钥
    if not api_key:
        # 尝试直接从config模块获取
        try:
            from config import settings
            api_key = settings.DASHSCOPE_API_KEY
            logger.info("✅ 从config模块获取API密钥")
        except Exception as e:
            logger.warning(f"⚠️ 无法从config模块获取API密钥: {e}")
    
    if not api_key:
        logger.warning("⚠️ 未找到DASHSCOPE_API_KEY，意图识别模型将不可用")
        return None
    
    def intent_model(messages: list):
        """
        调用意图识别模型
        
        Args:
            messages: 消息列表，格式为[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        
        Returns:
            模型的响应
        """
        try:
            response = Generation.call(
                api_key=api_key,
                model="tongyi-intent-detect-v3",
                messages=messages,
                result_format="message"
            )
            
            if response and hasattr(response, 'output') and hasattr(response.output, 'choices'):
                return response.output.choices[0].message.content
            else:
                logger.error("❌ 意图识别模型响应格式错误")
                return None
        except Exception as e:
            logger.error(f"❌ 调用意图识别模型失败: {e}")
            return None
    
    logger.info("✅ 成功初始化意图识别模型: tongyi-intent-detect-v3")
    return intent_model


def create_intent_messages(system_prompt: str, user_query: str):
    """
    创建意图识别模型的消息列表
    
    Args:
        system_prompt: 系统提示词
        user_query: 用户查询
    
    Returns:
        消息列表
    """
    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_query}
    ]


if __name__ == "__main__":
    # 测试脚本
    model = get_intent_model()
    if model:
        # 测试消息
        test_system_prompt = "你是一个专业的生物数据意图识别助手，负责分析用户查询并提取准确的意图和实体信息。"
        test_user_query = "小鼠心脏的单细胞转录组实验数据"
        
        messages = create_intent_messages(test_system_prompt, test_user_query)
        response = model(messages)
        
        print("测试响应:")
        print(response)
    else:
        print("无法初始化模型，请检查API密钥")
