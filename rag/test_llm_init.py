#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试LLM初始化是否成功
"""
import os
import logging
from llama_index.core import Settings
from llm.inference import DashScopeLLM

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检查环境变量
logger.info(f"DASHSCOPE_API_KEY 环境变量存在: {'DASHSCOPE_API_KEY' in os.environ}")
logger.info(f"DASHSCOPE_API_KEY 值: {os.getenv('DASHSCOPE_API_KEY')}")

# 尝试初始化LLM
try:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    logger.info(f"尝试使用API_KEY: {api_key} 初始化LLM")
    llm = DashScopeLLM(
        model_name="qwen-plus", 
        api_key=api_key
    )
    logger.info("✅ LLM 初始化成功")
    logger.info(f"LLM 实例: {llm}")
    logger.info(f"LLM 模型名称: {llm.model_name}")
    logger.info(f"LLM API_KEY: {llm.api_key}")
    
    # 尝试调用LLM
    logger.info("尝试调用LLM...")
    response = llm.complete("你好，世界")
    logger.info(f"✅ LLM 调用成功: {response.text}")
except Exception as e:
    logger.error(f"❌ 错误: {e}")
    import traceback
    logger.error(f"❌ 堆栈信息: {traceback.format_exc()}")
