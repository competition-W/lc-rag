#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试LLM生成功能
"""
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 从.env文件加载API密钥
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")
logger.info(f"🔍 使用API_KEY: {api_key} 测试LLM")

# 直接初始化DashScopeLLM
from llm.inference import DashScopeLLM

# 初始化LLM
llm = DashScopeLLM(
    model_name="qwen-plus", 
    api_key=api_key
)

logger.info("✅ LLM 初始化成功")

# 测试简单的生成
logger.info("📞 测试简单生成...")
response = llm.complete("你好，世界")
logger.info(f"✅ 生成成功: {response.text}")

# 测试复杂的生成
logger.info("📞 测试复杂生成...")
system_prompt = "你是一位资深的生物数据分析专家。"
user_prompt = "根据以下小鼠心脏冻存组织的实验数据，总结实验指标：\n\n**实验指标**\n组织重量数值：262.00\n是否裂红：否\n是否去死：否\n样本保存方案：4℃"

full_prompt = f"{system_prompt}\n\n用户问题：{user_prompt}"
response = llm.complete(full_prompt)
logger.info(f"✅ 生成成功: {response.text}")
logger.info(f"📏 回答长度: {len(response.text)} 字符")
