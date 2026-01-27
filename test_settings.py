#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试settings模块是否正确加载.env文件
"""
from rag.config import settings
import os

print("🔍 测试settings模块的环境变量加载...")
print(f"当前工作目录: {os.getcwd()}")
print(f".env文件存在: {os.path.exists('.env')}")

# 检查DASHSCOPE_API_KEY
print(f"\n📋 DASHSCOPE_API_KEY相关信息:")
print(f"settings.DASHSCOPE_API_KEY: {settings.DASHSCOPE_API_KEY}")
print(f"settings.DASHSCOPE_API_KEY类型: {type(settings.DASHSCOPE_API_KEY)}")

# 直接检查环境变量
print(f"\n📋 系统环境变量:")
print(f"os.getenv('DASHSCOPE_API_KEY'): {os.getenv('DASHSCOPE_API_KEY')}")

# 检查其他settings值
print(f"\n📋 其他settings配置:")
print(f"settings.OPENAI_BASE_URL: {settings.OPENAI_BASE_URL}")
print(f"settings.LLM_PROVIDER: {settings.LLM_PROVIDER}")
print(f"settings.EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")
print(f"settings.EMBEDDING_MODEL: {settings.EMBEDDING_MODEL}")

# 检查env_file路径
print(f"\n📋 Settings模型配置:")
print(f"settings.model_config: {settings.model_config}")