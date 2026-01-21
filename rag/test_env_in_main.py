#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试main.py中的环境变量读取
"""
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检查环境变量
logger.info(f"DASHSCOPE_API_KEY 环境变量存在: {'DASHSCOPE_API_KEY' in os.environ}")
logger.info(f"DASHSCOPE_API_KEY 值: {os.getenv('DASHSCOPE_API_KEY')}")

# 检查当前工作目录
logger.info(f"当前工作目录: {os.getcwd()}")

# 检查Python路径
logger.info(f"Python路径: {os.environ.get('PYTHONPATH')}")

# 检查所有环境变量中包含DASH的变量
logger.info("\n所有包含DASH的环境变量:")
for key, value in os.environ.items():
    if 'DASH' in key:
        logger.info(f"  {key}: {value}")
