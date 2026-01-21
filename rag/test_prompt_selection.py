#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试提示词选择逻辑
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.prompt_manager import prompt_manager

# 测试1：检查annotation_query意图是否存在
print("测试1：检查annotation_query意图是否存在")
intents = prompt_manager.list_intents()
print(f"支持的意图列表: {intents}")
print(f"annotation_query意图是否存在: {'annotation_query' in intents}")

# 测试2：获取annotation_query提示词
print("\n测试2：获取annotation_query提示词")
annotation_prompt = prompt_manager.get_prompt("annotation_query")
print(f"意图名称: {annotation_prompt['name']}")
print(f"系统提示词: {annotation_prompt['system_prompt'][:200]}...")

# 测试3：获取不存在的意图，应该返回默认提示词
print("\n测试3：获取不存在的意图")
unknown_prompt = prompt_manager.get_prompt("unknown_intent")
print(f"意图名称: {unknown_prompt['name']}")

print("\n✅ 提示词管理器测试完成！")
