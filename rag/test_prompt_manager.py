#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试提示词管理器
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.prompt_manager import prompt_manager

def test_prompt_manager():
    """
    测试提示词管理器功能
    """
    print("=== 测试提示词管理器 ===")
    
    # 测试1: 获取所有支持的意图
    print("\n1. 测试获取所有支持的意图:")
    intents = prompt_manager.list_intents()
    print(f"支持的意图: {intents}")
    
    # 测试2: 获取意图识别提示词
    print("\n2. 测试获取意图识别提示词:")
    intent_prompt = prompt_manager.get_intent_detection_prompt()
    if intent_prompt:
        print(f"意图识别提示词: {intent_prompt[:100]}...")
    else:
        print("未获取到意图识别提示词")
    
    # 测试3: 获取响应提示词
    print("\n3. 测试获取响应提示词:")
    if intents:
        for intent in intents[:3]:  # 只测试前3个意图
            prompt_data = prompt_manager.get_response_prompt(intent)
            if prompt_data:
                print(f"意图 '{intent}' 的提示词: {prompt_data['name']}")
            else:
                print(f"未获取到意图 '{intent}' 的提示词")
    
    # 测试4: 测试get_prompt方法
    print("\n4. 测试get_prompt方法:")
    if intents:
        test_intent = intents[0]
        prompt_config = prompt_manager.get_prompt(test_intent)
        print(f"意图 '{test_intent}' 的配置: {prompt_config}")
    
    # 测试5: 测试获取不存在的意图
    print("\n5. 测试获取不存在的意图:")
    nonexistent_intent = "nonexistent_intent"
    prompt_config = prompt_manager.get_prompt(nonexistent_intent)
    print(f"不存在的意图 '{nonexistent_intent}' 的配置: {prompt_config}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_prompt_manager()
