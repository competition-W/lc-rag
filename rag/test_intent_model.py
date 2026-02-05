#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试意图识别模型是否可用
"""

import os
import sys
from llm.get_intent_model import get_intent_model, create_intent_messages
from utils.logger import logger


def test_intent_model_availability():
    """
    测试意图识别模型是否可用
    """
    print("\n=== 测试意图识别模型可用性 ===")
    
    # 检查环境变量
    print("检查环境变量:")
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if dashscope_api_key:
        print("✅ DASHSCOPE_API_KEY: 已设置")
        print(f"   密钥长度: {len(dashscope_api_key)} 字符")
    else:
        print("❌ DASHSCOPE_API_KEY: 未设置")
    
    # 尝试初始化模型
    print("\n尝试初始化意图识别模型...")
    model = get_intent_model()
    
    if model:
        print("✅ 意图识别模型初始化成功!")
        
        # 测试模型调用
        print("\n测试模型调用...")
        test_system_prompt = "你是一个专业的生物数据意图识别助手，负责分析用户查询并提取准确的意图和实体信息。"
        test_user_query = "小鼠心脏的单细胞转录组实验数据"
        
        messages = create_intent_messages(test_system_prompt, test_user_query)
        print(f"创建的消息: {messages}")
        
        try:
            response = model(messages)
            if response:
                print(f"✅ 模型调用成功!")
                print(f"响应内容: {response}")
            else:
                print("❌ 模型调用失败: 未返回响应")
        except Exception as e:
            print(f"❌ 模型调用失败: {e}")
    else:
        print("❌ 意图识别模型初始化失败")


def test_model_import():
    """
    测试模型导入是否成功
    """
    print("\n=== 测试模型导入 ===")
    
    try:
        from llm.get_intent_model import get_intent_model
        print("✅ 成功导入 get_intent_model")
        
        from llm.get_intent_model import create_intent_messages
        print("✅ 成功导入 create_intent_messages")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")


def test_dashscope_import():
    """
    测试dashscope库是否安装
    """
    print("\n=== 测试dashscope库 ===")
    
    try:
        import dashscope
        print(f"✅ 成功导入 dashscope 库")
        print(f"   版本: {getattr(dashscope, '__version__', '未知')}")
    except ImportError as e:
        print(f"❌ dashscope 库未安装: {e}")


if __name__ == "__main__":
    print("开始测试意图识别模型...")
    
    # 测试dashscope库
    test_dashscope_import()
    
    # 测试模型导入
    test_model_import()
    
    # 测试模型可用性
    test_intent_model_availability()
    
    print("\n测试完成!")
