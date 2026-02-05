#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试意图检测器
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.intent_detector import IntentDetector

def test_intent_detector():
    """
    测试意图检测器功能
    """
    print("=== 测试意图检测器 ===")
    
    # 初始化意图检测器
    try:
        detector = IntentDetector()
        print("✅ 意图检测器初始化成功")
    except Exception as e:
        print(f"❌ 意图检测器初始化失败: {e}")
        return
    
    # 测试样例查询
    test_queries = [
        "我想查询样本A的实验数据",
        "项目X的细胞注释结果是什么？",
        "有哪些关于基因表达的实验？",
        "样本B的质量控制指标如何？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. 测试查询: {query}")
        try:
            result = detector.detect_intent(query)
            print(f"意图检测结果: {result}")
        except Exception as e:
            print(f"意图检测失败: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_intent_detector()
