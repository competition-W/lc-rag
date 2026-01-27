#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试意图命名体系统一后的功能
"""
import asyncio
import logging
from rag.services.query_parser import parse_user_query
from utils.auth import AuthContext

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_intent_unification():
    """测试意图命名体系统一"""
    print("=== 测试意图命名体系统一 ===\n")
    
    # 测试用例
    test_cases = [
        "人新鲜组织的细胞活率情况如何？",
        "小鼠冻存组织的抽核方案",
        "单细胞样本制备的送样要求",
        "肺组织的解离实验方案"
    ]
    
    for query in test_cases:
        print(f"测试查询: {query}")
        try:
            # 调用查询解析
            search_term, filters, intent = await parse_user_query(query, AuthContext(user_id="test", department="test"))
            
            # 打印结果
            print(f"  意图: {intent}")
            print(f"  搜索词: {search_term}")
            print(f"  过滤条件: {filters}")
            print()
        except Exception as e:
            print(f"  错误: {e}")
            print()
    
    print("=== 测试完成 ===")

if __name__ == "__main__":
    asyncio.run(test_intent_unification())
