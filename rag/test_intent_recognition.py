#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试意图识别和提示词管理功能
"""

import asyncio
import logging
from services.query_parser import parse_user_query
from utils.auth import AuthContext

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_intent_recognition():
    """
    测试意图识别和提示词管理功能
    """
    # 创建一个简单的 AuthContext 对象
    auth = AuthContext(user_id="1", department="market")
    
    # 测试查询
    test_queries = [
        "小鼠心脏冻存组织样本的保存方案是怎样的？",
        "2023年的项目A经验",
        "帮我找一下猪的所有实验数据",
        "哪些样本的实验结果显示有炎症反应？",
        "查看小鼠大脑的相关记录"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"测试查询: {query}")
        print(f"{'='*50}")
        
        try:
            # 调用 parse_user_query 函数
            search_term, filters, intent = await parse_user_query(query, auth)
            
            print(f"识别意图: {intent}")
            print(f"提取关键词: {search_term}")
            print(f"提取过滤条件: {filters}")
            print(f"测试成功!")
        except Exception as e:
            print(f"测试失败: {e}")
            logging.exception(f"测试失败: {e}")

if __name__ == "__main__":
    asyncio.run(test_intent_recognition())
