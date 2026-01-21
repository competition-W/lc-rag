#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试纯关键词检索策略
"""

import asyncio
import logging
from services.query_parser import parse_user_query
from services.query_service import unified_query_service
from utils.auth import AuthContext

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_keyword_retrieval():
    """
    测试纯关键词检索策略
    """
    # 创建一个简单的 AuthContext 对象
    auth = AuthContext(user_id="1", department="market")
    
    # 测试查询
    test_queries = [
        "查询小鼠的心脏冻存组织数据",
        "小鼠心脏冻存组织样本的保存方案是怎样的？",
        "2023年的项目A经验",
        "帮我找一下猪的所有实验数据",
        "哪些样本的实验结果显示有炎症反应？"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"测试查询: {query}")
        print(f"{'='*60}")
        
        try:
            # 调用 parse_user_query 函数
            search_term, filters, intent = await parse_user_query(query, auth)
            
            print(f"\n1. 意图识别结果:")
            print(f"   - 识别意图: {intent}")
            print(f"   - 提取关键词: {search_term}")
            print(f"   - 提取过滤条件: {filters}")
            
            # 调用 unified_query_service 函数
            print(f"\n2. 检索策略执行:")
            result = await unified_query_service(
                auth=auth,
                query_text=search_term,
                column_filters=filters,
                intent=intent
            )
            
            print(f"   - 检索模式: {result['mode']}")
            print(f"   - 回答: {result['answer']}")
            print(f"   - 检索结果条数: {len(result['sources'])}")
            
            print(f"\n测试成功!")
        except Exception as e:
            print(f"\n测试失败: {e}")
            logger.exception(f"测试失败: {e}")

if __name__ == "__main__":
    asyncio.run(test_keyword_retrieval())
