#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的查询服务，验证集合加载问题是否解决
"""
import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.query_parser import parse_user_query
from services.query_service import unified_query_service
from utils.auth import AuthContext

async def test_query_fix():
    """测试修复后的查询服务"""
    print("📊 测试修复后的查询服务")
    print("=" * 60)
    
    # 测试查询
    test_query = "查询小鼠的实验数据"
    
    print(f"测试查询: {test_query}")
    print("=" * 60)
    
    try:
        # 1. 解析查询
        auth_context = AuthContext(user_id="1", department="market", user_role=None)
        search_term, filters, intent = await parse_user_query(test_query, auth_context)
        
        print(f"意图识别: {intent}")
        print(f"搜索词: {search_term}")
        print(f"过滤条件: {filters}")
        
        # 2. 执行查询服务
        print("\n🔍 执行查询服务...")
        result = await unified_query_service(
            auth=auth_context,
            query_text=search_term,
            column_filters=filters,
            llm_top_k=20,
            semantic_top_k=15,
            intent=intent
        )
        
        print(f"\n✅ 查询成功")
        print(f"模式: {result.get('mode')}")
        print(f"结果数量: {len(result.get('sources', []))}")
        print(f"所有结果数量: {len(result.get('all_rows', []))}")
        
        # 3. 打印部分结果
        if result.get('sources'):
            print(f"\n📋 前3条结果预览:")
            for i, source in enumerate(result['sources'][:3]):
                print(f"\n[{i+1}] 文本: {source.get('text', '')[:100]}...")
                print(f"   物种: {source.get('metadata', {}).get('species')}")
                print(f"   样本大类: {source.get('metadata', {}).get('category')}")
                print(f"   样本详细类型: {source.get('metadata', {}).get('tissue')}")
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成")

if __name__ == "__main__":
    asyncio.run(test_query_fix())
