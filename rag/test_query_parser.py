#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试查询解析器修复
"""
import asyncio
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.query_parser import parse_user_query
from utils.auth import AuthContext

async def test_query_parser():
    """测试查询解析器"""
    # 测试查询
    test_query = "小鼠心脏冻存组织的细胞活率数据"
    
    print(f"测试查询: {test_query}")
    print("=" * 50)
    
    # 调用查询解析器
    auth_context = AuthContext(user_id="1", department="market", user_role=None)
    search_term, filters, intent = await parse_user_query(test_query, auth_context)
    
    print(f"意图识别: {intent}")
    print(f"搜索词: {search_term}")
    print(f"过滤条件: {filters}")
    
    # 验证过滤条件是否正确
    expected_fields = ["species", "tissue", "category", "col_xi_bao_huo_lv"]
    for field in expected_fields:
        if field in filters:
            print(f"✅ 字段 {field} 正确提取: {filters[field]}")
        else:
            print(f"❌ 字段 {field} 未提取")
    
    print("=" * 50)
    print("测试完成")

if __name__ == "__main__":
    asyncio.run(test_query_parser())
