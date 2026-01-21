#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Milvus中存储的数据，检查字段名和值是否与过滤条件匹配
"""
import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_manager import milvus_manager
from config import settings, get_collection_name
from utils.auth import AuthContext

async def test_milvus_data():
    """测试Milvus中的数据"""
    # 获取部门
    department = "market"
    collection_name = get_collection_name(department)
    
    print(f"测试部门: {department}")
    print(f"Collection名称: {collection_name}")
    print("=" * 50)
    
    # 获取索引
    index = milvus_manager.get_existing_index(department)
    if not index:
        print(f"❌ 未找到部门 {department} 的索引")
        return
    
    print("✅ 成功获取索引")
    
    # 测试1: 查询所有数据，看看实际存储了什么
    print("\n测试1: 查询所有数据")
    print("-" * 30)
    
    try:
        retriever = index.as_retriever(similarity_top_k=10)
        all_nodes = await retriever.aretrieve(" ")
        
        print(f"检索到 {len(all_nodes)} 条数据")
        
        if all_nodes:
            # 打印前3条数据的metadata
            for i, node in enumerate(all_nodes[:3]):
                print(f"\n数据 {i+1}:")
                print(f"文本: {node.text}")
                print("Metadata:")
                for k, v in node.metadata.items():
                    # 只打印关键字段
                    if k in ["species", "tissue", "category", "department"]:
                        print(f"  {k}: {v}")
    except Exception as e:
        print(f"❌ 查询失败: {e}")
    
    # 测试2: 只使用department过滤，看看是否能查到数据
    print("\n测试2: 只使用department过滤")
    print("-" * 30)
    
    try:
        from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
        
        filters = MetadataFilters(filters=[
            MetadataFilter(key="department", operator=FilterOperator.EQ, value=department)
        ])
        
        retriever = index.as_retriever(similarity_top_k=10, filters=filters)
        dept_nodes = await retriever.aretrieve(" ")
        
        print(f"按部门过滤检索到 {len(dept_nodes)} 条数据")
    except Exception as e:
        print(f"❌ 按部门过滤查询失败: {e}")
    
    # 测试3: 检查字段名是否存在
    print("\n测试3: 检查字段名是否存在")
    print("-" * 30)
    
    if all_nodes:
        # 打印第一条数据的所有字段名
        first_node = all_nodes[0]
        print("第一条数据的所有字段名:")
        for k in sorted(first_node.metadata.keys()):
            print(f"  {k}")
    
    print("\n" + "=" * 50)
    print("测试完成")

if __name__ == "__main__":
    asyncio.run(test_milvus_data())
