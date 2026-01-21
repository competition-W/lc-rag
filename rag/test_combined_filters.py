#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试组合过滤条件，检查为什么符合条件的数据查询不到
"""
import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_manager import milvus_manager
from utils.auth import AuthContext

async def test_combined_filters():
    """测试组合过滤条件"""
    # 获取部门
    department = "market"
    
    print(f"测试部门: {department}")
    print("=" * 50)
    
    # 获取索引
    index = milvus_manager.get_existing_index(department)
    if not index:
        print(f"❌ 未找到部门 {department} 的索引")
        return
    
    print("✅ 成功获取索引")
    
    try:
        from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
        
        # 测试1: 单独使用species=小鼠过滤
        print("\n测试1: 单独使用species=小鼠过滤")
        print("-" * 30)
        
        filters1 = MetadataFilters(filters=[
            MetadataFilter(key="department", operator=FilterOperator.EQ, value=department),
            MetadataFilter(key="species", operator=FilterOperator.EQ, value="小鼠")
        ])
        
        retriever = index.as_retriever(similarity_top_k=2000, filters=filters1)
        nodes1 = await retriever.aretrieve(" ")
        
        print(f"species=小鼠: 检索到 {len(nodes1)} 条数据")
        
        if nodes1:
            # 打印前3条数据的信息
            for i, node in enumerate(nodes1[:3]):
                print(f"\n数据 {i+1}:")
                print(f"  物种: {node.metadata.get('species')}")
                print(f"  样本大类: {node.metadata.get('category')}")
                print(f"  样本详细类型: {node.metadata.get('tissue')}")
        
        # 测试2: 单独使用category=冻存组织过滤
        print("\n测试2: 单独使用category=冻存组织过滤")
        print("-" * 30)
        
        filters2 = MetadataFilters(filters=[
            MetadataFilter(key="department", operator=FilterOperator.EQ, value=department),
            MetadataFilter(key="category", operator=FilterOperator.EQ, value="冻存组织")
        ])
        
        retriever = index.as_retriever(similarity_top_k=2000, filters=filters2)
        nodes2 = await retriever.aretrieve(" ")
        
        print(f"category=冻存组织: 检索到 {len(nodes2)} 条数据")
        
        if nodes2:
            # 打印前3条数据的信息
            for i, node in enumerate(nodes2[:3]):
                print(f"\n数据 {i+1}:")
                print(f"  物种: {node.metadata.get('species')}")
                print(f"  样本大类: {node.metadata.get('category')}")
                print(f"  样本详细类型: {node.metadata.get('tissue')}")
        
        # 测试3: 组合使用species=小鼠和category=冻存组织过滤
        print("\n测试3: 组合使用species=小鼠和category=冻存组织过滤")
        print("-" * 30)
        
        filters3 = MetadataFilters(filters=[
            MetadataFilter(key="department", operator=FilterOperator.EQ, value=department),
            MetadataFilter(key="species", operator=FilterOperator.EQ, value="小鼠"),
            MetadataFilter(key="category", operator=FilterOperator.EQ, value="冻存组织")
        ])
        
        retriever = index.as_retriever(similarity_top_k=2000, filters=filters3)
        nodes3 = await retriever.aretrieve(" ")
        
        print(f"species=小鼠 AND category=冻存组织: 检索到 {len(nodes3)} 条数据")
        
        # 测试4: 组合使用所有过滤条件
        print("\n测试4: 组合使用所有过滤条件")
        print("-" * 30)
        
        filters4 = MetadataFilters(filters=[
            MetadataFilter(key="department", operator=FilterOperator.EQ, value=department),
            MetadataFilter(key="species", operator=FilterOperator.EQ, value="小鼠"),
            MetadataFilter(key="category", operator=FilterOperator.EQ, value="冻存组织"),
            MetadataFilter(key="tissue", operator=FilterOperator.EQ, value="心脏")
        ])
        
        retriever = index.as_retriever(similarity_top_k=2000, filters=filters4)
        nodes4 = await retriever.aretrieve(" ")
        
        print(f"species=小鼠 AND category=冻存组织 AND tissue=心脏: 检索到 {len(nodes4)} 条数据")
        
        # 测试5: 查看所有小鼠数据的组织类型
        print("\n测试5: 查看所有小鼠数据的组织类型")
        print("-" * 30)
        
        if nodes1:
            # 统计小鼠数据的组织类型
            tissue_count = {}
            for node in nodes1:
                tissue = node.metadata.get("tissue", "未知")
                tissue_count[tissue] = tissue_count.get(tissue, 0) + 1
            
            print(f"小鼠数据的组织类型统计: {tissue_count}")
            
            # 打印所有小鼠数据的组织类型
            print("所有小鼠数据的组织类型:")
            for node in nodes1:
                print(f"  - {node.metadata.get('tissue')}")
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
    
    print("\n" + "=" * 50)
    print("测试完成")

if __name__ == "__main__":
    asyncio.run(test_combined_filters())
