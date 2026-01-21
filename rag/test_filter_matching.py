#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试过滤条件匹配，检查为什么符合条件的数据查询不到
"""
import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_manager import milvus_manager
from utils.auth import AuthContext

async def test_filter_matching():
    """测试过滤条件匹配"""
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
    
    # 测试1: 查询所有数据，打印更完整的信息
    print("\n测试1: 查询所有数据，打印更完整的信息")
    print("-" * 30)
    
    try:
        retriever = index.as_retriever(similarity_top_k=100)
        all_nodes = await retriever.aretrieve(" ")
        
        print(f"检索到 {len(all_nodes)} 条数据")
        
        # 统计各类别的数量
        species_count = {}
        category_count = {}
        tissue_count = {}
        
        for node in all_nodes:
            # 统计species
            species = node.metadata.get("species", "未知")
            species_count[species] = species_count.get(species, 0) + 1
            
            # 统计category
            category = node.metadata.get("category", "未知")
            category_count[category] = category_count.get(category, 0) + 1
            
            # 统计tissue
            tissue = node.metadata.get("tissue", "未知")
            tissue_count[tissue] = tissue_count.get(tissue, 0) + 1
        
        print("\n各类别统计:")
        print(f"物种: {species_count}")
        print(f"样本大类: {category_count}")
        print(f"样本详细类型: {tissue_count}")
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
    
    # 测试2: 只使用species过滤，看看是否能查到数据
    print("\n测试2: 只使用species过滤")
    print("-" * 30)
    
    try:
        from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
        
        # 测试各种species值
        test_species = ["人", "小鼠", "猪", "大鼠"]
        
        for species in test_species:
            filters = MetadataFilters(filters=[
                MetadataFilter(key="department", operator=FilterOperator.EQ, value=department),
                MetadataFilter(key="species", operator=FilterOperator.EQ, value=species)
            ])
            
            retriever = index.as_retriever(similarity_top_k=10, filters=filters)
            nodes = await retriever.aretrieve(" ")
            
            print(f"species={species}: 检索到 {len(nodes)} 条数据")
    except Exception as e:
        print(f"❌ 按species过滤查询失败: {e}")
    
    # 测试3: 只使用category过滤，看看是否能查到数据
    print("\n测试3: 只使用category过滤")
    print("-" * 30)
    
    try:
        from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
        
        # 测试各种category值
        test_categories = ["冻存组织", "液体类样本", "细胞类样本", "新鲜实体组织"]
        
        for category in test_categories:
            filters = MetadataFilters(filters=[
                MetadataFilter(key="department", operator=FilterOperator.EQ, value=department),
                MetadataFilter(key="category", operator=FilterOperator.EQ, value=category)
            ])
            
            retriever = index.as_retriever(similarity_top_k=10, filters=filters)
            nodes = await retriever.aretrieve(" ")
            
            print(f"category={category}: 检索到 {len(nodes)} 条数据")
    except Exception as e:
        print(f"❌ 按category过滤查询失败: {e}")
    
    print("\n" + "=" * 50)
    print("测试完成")

if __name__ == "__main__":
    asyncio.run(test_filter_matching())
