#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Milvus中小鼠数据的实际数量和分布
"""
import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.milvus_manager import milvus_manager
from utils.auth import AuthContext

async def test_mouse_data_count():
    """测试Milvus中小鼠数据的实际数量"""
    # 获取部门
    department = "market"
    
    print(f"测试部门: {department}")
    print("=" * 60)
    
    # 获取索引
    index = milvus_manager.get_existing_index(department)
    if not index:
        print(f"❌ 未找到部门 {department} 的索引")
        return
    
    print("✅ 成功获取索引")
    
    try:
        from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
        
        # 测试1: 查询所有小鼠数据
        print("\n测试1: 查询所有小鼠数据")
        print("-" * 30)
        
        filters = MetadataFilters(filters=[
            MetadataFilter(key="department", operator=FilterOperator.EQ, value=department),
            MetadataFilter(key="species", operator=FilterOperator.EQ, value="小鼠")
        ])
        
        retriever = index.as_retriever(similarity_top_k=2000, filters=filters)
        mouse_nodes = await retriever.aretrieve(" ")
        
        print(f"species=小鼠: 检索到 {len(mouse_nodes)} 条数据")
        
        # 测试2: 查询所有包含"小鼠"的数据（包括多值情况）
        print("\n测试2: 查询所有包含'小鼠'的数据")
        print("-" * 30)
        
        # 由于MetadataFilter的EQ是精确匹配，我们需要查询所有数据然后过滤
        retriever_all = index.as_retriever(similarity_top_k=2000)
        all_nodes = await retriever_all.aretrieve(" ")
        
        # 手动过滤包含"小鼠"的数据
        mouse_related_nodes = []
        for node in all_nodes:
            species = node.metadata.get("species", "")
            if "小鼠" in species:
                mouse_related_nodes.append(node)
        
        print(f"所有包含'小鼠'的数据: 共 {len(mouse_related_nodes)} 条")
        
        # 统计species字段的分布
        species_dist = {}
        for node in all_nodes:
            species = node.metadata.get("species", "未知")
            species_dist[species] = species_dist.get(species, 0) + 1
        
        print(f"\nspecies字段分布:")
        for species, count in sorted(species_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {species}: {count} 条")
        
        # 测试3: 检查数据导入情况
        print("\n测试3: 检查数据导入情况")
        print("-" * 30)
        
        # 统计uploader分布
        uploader_dist = {}
        for node in all_nodes:
            uploader = node.metadata.get("uploader", "未知")
            uploader_dist[uploader] = uploader_dist.get(uploader, 0) + 1
        
        print(f"数据上传者分布:")
        for uploader, count in uploader_dist.items():
            print(f"  {uploader}: {count} 条")
        
        # 统计filename分布
        filename_dist = {}
        for node in all_nodes:
            filename = node.metadata.get("filename", "未知")
            filename_dist[filename] = filename_dist.get(filename, 0) + 1
        
        print(f"\n文件名分布:")
        for filename, count in sorted(filename_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {filename}: {count} 条")
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成")

if __name__ == "__main__":
    asyncio.run(test_mouse_data_count())
