#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Milvus中小鼠心脏数据的实际情况
"""
import sys
import os
from pymilvus import connections, Collection, utility
from config import settings

def test_mouse_heart_data():
    """测试Milvus中小鼠心脏数据的实际情况"""
    print("📊 测试Milvus中小鼠心脏数据的实际情况")
    print("=" * 60)
    
    try:
        # 连接Milvus
        print(f"🔌 连接Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            timeout=10
        )
        
        # 获取集合
        collection_name = f"dept_market"
        coll = Collection(collection_name)
        
        # 加载集合
        coll.load()
        
        # 获取实体数量
        entity_count = coll.num_entities
        print(f"✅ 实体数量: {entity_count}")
        
        # 测试1: 查询所有小鼠数据
        print(f"\n测试1: 查询所有小鼠数据")
        print("-" * 40)
        
        results = coll.query(
            expr="species == '小鼠'",
            output_fields=["id", "text", "species", "tissue"],
            limit=20
        )
        
        print(f"物种=小鼠: 检索到 {len(results)} 条数据")
        
        # 打印部分结果，查看tissue字段值
        print(f"\n前10条小鼠数据的组织类型:")
        for i, result in enumerate(results[:10]):
            tissue = result.get("tissue", "未知")
            print(f"   [{i+1}] {tissue}")
        
        # 测试2: 查询所有心脏数据
        print(f"\n测试2: 查询所有心脏数据")
        print("-" * 40)
        
        results = coll.query(
            expr="tissue == '心脏'",
            output_fields=["id", "text", "species", "tissue"],
            limit=20
        )
        
        print(f"组织=心脏: 检索到 {len(results)} 条数据")
        
        # 打印部分结果
        print(f"\n前10条心脏数据的物种:")
        for i, result in enumerate(results[:10]):
            species = result.get("species", "未知")
            print(f"   [{i+1}] {species}")
        
        # 测试3: 查询所有小鼠+心脏数据
        print(f"\n测试3: 查询所有小鼠+心脏数据")
        print("-" * 40)
        
        results = coll.query(
            expr="species == '小鼠' and tissue == '心脏'",
            output_fields=["id", "text", "species", "tissue"],
            limit=20
        )
        
        print(f"物种=小鼠 and 组织=心脏: 检索到 {len(results)} 条数据")
        
        # 测试4: 模糊匹配心脏相关数据
        print(f"\n测试4: 模糊匹配心脏相关数据")
        print("-" * 40)
        
        # 执行模糊查询
        all_results = coll.query(
            expr="",
            output_fields=["id", "text", "species", "tissue"],
            limit=100
        )
        
        # 手动过滤包含心脏的数据
        heart_related = []
        for result in all_results:
            tissue = result.get("tissue", "")
            if "心脏" in tissue or "心肌" in tissue:
                heart_related.append(result)
        
        print(f"模糊匹配心脏相关数据: 检索到 {len(heart_related)} 条数据")
        
        # 打印结果
        print(f"\n心脏相关数据的物种和组织类型:")
        for i, result in enumerate(heart_related[:10]):
            species = result.get("species", "未知")
            tissue = result.get("tissue", "未知")
            print(f"   [{i+1}] 物种: {species}, 组织: {tissue}")
        
        # 测试5: 检查所有组织类型
        print(f"\n测试5: 检查所有组织类型")
        print("-" * 40)
        
        # 获取所有组织类型
        tissue_types = set()
        all_results = coll.query(
            expr="species == '小鼠'",
            output_fields=["tissue"],
            limit=1000
        )
        
        for result in all_results:
            tissue = result.get("tissue", "未知")
            if tissue != "/":
                tissue_types.add(tissue)
        
        print(f"小鼠数据的组织类型共有 {len(tissue_types)} 种")
        print(f"\n所有组织类型列表:")
        for tissue in sorted(tissue_types):
            print(f"   - {tissue}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 断开连接
        try:
            connections.disconnect("default")
            print(f"\n🔌 已断开Milvus连接")
        except:
            pass

if __name__ == "__main__":
    test_mouse_heart_data()
