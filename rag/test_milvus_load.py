#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Milvus集合加载问题
"""
import sys
import os
from pymilvus import connections, Collection, utility
from config import settings

def test_milvus_load():
    """测试Milvus集合加载"""
    print("📊 测试Milvus集合加载")
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
        
        # 获取集合名称
        collection_name = f"dept_market"
        print(f"✅ 集合名称: {collection_name}")
        
        # 获取Collection对象
        coll = Collection(collection_name)
        
        # 加载集合（无论是否已加载，显式加载一次）
        print(f"📥 正在加载集合...")
        coll.load()
        print(f"✅ 集合加载成功")
        
        # 检查集合加载状态
        load_state = utility.load_state(collection_name)
        print(f"🔍 集合加载状态: {load_state}")
        
        # 获取实体数量
        entity_count = coll.num_entities
        print(f"✅ 实体数量: {entity_count}")
        
        # 测试搜索（如果集合已加载，应该不会报错）
        print(f"\n🔍 测试简单查询...")
        
        # 执行简单查询
        results = coll.query(
            expr="",
            output_fields=["id"],
            limit=10
        )
        
        print(f"✅ 查询成功，返回 {len(results)} 条数据")
        
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
    test_milvus_load()
