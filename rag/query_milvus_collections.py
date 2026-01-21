#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询Milvus中所有集合的详细信息，并将结果保存到文件
"""
import sys
import os
import json
from datetime import datetime
from pymilvus import connections, utility, Collection
from config import settings

def query_milvus_collections():
    """查询Milvus中所有集合的详细信息"""
    print("📊 查询Milvus中所有集合的详细信息")
    print("=" * 60)
    
    # 结果存储字典
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "milvus_server": f"{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
        "collections": []
    }
    
    try:
        # 连接Milvus
        print(f"🔌 连接Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            timeout=10
        )
        
        # 获取所有集合名称
        collections = utility.list_collections()
        print(f"✅ 成功连接，共发现 {len(collections)} 个集合")
        print("=" * 60)
        
        # 遍历每个集合，获取详细信息
        for collection_name in collections:
            print(f"\n📋 集合名称: {collection_name}")
            print("-" * 40)
            
            collection_info = {
                "name": collection_name,
                "exists": False,
                "stats": {},
                "schema": {},
                "indexes": [],
                "sample_data": []
            }
            
            # 1. 检查集合是否存在
            if utility.has_collection(collection_name):
                print(f"✅ 集合存在")
                collection_info["exists"] = True
            else:
                print(f"❌ 集合不存在")
                results["collections"].append(collection_info)
                continue
            
            # 2. 获取集合统计信息
            try:
                coll = Collection(collection_name)
                coll.load()
                row_count = coll.num_entities
                print(f"📊 集合统计:")
                print(f"   实体数量: {row_count}")
                
                collection_info["stats"]["row_count"] = row_count
            except Exception as e:
                print(f"❌ 获取集合统计失败: {e}")
            
            # 3. 获取集合字段信息
            try:
                schema = coll.schema
                print(f"� 集合字段信息:")
                
                collection_info["schema"]["description"] = schema.description
                collection_info["schema"]["fields"] = []
                
                for field in schema.fields:
                    field_info = {
                        "name": field.name,
                        "type": str(field.dtype),
                        "is_primary": field.is_primary,
                        "description": field.description
                    }
                    collection_info["schema"]["fields"].append(field_info)
                    print(f"   - {field.name} ({field.dtype}, primary: {field.is_primary})")
            except Exception as e:
                print(f"❌ 获取集合字段信息失败: {e}")
            
            # 4. 获取集合索引信息
            try:
                indexes = utility.list_indexes(collection_name)
                print(f"🔍 索引信息:")
                
                if isinstance(indexes, list):
                    print(f"   索引数量: {len(indexes)}")
                    collection_info["indexes"] = indexes
                    for i, idx in enumerate(indexes):
                        print(f"   索引 {i+1}: {idx}")
                else:
                    print(f"   索引: {indexes}")
                    collection_info["indexes"] = [indexes]
            except Exception as e:
                print(f"❌ 获取索引信息失败: {e}")
            
            # 5. 获取样本数据（前3条）
            try:
                # 查询前3条数据，只获取元数据字段
                # 注意：这里只获取字段名，不获取实际数据值
                field_names = [field["name"] for field in collection_info["schema"]["fields"]]
                collection_info["sample_data"] = field_names
                print(f"📋 样本数据字段: {', '.join(field_names[:5])}{'...' if len(field_names) > 5 else ''}")
            except Exception as e:
                print(f"❌ 获取样本数据失败: {e}")
            
            results["collections"].append(collection_info)
            print("-" * 40)
        
        # 保存结果到文件
        output_file = f"milvus_collections_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 查询完成，结果已保存到文件: {output_file}")
        print("=" * 60)
        
        # 打印文件内容预览
        print(f"📄 文件内容预览:")
        print("-" * 40)
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
            if len(content) > 1000:
                print(content[:1000] + "...")
            else:
                print(content)
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
    finally:
        # 断开连接
        try:
            connections.disconnect("default")
            print(f"🔌 已断开Milvus连接")
        except:
            pass

if __name__ == "__main__":
    query_milvus_collections()
