#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用PyMilvus原生API测试Milvus中的实际数据量和分布
"""
import sys
import os
from pymilvus import connections, Collection, utility
from config import settings

def test_milvus_raw_count():
    """使用PyMilvus原生API测试数据量"""
    print("📊 使用PyMilvus原生API测试Milvus中的实际数据量")
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
        
        # 获取所有集合
        collections = utility.list_collections()
        print(f"✅ 成功连接，共发现 {len(collections)} 个集合")
        
        for collection_name in collections:
            print(f"\n📋 集合: {collection_name}")
            print("-" * 40)
            
            # 获取Collection对象
            coll = Collection(collection_name)
            
            # 加载集合
            coll.load()
            
            # 获取实体数量
            entity_count = coll.num_entities
            print(f"✅ 实体数量: {entity_count}")
            
            # 获取分区信息
            partitions = coll.partitions
            print(f"✅ 分区数量: {len(partitions)}")
            for part in partitions:
                print(f"   - {part.name}: {part.num_entities} 个实体")
            
            # 查询前10条数据，查看实际内容
            print(f"\n🔍 查询前10条数据的species字段:")
            
            # 使用原生查询获取species字段
            try:
                # 定义要查询的字段
                fields = ["id", "text"]
                
                # 执行查询，获取前10条数据
                results = coll.query(
                    expr="",  # 空表达式表示查询所有数据
                    output_fields=fields,
                    limit=10
                )
                
                # 统计species分布
                species_dist = {}
                total_processed = 0
                max_process = 200  # 最多处理200条数据
                
                # 分页查询，统计species分布
                offset = 0
                batch_size = 100
                
                while True:
                    # 执行分页查询
                    batch_results = coll.query(
                        expr="",
                        output_fields=fields,
                        limit=batch_size,
                        offset=offset
                    )
                    
                    if not batch_results:
                        break
                    
                    for result in batch_results:
                        total_processed += 1
                        
                        # 从text字段中提取species信息
                        text = result.get("text", "")
                        
                        # 简单提取species信息（根据格式）
                        species = "未知"
                        if "物种:" in text:
                            species_part = text.split("物种:")[1].split("|")[0].strip()
                            species = species_part
                        
                        species_dist[species] = species_dist.get(species, 0) + 1
                    
                    offset += batch_size
                    
                    if total_processed >= max_process:
                        break
                
                print(f"\n📊 处理了 {total_processed} 条数据")
                print(f"📋 species字段分布:")
                for species, count in sorted(species_dist.items(), key=lambda x: x[1], reverse=True):
                    print(f"   {species}: {count} 条")
                
                # 测试查询小鼠数据
                print(f"\n🔍 查询species='小鼠'的数据:")
                
                # 由于数据存储在text字段中，我们需要使用like查询
                # 注意：这取决于数据的实际存储格式
                mouse_expr = "text like '%物种: 小鼠%'"
                mouse_results = coll.query(
                    expr=mouse_expr,
                    output_fields=fields,
                    limit=2000
                )
                
                print(f"species='小鼠'的数据量: {len(mouse_results)}")
                
                # 测试查询包含'小鼠'的数据
                mouse_contains_expr = "text like '%小鼠%'"
                mouse_contains_results = coll.query(
                    expr=mouse_contains_expr,
                    output_fields=fields,
                    limit=2000
                )
                
                print(f"包含'小鼠'的数据量: {len(mouse_contains_results)}")
                
            except Exception as e:
                print(f"❌ 查询数据失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 释放集合
            coll.release()
        
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
    test_milvus_raw_count()
