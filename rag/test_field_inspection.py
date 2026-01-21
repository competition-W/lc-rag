#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查Milvus中实际存储的字段名，确认实验方案字段的真实名称
"""
import logging
from pymilvus import connections, Collection, utility
from config import settings, get_collection_name

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("test_field_inspection")

def connect_to_milvus():
    """连接到Milvus"""
    if not connections.has_connection("default"):
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            user=settings.MILVUS_USER,
            password=settings.MILVUS_PASSWORD
        )
    logger.info("✅ Milvus连接成功")

def inspect_collection():
    """检查集合结构和数据"""
    # 使用正确的方式获取集合名称
    collection_name = get_collection_name("market")
    
    if not utility.has_collection(collection_name):
        logger.error(f"❌ 集合 {collection_name} 不存在")
        return
    
    collection = Collection(collection_name)
    logger.info(f"✅ 成功获取集合: {collection_name}")
    
    # 加载集合
    collection.load()
    logger.info("✅ 集合加载成功")
    
    # 获取集合结构
    schema = collection.schema
    logger.info("\n=== 集合结构 ===")
    for field in schema.fields:
        logger.info(f"字段名: {field.name}, 类型: {field.dtype}, 主键: {field.is_primary}")
    
    # 查询少量数据，查看实际字段名和值
    logger.info("\n=== 示例数据（前5条）===")
    
    # 先执行一个简单查询，获取所有字段
    results = collection.query(
        expr="department == 'market'",
        output_fields=["*"],
        limit=5
    )
    
    if results:
        # 打印第一条数据的所有字段名
        logger.info(f"\n所有字段名: {list(results[0].keys())}")
        
        # 寻找与实验方案相关的字段
        logger.info("\n=== 查找实验方案相关字段 ===")
        for result in results[:3]:
            logger.info("\n示例数据:")
            # 打印所有字段和值
            for key, value in result.items():
                # 特别关注包含'syfa'或'方案'相关的字段
                if 'syfa' in key.lower() or '方案' in str(key) or 'protocol' in key.lower():
                    logger.info(f"  {key}: {value}")
    else:
        logger.warning("⚠️ 查询返回0条结果")
    
    # 尝试查询小鼠心脏数据，查看实际字段值
    logger.info("\n=== 查询小鼠心脏数据 ===")
    mouse_heart_results = collection.query(
        expr="department == 'market' and species == '小鼠' and tissue == '心脏'",
        output_fields=["*"],
        limit=5
    )
    
    if mouse_heart_results:
        logger.info(f"找到 {len(mouse_heart_results)} 条小鼠心脏数据")
        for i, result in enumerate(mouse_heart_results[:3]):
            logger.info(f"\n结果 {i+1}:")
            # 打印关键字段
            key_fields = ["species", "tissue", "category"]
            for field in key_fields:
                if field in result:
                    logger.info(f"  {field}: {result[field]}")
            # 打印所有其他字段
            logger.info("  其他字段:")
            for key, value in result.items():
                if key not in key_fields and key not in ["department", "text", "id", "vector_id"]:
                    logger.info(f"    {key}: {value}")
    else:
        logger.warning("⚠️ 未找到小鼠心脏数据")

def main():
    """主函数"""
    logger.info("🚀 开始检查Milvus字段...")
    
    try:
        connect_to_milvus()
        inspect_collection()
    except Exception as e:
        logger.exception(f"❌ 检查过程中发生错误: {e}")
    finally:
        # 断开连接
        if connections.has_connection("default"):
            connections.disconnect("default")
        logger.info("✅ Milvus连接已断开")

if __name__ == "__main__":
    main()
