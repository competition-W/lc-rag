#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接查询Milvus，了解实际存储的字段名
"""
import logging
from pymilvus import connections, Collection, utility
from config import settings

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("test_direct_query")

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

def direct_query():
    """直接执行Milvus查询"""
    collection_name = "dept_market"
    
    if not utility.has_collection(collection_name):
        logger.error(f"❌ 集合 {collection_name} 不存在")
        return
    
    collection = Collection(collection_name)
    logger.info(f"✅ 成功获取集合: {collection_name}")
    
    # 加载集合
    collection.load()
    logger.info("✅ 集合加载成功")
    
    # 先执行一个简单查询，获取所有字段
    logger.info("\n=== 执行简单查询获取所有字段 ===")
    
    # 只查询前3条数据，获取字段名
    results = collection.query(
        expr="department == 'market'",
        output_fields=["*"],
        limit=3
    )
    
    if results:
        logger.info(f"找到 {len(results)} 条数据")
        # 打印第一条数据的所有字段名
        first_result = results[0]
        logger.info(f"\n所有字段名: {list(first_result.keys())}")
        
        # 寻找实验方案相关的字段
        logger.info("\n=== 寻找实验方案相关字段 ===")
        for key, value in first_result.items():
            if 'syfa' in key.lower() or '方案' in str(key) or 'protocol' in key.lower():
                logger.info(f"  {key}: {value}")
        
        # 打印完整数据
        logger.info("\n=== 完整数据示例 ===")
        for i, result in enumerate(results[:1]):
            logger.info(f"\n结果 {i+1}:")
            for key, value in result.items():
                logger.info(f"  {key}: {value}")
        
        # 尝试执行简单的小鼠心脏查询
        logger.info("\n=== 执行小鼠心脏查询 ===")
        mouse_heart_results = collection.query(
            expr="department == 'market' and species == '小鼠' and tissue == '心脏' and category == '冻存组织'",
            output_fields=["*"],
            limit=5
        )
        
        logger.info(f"小鼠心脏数据: {len(mouse_heart_results)} 条")
        
        if mouse_heart_results:
            # 查看这些数据的实验方案字段
            logger.info("\n=== 小鼠心脏数据的实验方案字段 ===")
            for i, result in enumerate(mouse_heart_results[:3]):
                logger.info(f"\n结果 {i+1}:")
                # 查找实验方案相关字段
                for key, value in result.items():
                    if 'syfa' in key.lower() or '方案' in str(key) or 'protocol' in key.lower():
                        logger.info(f"  实验方案相关字段: {key} = {value}")
                # 打印主要字段
                logger.info(f"  species: {result.get('species')}")
                logger.info(f"  tissue: {result.get('tissue')}")
                logger.info(f"  category: {result.get('category')}")
    else:
        logger.warning("⚠️ 查询返回0条结果")

def main():
    """主函数"""
    logger.info("🚀 开始直接查询Milvus...")
    
    try:
        connect_to_milvus()
        direct_query()
    except Exception as e:
        logger.exception(f"❌ 查询过程中发生错误: {e}")
    finally:
        # 断开连接
        if connections.has_connection("default"):
            connections.disconnect("default")
        logger.info("✅ Milvus连接已断开")

if __name__ == "__main__":
    main()
