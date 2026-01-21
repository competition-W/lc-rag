#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Milvus中实际存储的字段名
"""

import logging
import sys
from pymilvus import connections, Collection
from config import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_field_names")

def test_field_names():
    """测试Milvus中实际存储的字段名"""
    logger.info("🔍 开始测试Milvus字段名")
    
    try:
        # 连接 Milvus
        logger.info(f"📞 连接 Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            timeout=30
        )
        logger.info("✅ Milvus连接成功")

        # 指定集合名称
        collection_name = "dept_market"
        logger.info(f"📦 使用集合: {collection_name}")

        # 获取集合
        collection = Collection(collection_name)
        
        # 查看集合结构
        schema = collection.schema
        logger.info(f"📋 集合字段信息:")
        for field in schema.fields:
            logger.info(f"   - 字段名: '{field.name}', 类型: {field.dtype}")
        
        # 查询一条包含实验方案的数据
        logger.info("🔍 查询一条包含实验方案的数据...")
        expr = f"department == 'market' and species == '小鼠' and tissue == '心脏'"
        result = collection.query(
            expr=expr,
            output_fields=["*"],
            limit=1
        )
        
        if result:
            logger.info(f"📝 查询结果: {result[0]}")
            # 打印所有字段名
            logger.info(f"🔑 所有字段名: {list(result[0].keys())}")
        
        # 断开连接
        connections.disconnect("default")
        
        logger.info("🎉 测试完成")
        
    except Exception as e:
        logger.error(f"❌ 测试过程中发生错误: {e}", exc_info=True)
    finally:
        # 断开连接
        if connections.has_connection("default"):
            connections.disconnect("default")
            logger.info("📞 Milvus连接已断开")

if __name__ == "__main__":
    test_field_names()
