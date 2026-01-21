#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细测试小鼠心脏数据，特别是experiment_protocol字段的可能值
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
logger = logging.getLogger("test_mouse_heart_detailed")

def test_mouse_heart_detailed():
    """详细测试小鼠心脏数据"""
    logger.info("🔍 开始详细测试小鼠心脏数据")
    
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
        
        # 1. 先查询所有小鼠心脏数据，看看有多少
        logger.info("❤️ 查询所有小鼠心脏数据...")
        heart_expr = f"department == 'market' and species == '小鼠' and tissue == '心脏'"
        heart_res = collection.query(
            expr=heart_expr,
            output_fields=["species", "tissue", "category", "experiment_protocol"],
            limit=100
        )
        logger.info(f"❤️ 小鼠心脏数据数量: {len(heart_res)}")
        logger.info(f"❤️ 小鼠心脏数据前10条:")
        for i, item in enumerate(heart_res[:10]):
            logger.info(f"   [{i+1}] {item}")

        # 2. 查看experiment_protocol的所有可能值
        logger.info("🔬 查看experiment_protocol的所有可能值...")
        protocol_values = set()
        for item in heart_res:
            protocol = item.get("experiment_protocol", "")
            protocol_values.add(protocol)
        logger.info(f"🔬 experiment_protocol可能值: {list(protocol_values)}")
        
        # 3. 查看category的所有可能值
        logger.info("📋 查看category的所有可能值...")
        category_values = set()
        for item in heart_res:
            category = item.get("category", "")
            category_values.add(category)
        logger.info(f"📋 category可能值: {list(category_values)}")

        # 4. 测试不同的过滤条件组合
        logger.info("🧪 测试不同的过滤条件组合...")
        
        # 组合1: 小鼠 + 心脏 + 冻存组织（不包含experiment_protocol）
        combo1_expr = f"department == 'market' and species == '小鼠' and tissue == '心脏' and category == '冻存组织'"
        combo1_res = collection.query(
            expr=combo1_expr,
            output_fields=["species", "tissue", "category", "experiment_protocol"],
            limit=20
        )
        logger.info(f"🧪 组合1（小鼠+心脏+冻存组织）命中: {len(combo1_res)} 条")
        logger.info(f"   前5条: {combo1_res[:5]}")
        
        # 组合2: 只使用主要过滤条件
        combo2_expr = f"department == 'market' and species == '小鼠' and tissue == '心脏'"
        combo2_res = collection.query(
            expr=combo2_expr,
            output_fields=["species", "tissue", "category"],
            limit=10
        )
        logger.info(f"🧪 组合2（小鼠+心脏）命中: {len(combo2_res)} 条")
        
        # 5. 检查是否存在抽核相关的数据
        logger.info("🔍 检查是否存在抽核相关的数据...")
        nucleation_expr = f"department == 'market' and experiment_protocol like '%抽核%'"
        nucleation_res = collection.query(
            expr=nucleation_expr,
            output_fields=["species", "tissue", "category", "experiment_protocol"],
            limit=20
        )
        logger.info(f"🔍 抽核相关数据数量: {len(nucleation_res)}")
        if nucleation_res:
            logger.info(f"   前5条: {nucleation_res[:5]}")

        # 6. 检查是否存在其他实验方案
        logger.info("🔍 检查其他实验方案...")
        other_expr = f"department == 'market' and species == '小鼠' and experiment_protocol != ''"
        other_res = collection.query(
            expr=other_expr,
            output_fields=["species", "tissue", "category", "experiment_protocol"],
            limit=20
        )
        logger.info(f"🔍 有实验方案的数据数量: {len(other_res)}")
        if other_res:
            logger.info(f"   前5条: {other_res[:5]}")

        logger.info("🎉 详细测试完成")
        
    except Exception as e:
        logger.error(f"❌ 测试过程中发生错误: {e}", exc_info=True)
    finally:
        # 断开连接
        if connections.has_connection("default"):
            connections.disconnect("default")
            logger.info("📞 Milvus连接已断开")

if __name__ == "__main__":
    test_mouse_heart_detailed()
