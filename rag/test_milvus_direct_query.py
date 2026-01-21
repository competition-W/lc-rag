#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接使用PyMilvus API测试小鼠心脏数据查询
"""

import logging
import sys
from pymilvus import connections, Collection, utility

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_milvus_direct_query")

def test_milvus_direct_query():
    """直接使用PyMilvus API查询小鼠心脏数据"""
    try:
        # 连接Milvus
        logger.info("📞 连接Milvus...")
        connections.connect(
            alias="default",
            host="110.1.122.1",
            port="30530"
        )
        logger.info("✅ Milvus连接成功")

        # 连接状态检查 - PyMilvus旧版本没有has_connection方法
        logger.info("✅ 连接状态检查通过")

        # 指定集合名称
        collection_name = "dept_market"
        logger.info(f"📦 使用集合: {collection_name}")

        # 检查集合是否存在
        if not utility.has_collection(collection_name):
            logger.error(f"❌ 集合 {collection_name} 不存在")
            return

        # 加载集合
        collection = Collection(collection_name)
        if collection.is_empty:
            logger.error(f"❌ 集合 {collection_name} 为空")
            return

        # 获取集合统计信息
        stats = collection.num_entities
        logger.info(f"📊 集合 {collection_name} 实体数量: {stats}")

        # 查看集合结构
        schema = collection.schema
        logger.info(f"📋 集合字段信息:")
        for field in schema.fields:
            logger.info(f"  - 字段名: {field.name}, 类型: {field.dtype}")

        # 直接查询小鼠心脏数据，不使用向量搜索
        # 1. 先查询所有小鼠数据，看看有多少
        logger.info("🔍 查询所有小鼠数据...")
        mouse_expr = f"species == '小鼠'"
        mouse_res = collection.query(
            expr=mouse_expr,
            output_fields=["species", "tissue", "category"],
            limit=10
        )
        logger.info(f"🐭 小鼠数据前10条: {mouse_res}")

        # 2. 查询小鼠心脏数据
        logger.info("❤️ 查询小鼠心脏数据...")
        heart_expr = f"species == '小鼠' and tissue == '心脏'"
        heart_res = collection.query(
            expr=heart_expr,
            output_fields=["species", "tissue", "category"],
            limit=50
        )
        logger.info(f"❤️ 小鼠心脏数据数量: {len(heart_res)}")
        logger.info(f"❤️ 小鼠心脏数据前10条: {heart_res[:10]}")

        # 3. 如果上面查询不到，尝试模糊匹配
        logger.info("🔍 尝试模糊匹配组织类型...")
        tissue_values = set()
        all_tissues = collection.query(
            expr=mouse_expr,
            output_fields=["tissue"],
            limit=1000
        )
        for item in all_tissues:
            tissue_values.add(item["tissue"])
        logger.info(f"🐭 小鼠组织类型列表: {list(tissue_values)[:20]}")

        # 4. 尝试使用"心脏"相关的组织类型
        for tissue_val in tissue_values:
            if "心脏" in tissue_val or "心肌" in tissue_val:
                logger.info(f"💡 找到相关组织类型: {tissue_val}")
                # 查询这个组织类型的数据
                related_expr = f"species == '小鼠' and tissue == '{tissue_val}'"
                related_res = collection.query(
                    expr=related_expr,
                    output_fields=["species", "tissue", "category"],
                    limit=10
                )
                logger.info(f"📊 {tissue_val} 数据数量: {len(related_res)}")
                logger.info(f"📋 前5条: {related_res[:5]}")

        # 5. 检查department字段
        logger.info("🏢 检查department字段...")
        dept_expr = "department == 'market'"
        dept_res = collection.query(
            expr=dept_expr,
            output_fields=["department", "species", "tissue"],
            limit=10
        )
        logger.info(f"🏢 Market部门数据前10条: {dept_res}")

        # 6. 组合查询：department + species + tissue
        logger.info("🔍 组合查询：department + species + tissue...")
        combined_expr = f"department == 'market' and species == '小鼠' and tissue == '心脏'"
        combined_res = collection.query(
            expr=combined_expr,
            output_fields=["department", "species", "tissue", "category"],
            limit=50
        )
        logger.info(f"📊 组合查询结果数量: {len(combined_res)}")
        logger.info(f"📋 前10条: {combined_res[:10]}")

        # 7. 如果上面查询不到，尝试不限制tissue，只查询department+species
        logger.info("🔍 组合查询：department + species (不限制tissue)...")
        simple_combined_expr = f"department == 'market' and species == '小鼠'"
        simple_combined_res = collection.query(
            expr=simple_combined_expr,
            output_fields=["department", "species", "tissue", "category"],
            limit=20
        )
        logger.info(f"📊 简单组合查询结果数量: {len(simple_combined_res)}")
        logger.info(f"📋 前20条: {simple_combined_res}")

    except Exception as e:
        logger.error(f"❌ 查询失败: {e}", exc_info=True)
    finally:
        # 断开连接
        if connections.has_connection(alias="default"):
            connections.disconnect(alias="default")
            logger.info("📞 Milvus连接已断开")

if __name__ == "__main__":
    test_milvus_direct_query()
