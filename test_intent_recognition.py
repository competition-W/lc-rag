#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图识别测试脚本
用于测试意图识别功能的准确性，并显示详细日志
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 初始化日志
from rag.utils.logger import logger

# 导入必要的组件
from rag.services.intent_recognizer import intent_recognizer

async def test_milvus_expression_building():
    """
    测试Milvus表达式构建功能
    """
    logger.info("=" * 60)
    logger.info("Milvus表达式构建测试开始")
    logger.info("=" * 60)
    
    # 测试用例1：数值范围查询
    logger.info("\n" + "-" * 60)
    logger.info("测试用例1：数值范围查询")
    logger.info("-" * 60)
    
    filters1 = {
        "cell_viability_percent": {"op": "gt", "value": 90},
        "total_cells_10k": {"op": "lte", "value": 50},
        "species": "小鼠"
    }
    
    expr1 = intent_recognizer.build_milvus_expr(filters1)
    logger.info(f"过滤条件: {filters1}")
    logger.info(f"生成表达式: {expr1}")
    
    # 测试用例2：数组包含查询
    logger.info("\n" + "-" * 60)
    logger.info("测试用例2：数组包含查询")
    logger.info("-" * 60)
    
    filters2 = {
        "target_cell_types": ["巨噬细胞", "内皮细胞"],
        "species": "人"
    }
    
    expr2 = intent_recognizer.build_milvus_expr(filters2)
    logger.info(f"过滤条件: {filters2}")
    logger.info(f"生成表达式: {expr2}")
    
    # 测试用例3：风险查询
    logger.info("\n" + "-" * 60)
    logger.info("测试用例3：风险查询")
    logger.info("-" * 60)
    
    filters3 = {
        "risk_query": True,
        "sample_type": "新鲜组织"
    }
    
    expr3 = intent_recognizer.build_milvus_expr(filters3)
    logger.info(f"过滤条件: {filters3}")
    logger.info(f"生成表达式: {expr3}")
    
    # 测试用例4：混合查询
    logger.info("\n" + "-" * 60)
    logger.info("测试用例4：混合查询")
    logger.info("-" * 60)
    
    filters4 = {
        "cell_viability_percent": {"op": "gte", "value": 85},
        "target_cell_types": ["T细胞", "B细胞"],
        "species": "小鼠",
        "sample_detailed_type": "心脏组织"
    }
    
    expr4 = intent_recognizer.build_milvus_expr(filters4)
    logger.info(f"过滤条件: {filters4}")
    logger.info(f"生成表达式: {expr4}")
    
    # 测试用例5：处理数值范围过滤器
    logger.info("\n" + "-" * 60)
    logger.info("测试用例5：处理数值范围过滤器")
    logger.info("-" * 60)
    
    # 模拟LLM返回的结果
    intent_result = {
        "milvus_filters": {
            "cell_viability_percent": ">90",
            "total_cells_10k": "<=50",
            "species": "小鼠"
        }
    }
    
    logger.info(f"原始过滤器: {intent_result['milvus_filters']}")
    intent_recognizer._process_numeric_filters(intent_result)
    logger.info(f"处理后过滤器: {intent_result['milvus_filters']}")
    
    # 构建最终表达式
    final_expr = intent_recognizer.build_milvus_expr(intent_result['milvus_filters'])
    logger.info(f"最终表达式: {final_expr}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Milvus表达式构建测试结束")
    logger.info("=" * 60)

async def test_intent_types():
    """
    测试意图类型相关功能
    """
    logger.info("\n" + "=" * 60)
    logger.info("意图类型测试开始")
    logger.info("=" * 60)
    
    # 测试意图类型获取
    logger.info("\n" + "-" * 60)
    logger.info("测试意图类型获取")
    logger.info("-" * 60)
    
    for intent_key, intent_desc in intent_recognizer.intent_types.items():
        desc = intent_recognizer.get_intent_type(intent_key)
        logger.info(f"意图键: {intent_key} -> 描述: {desc}")
    
    # 测试未知意图
    unknown_desc = intent_recognizer.get_intent_type("unknown_intent")
    logger.info(f"未知意图: unknown_intent -> 描述: {unknown_desc}")
    
    logger.info("\n" + "=" * 60)
    logger.info("意图类型测试结束")
    logger.info("=" * 60)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_milvus_expression_building())
    asyncio.run(test_intent_types())
