#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证包含特殊字符的字段名处理
"""

import asyncio
import logging
from services.query_service import unified_query_service
from utils.auth import AuthContext

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_special_field_query():
    """测试包含特殊字符的字段名查询"""
    logger.info("=" * 50)
    logger.info("开始测试：包含特殊字符的字段名查询")
    logger.info("=" * 50)
    
    # 创建认证上下文
    auth = AuthContext(
        user_id="test_user",
        department="market"
    )
    
    # 测试查询：包含特殊字符的字段名（实验方案）
    query_text = "查询小鼠心脏冻存组织抽核方案的结团率"
    column_filters = {
        "species": "小鼠",
        "tissue": "心脏",
        "category": "冻存组织",
        "col_syfa\n（jl/ch）": "抽核",  # 包含换行符和中文括号的特殊字段名
        "col_jie_tuan_lv": "*"
    }
    
    logger.info(f"查询文本: {query_text}")
    logger.info(f"过滤条件: {column_filters}")
    
    try:
        result = await unified_query_service(
            auth=auth,
            query_text=query_text,
            column_filters=column_filters,
            intent="sample_query"
        )
        
        logger.info(f"\n✅ 查询成功！")
        logger.info(f"结果模式: {result['mode']}")
        logger.info(f"命中数据条数: {len(result['all_rows'])}")
        logger.info(f"返回最佳结果数: {len(result['sources'])}")
        
        if result['sources']:
            logger.info("\n📋 部分结果示例:")
            for i, source in enumerate(result['sources'][:3]):
                logger.info(f"\n--- 结果 {i+1} ---")
                logger.info(f"元数据: {source['metadata']}")
                logger.info(f"文本: {source['text'][:100]}...")
        
        logger.info(f"\n💬 LLM回答: {result['answer'][:100]}...")
        
    except Exception as e:
        logger.error(f"\n❌ 查询失败: {str(e)}")
    
    logger.info("\n" + "=" * 50)
    logger.info("测试结束")
    logger.info("=" * 50)

if __name__ == "__main__":
    asyncio.run(test_special_field_query())
