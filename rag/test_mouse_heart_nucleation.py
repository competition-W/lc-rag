#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试小鼠心脏冻存组织抽核方案的实验数据查询
"""

import logging
import sys
from utils.auth import AuthContext

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_mouse_heart_nucleation")

async def test_mouse_heart_nucleation():
    """测试小鼠心脏冻存组织抽核方案的实验数据查询"""
    logger.info("🔍 开始测试：查询小鼠心脏冻存组织抽核方案的实验数据")
    
    # 创建认证上下文
    auth = AuthContext(
        user_id="1",
        department="market",
        user_role=None
    )
    
    # 测试查询
    query_text = "查询小鼠心脏冻存组织抽核方案的实验数据"
    
    try:
        # 解析查询
        from services.query_parser import parse_user_query
        logger.info(f"📝 解析查询：{query_text}")
        search_term, filters, intent = await parse_user_query(query_text, auth)
        logger.info(f"✅ 解析结果：")
        logger.info(f"   - 搜索词：{search_term}")
        logger.info(f"   - 过滤条件：{filters}")
        logger.info(f"   - 意图：{intent}")
        
        # 调用查询服务
        from services.query_service import unified_query_service
        logger.info(f"🚀 调用查询服务...")
        result = await unified_query_service(
            auth=auth,
            query_text=search_term,
            column_filters=filters,
            llm_top_k=8,
            semantic_top_k=15,
            intent=intent
        )
        
        logger.info(f"✅ 查询结果：")
        logger.info(f"   - 模式：{result['mode']}")
        logger.info(f"   - 回答：{result['answer']}")
        logger.info(f"   - 最佳结果数：{len(result['sources'])}")
        logger.info(f"   - 所有结果数：{len(result['all_rows'])}")
        
        # 打印前5条结果的关键信息
        if result['all_rows']:
            logger.info("📋 前5条结果：")
            for i, row in enumerate(result['all_rows'][:5]):
                metadata = row.get('metadata', {})
                logger.info(f"   [{i+1}] {metadata.get('species')} {metadata.get('tissue')} {metadata.get('category')} {metadata.get('experiment_protocol')}")
        
        # 验证结果
        if len(result['all_rows']) > 0:
            logger.info("🎉 测试通过！检索到小鼠心脏冻存组织抽核方案的实验数据")
            return True
        else:
            logger.error("❌ 测试失败！未检索到小鼠心脏冻存组织抽核方案的实验数据")
            return False
            
    except Exception as e:
        logger.error(f"❌ 测试过程中发生错误：{e}", exc_info=True)
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_mouse_heart_nucleation())
    sys.exit(0 if success else 1)
