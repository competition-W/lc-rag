#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试LLM生成功能
"""
import asyncio
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from llama_index.core import Settings
from llm.inference import DashScopeLLM
from llm.get_embedding import get_embed_model
from services.query_service import _generate_summary

# 初始化LLM
logger.info("🔧 初始化LLM...")
try:
    # 从.env文件加载API密钥
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    logger.info(f"🔍 使用API_KEY: {api_key} 初始化LLM")
    
    # 初始化LLM
    Settings.llm = DashScopeLLM(
        model_name="qwen-plus", 
        api_key=api_key
    )
    logger.info("✅ LLM 初始化成功")
    
    # 初始化Embedding
    Settings.embed_model = get_embed_model(model_name="text-embedding-v4")
    logger.info("✅ Embedding 初始化成功")
except Exception as e:
    logger.error(f"❌ 初始化失败: {e}")
    import traceback
    logger.error(f"❌ 堆栈信息: {traceback.format_exc()}")
    exit(1)

async def test_generate_summary():
    """测试_generate_summary函数"""
    logger.info("🚀 开始测试_generate_summary函数")
    
    # 准备测试数据
    test_nodes = [
        {
            "metadata": {
                "platform": "10X单细胞3‘转录本-抽核(V3试剂)",
                "filename": "test.xlsx",
                "sheet_name": "Sheet1",
                "species": "小鼠",
                "category": "冻存组织",
                "tissue": "心脏",
                "col_zu_zhi_zhong_liang_shu_zhi": "262.00",
                "col_hszl （rinz）": "/",
                "col_liu_shi_yu_fou": "/",
                "col_kang_ti_xin_xi": "/",
                "col_lsfxfa": "/",
                "is_lysis": "否",
                "is_dead_removal": "否",
                "storage_method": "4℃"
            }
        }
    ]
    
    query_text = "小鼠心脏冻存组织实验指标如何"
    intent = "sample_query"
    
    # 调用_generate_summary函数
    logger.info("📞 调用_generate_summary函数")
    result = await _generate_summary(
        query_text=query_text,
        context_nodes=test_nodes,
        intent=intent
    )
    
    logger.info(f"✅ 测试完成")
    logger.info(f"📝 生成的回答: {result}")
    logger.info(f"📏 回答长度: {len(result)} 字符")

if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_generate_summary())
