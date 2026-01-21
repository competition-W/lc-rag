#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试动态上下文构建功能
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

# 创建测试数据
async def test_dynamic_context():
    logger.info("🚀 开始测试动态上下文构建功能")
    
    # 创建测试数据
    test_data = [
        {
            "metadata": {
                "platform": "10X单细胞3‘转录本-抽核(V3试剂)",
                "filename": "test.xlsx",
                "sheet_name": "Sheet1",
                "species": "小鼠",
                "category": "冻存组织",
                "tissue": "心脏",
                "col_zu_zhi_zhong_liang_shu_zhi": "262.00",
                "col_shu_ju_liang": "35225",
                "col_ji_yin_zhong_wei_shu": "1272",
                "cell_annotation_result": "心肌细胞、内皮细胞、成纤维细胞、免疫细胞",
                "is_lysis": "否",
                "is_dead_removal": "否",
                "storage_method": "4℃"
            }
        },
        {
            "metadata": {
                "platform": "10X单细胞3‘转录本-抽核(V3试剂)",
                "filename": "test.xlsx",
                "sheet_name": "Sheet1",
                "species": "小鼠",
                "category": "冻存组织",
                "tissue": "心脏",
                "col_zu_zhi_zhong_liang_shu_zhi": "250.00",
                "col_shu_ju_liang": "32150",
                "col_ji_yin_zhong_wei_shu": "1180",
                "cell_annotation_result": "心肌细胞、内皮细胞、成纤维细胞、周细胞",
                "is_lysis": "否",
                "is_dead_removal": "否",
                "storage_method": "4℃"
            }
        },
        {
            "metadata": {
                "platform": "10X单细胞3‘转录本-抽核(V3试剂)",
                "filename": "test.xlsx",
                "sheet_name": "Sheet1",
                "species": "小鼠",
                "category": "冻存组织",
                "tissue": "心脏",
                "col_zu_zhi_zhong_liang_shu_zhi": "275.00",
                "col_shu_ju_liang": "38450",
                "col_ji_yin_zhong_wei_shu": "1350",
                "cell_annotation_result": "心肌细胞、内皮细胞、成纤维细胞、免疫细胞、周细胞",
                "is_lysis": "否",
                "is_dead_removal": "否",
                "storage_method": "4℃"
            }
        }
    ]
    
    # 测试1: 数值型查询
    logger.info("\n=== 测试1: 数值型查询 ===")
    numeric_query = "小鼠心脏冻存组织实验指标如何？"
    logger.info(f"🔍 查询: {numeric_query}")
    
    numeric_result = await _generate_summary(
        query_text=numeric_query,
        context_nodes=test_data,
        intent="sample_query",
        all_retrieved_rows=test_data
    )
    
    logger.info(f"✅ 生成的回答: {numeric_result}")
    logger.info(f"📏 回答长度: {len(numeric_result)} 字符")
    
    # 测试2: 非数值型查询
    logger.info("\n=== 测试2: 非数值型查询 ===")
    non_numeric_query = "小鼠心脏冻存组织细胞注释结果怎么样？"
    logger.info(f"🔍 查询: {non_numeric_query}")
    
    non_numeric_result = await _generate_summary(
        query_text=non_numeric_query,
        context_nodes=test_data,
        intent="sample_query",
        all_retrieved_rows=test_data
    )
    
    logger.info(f"✅ 生成的回答: {non_numeric_result}")
    logger.info(f"📏 回答长度: {len(non_numeric_result)} 字符")
    
    logger.info("\n✅ 测试完成")

if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_dynamic_context())
