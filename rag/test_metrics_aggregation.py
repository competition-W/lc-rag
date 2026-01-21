#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数字指标聚合功能
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
from services.query_service import _generate_summary, _aggregate_numeric_metrics

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

# 创建测试数据，包含多个样本
async def test_metrics_aggregation():
    logger.info("🚀 开始测试数字指标聚合功能")
    
    # 创建多个样本的测试数据 - best_rows（用于上下文生成）
    best_rows = [
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
                "is_lysis": "否",
                "is_dead_removal": "否",
                "storage_method": "4℃"
            }
        }
    ]
    
    # 创建更多的样本数据 - all_rows（用于指标聚合）
    all_rows = best_rows + [
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
                "col_zu_zhi_zhong_liang_shu_zhi": "280.00",
                "col_shu_ju_liang": "40123",
                "col_ji_yin_zhong_wei_shu": "1420",
                "is_lysis": "是",
                "is_dead_removal": "是",
                "storage_method": "-80℃"
            }
        },
        {
            "metadata": {
                "platform": "10X单细胞3‘转录本-抽核(V3试剂)",
                "filename": "test.xlsx",
                "sheet_name": "Sheet1",
                "species": "小鼠",
                "category": "新鲜组织",
                "tissue": "心脏",
                "col_zu_zhi_zhong_liang_shu_zhi": "245.00",
                "col_shu_ju_liang": "31890",
                "col_ji_yin_zhong_wei_shu": "1150",
                "is_lysis": "是",
                "is_dead_removal": "否",
                "storage_method": "液氮"
            }
        },
        {
            "metadata": {
                "platform": "10X单细胞3‘转录本-抽核(V3试剂)",
                "filename": "test.xlsx",
                "sheet_name": "Sheet1",
                "species": "小鼠",
                "category": "冻存组织",
                "tissue": "肝脏",
                "col_zu_zhi_zhong_liang_shu_zhi": "300.00",
                "col_shu_ju_liang": "42567",
                "col_ji_yin_zhong_wei_shu": "1500",
                "is_lysis": "否",
                "is_dead_removal": "是",
                "storage_method": "-80℃"
            }
        }
    ]
    
    query_text = "小鼠心脏冻存组织实验指标如何"
    intent = "sample_query"
    
    # 测试_aggregate_numeric_metrics函数 - 基于所有数据
    logger.info("📊 测试_aggregate_numeric_metrics函数 - 基于所有数据")
    aggregate_result = _aggregate_numeric_metrics(all_rows)
    logger.info(f"✅ 基于所有数据的聚合结果: {aggregate_result}")
    
    # 测试_aggregate_numeric_metrics函数 - 仅基于best_rows
    logger.info("📊 测试_aggregate_numeric_metrics函数 - 仅基于best_rows")
    aggregate_result_best = _aggregate_numeric_metrics(best_rows)
    logger.info(f"✅ 仅基于best_rows的聚合结果: {aggregate_result_best}")
    
    # 测试完整的_generate_summary函数 - 传入all_retrieved_rows
    logger.info("📞 测试_generate_summary函数 - 传入all_retrieved_rows")
    result = await _generate_summary(
        query_text=query_text,
        context_nodes=best_rows,  # 仅使用best_rows作为上下文
        intent=intent,
        all_retrieved_rows=all_rows  # 基于所有检索到的数据进行指标聚合
    )
    
    logger.info(f"✅ 生成的回答: {result}")
    logger.info(f"📏 回答长度: {len(result)} 字符")
    
    logger.info("✅ 测试完成")

if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_metrics_aggregation())
