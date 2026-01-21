#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试token counter功能
"""

import asyncio
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入token counter
from utils.token_counter import token_counter

def test_token_counter():
    logger.info("🚀 开始测试token counter功能")
    
    # 测试1: LLM tokens统计
    logger.info("\n=== 测试1: LLM tokens统计 ===")
    input_text = "你好，我想了解一下小鼠心脏冻存组织实验指标如何？"
    output_text = "根据数据统计，小鼠心脏冻存组织的组织重量在250-275mg之间，平均值为262.33mg，数据量在32150-38450之间，基因中位数在1180-1350之间。"
    
    llm_stats = token_counter.count_llm_tokens(
        input_text=input_text,
        output_text=output_text,
        model_name="qwen-plus"
    )
    logger.info(f"✅ LLM tokens统计结果: {llm_stats}")
    
    # 测试2: Embedding tokens统计
    logger.info("\n=== 测试2: Embedding tokens统计 ===")
    embedding_text = "小鼠心脏冻存组织实验指标如何？"
    
    embedding_stats = token_counter.count_embedding_tokens(
        text=embedding_text,
        model_name="text-embedding-v4"
    )
    logger.info(f"✅ Embedding tokens统计结果: {embedding_stats}")
    
    # 测试3: Rerank tokens统计
    logger.info("\n=== 测试3: Rerank tokens统计 ===")
    query = "小鼠心脏冻存组织实验指标如何？"
    documents = [
        "小鼠心脏冻存组织的组织重量在250-275mg之间，平均值为262.33mg。",
        "小鼠心脏冻存组织的数据量在32150-38450之间，平均值为35275。",
        "小鼠心脏冻存组织的基因中位数在1180-1350之间，平均值为1267.33。"
    ]
    
    rerank_stats = token_counter.count_rerank_tokens(
        query=query,
        documents=documents,
        model_name="gte-rerank-v2"
    )
    logger.info(f"✅ Rerank tokens统计结果: {rerank_stats}")
    
    # 测试4: 获取所有统计数据
    logger.info("\n=== 测试4: 获取所有统计数据 ===")
    total_stats = token_counter.get_total_stats()
    logger.info(f"✅ 所有统计数据: {total_stats}")
    
    # 测试5: 重置统计数据
    logger.info("\n=== 测试5: 重置统计数据 ===")
    token_counter.reset_stats()
    reset_stats = token_counter.get_total_stats()
    logger.info(f"✅ 重置后统计数据: {reset_stats}")
    
    logger.info("\n✅ 所有测试完成")

if __name__ == "__main__":
    # 运行测试
    test_token_counter()
