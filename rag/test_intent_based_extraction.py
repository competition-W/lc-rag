#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证基于意图的指标提取
"""

import logging
from services.query_service import _format_node_for_llm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_intent_based_extraction():
    """测试基于意图的指标提取"""
    logger.info("=" * 50)
    logger.info("开始测试：基于意图的指标提取")
    logger.info("=" * 50)
    
    # 模拟节点数据
    test_node = {
        "platform": "10X单细胞3'转录本-抽核(V3试剂)",
        "species": "小鼠",
        "category": "冻存组织",
        "tissue": "心脏",
        "col_syfa\n（jl/ch）": "抽核",
        "col_dxms\n（jg/jtd）": "半个心脏",
        "col_hszl\n（rinz）": "8.9",
        "col_kang_ti_xin_xi": "7-AAD",
        "col_lsfxfa": "分选细胞核",
        "is_lysis": "否",
        "is_dead_removal": "否",
        "storage_method": "4℃",
        "col_xbzl\n（w）": "8",
        "col_jie_tuan_lv": "0%",
        "col_xi_bao_huo_lv": "0%",
        "col_you_he_lv": "99%",
        "col_bu_huo_xi_bao_shu": "12,033",
        "col_shu_ju_liang": "35,225",
        "col_ji_yin_zhong_wei_shu": "1,272",
        "col_zsjg_zztyxzs": "Endothelial(24.94%),Fibroblast(22.02%),Adipocyte(21.15%),Macrophage(9.4%),Pericyte(5.39%),Epithelial(5.31%),Cardiomyocyte(4.58%),B cell(3.79%),NK cell(2.52%),Fibrocyte(0.64%),Schwann cell(0.25%)",
        "col_zzxhfags": "1/2 NP40",
        "department": "market",
        "uploader": "system",
        "doc_type": "excel",
        "row_index": 100,
        "chunk_id": 100,
        "owner": "system"
    }
    
    # 测试不同查询意图
    test_queries = [
        "小鼠心脏组织解离实验指标如何？",  # 只需要实验指标
        "小鼠心脏组织解离数据指标如何？",  # 只需要数据指标
        "小鼠心脏组织解离注释结果怎么样？",  # 只需要注释结果
        "小鼠心脏组织解离有没有鉴定到特定细胞？",  # 只需要注释结果
        "小鼠心脏组织解离实验方案如何？",  # 需要综合信息
    ]
    
    for query in test_queries:
        logger.info(f"\n--- 查询意图: '{query}' ---")
        formatted = _format_node_for_llm(test_node, query)
        logger.info(f"提取结果:\n{formatted}")
    
    logger.info("\n" + "=" * 50)
    logger.info("测试结束")
    logger.info("=" * 50)

if __name__ == "__main__":
    test_intent_based_extraction()
