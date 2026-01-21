#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证实验指标提取功能
"""

import logging
from services.query_service import _format_node_for_llm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_experimental_metrics_extraction():
    """测试实验指标提取功能"""
    logger.info("=" * 50)
    logger.info("开始测试：实验指标提取功能")
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
    
    # 测试实验指标提取
    query_text = "小鼠心脏冻存组织实验指标如何"
    result = _format_node_for_llm(test_node, query_text)
    
    logger.info(f"查询文本: {query_text}")
    logger.info(f"提取结果:\n{result}")
    
    # 检查是否包含实验指标
    assert "实验指标" in result, "提取结果中缺少实验指标部分"
    assert "col_jie_tuan_lv" in result, "提取结果中缺少结团率"
    assert "col_xi_bao_huo_lv" in result, "提取结果中缺少细胞活率"
    assert "col_you_he_lv" in result, "提取结果中缺少有核率"
    
    # 检查是否不包含注释结果（因为查询是关于实验指标的）
    assert "注释结果" not in result, "提取结果中不应包含注释结果"
    
    logger.info("\n✅ 实验指标提取测试通过！")
    logger.info("\n" + "=" * 50)
    logger.info("测试结束")
    logger.info("=" * 50)

if __name__ == "__main__":
    test_experimental_metrics_extraction()
