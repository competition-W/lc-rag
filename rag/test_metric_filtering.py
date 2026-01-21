#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据指标和实验指标的过滤功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.query_service import _aggregate_numeric_metrics

def test_metric_filtering():
    """
    测试指标过滤功能
    """
    # 创建模拟数据
    mock_data = [
        {
            "metadata": {
                "col_shu_ju_liang": "10000",  # 数据指标
                "col_ji_yin_zhong_wei_shu": "5000",  # 数据指标
                "col_xbzl\n（w）": "1000",  # 实验指标
                "col_jie_tuan_lv": "5",  # 实验指标
                "col_xi_bao_huo_lv": "90"  # 实验指标
            }
        },
        {
            "metadata": {
                "col_shu_ju_liang": "20000",  # 数据指标
                "col_ji_yin_zhong_wei_shu": "6000",  # 数据指标
                "col_xbzl\n（w）": "2000",  # 实验指标
                "col_jie_tuan_lv": "3",  # 实验指标
                "col_xi_bao_huo_lv": "95"  # 实验指标
            }
        }
    ]
    
    print("=== 测试指标过滤功能 ===")
    
    # 测试1：数据指标查询
    print("\n1. 数据指标查询：")
    result = _aggregate_numeric_metrics(mock_data, "数据指标")
    print(result)
    
    # 检查是否只包含数据指标
    assert "数据量" in result, "数据指标查询应包含数据量"
    assert "基因中位数" in result, "数据指标查询应包含基因中位数"
    assert "细胞总量" not in result, "数据指标查询不应包含细胞总量"
    assert "结团率" not in result, "数据指标查询不应包含结团率"
    
    # 测试2：实验指标查询
    print("\n2. 实验指标查询：")
    result = _aggregate_numeric_metrics(mock_data, "实验指标")
    print(result)
    
    # 检查是否只包含实验指标
    assert "细胞总量" in result, "实验指标查询应包含细胞总量"
    assert "结团率" in result, "实验指标查询应包含结团率"
    assert "细胞活率" in result, "实验指标查询应包含细胞活率"
    assert "数据量" not in result, "实验指标查询不应包含数据量"
    assert "基因中位数" not in result, "实验指标查询不应包含基因中位数"
    
    # 测试3：默认查询（包含所有指标）
    print("\n3. 默认查询：")
    result = _aggregate_numeric_metrics(mock_data, "")
    print(result)
    
    # 检查是否包含所有指标
    assert "数据量" in result, "默认查询应包含数据量"
    assert "基因中位数" in result, "默认查询应包含基因中位数"
    assert "细胞总量" in result, "默认查询应包含细胞总量"
    assert "结团率" in result, "默认查询应包含结团率"
    assert "细胞活率" in result, "默认查询应包含细胞活率"
    
    print("\n✅ 所有测试通过！")

if __name__ == "__main__":
    test_metric_filtering()
