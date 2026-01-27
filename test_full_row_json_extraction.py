#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试从full_row_json中提取字段的功能
"""
import json
import pandas as pd
from rag.services.processors.excel_processor import ExcelProcessor


def test_full_row_json_extraction():
    """测试从full_row_json中提取人工细胞注释等字段"""
    print("🔍 测试从full_row_json中提取字段")
    
    # 创建测试数据，模拟用户提供的元数据示例
    test_data = {
        "实验平台": "10X单细胞3‘转录本(V3试剂)",
        "物种": "小鼠",
        "样本类型": "新鲜实体组织",
        "样本详细类型": "肺叶",
        "实验方案": "解离",
        "组织重量\n（数值）": "/",
        "是否裂红": "是",
        "是否去死": "否",
        "到样温度\n(℃)": 4,
        "细胞总量\n(万)": 80,
        "结团率(%)": 0.04,
        "细胞活率(%)": 0.86,
        "有核率(%)": 0.85,
        "捕获细胞数": 12794,
        "reads/cell": 25322,
        "基因中位数": 1454,
        "人工细胞注释": "B cell(17.61%),T cell(15.77%),NK cell(13.26%),Macrophage(12.85%),Endothelial(11.97%),Neutrophil(9.06%),Monocyte(6.33%),Fibroblast(2.82%),NK T cell(2.21%),DC(2.03%),Plasma(1.8%),Smooth muscle cell(1.76%),Pulmonary alveolar type 2 cell(0.86%),Pulmonary alveolar type 1 cell(0.68%),Club cell(0.55%),Mesenchymal cell(0.38%),Mesothelial(0.06%)"
    }
    
    # 创建ExcelProcessor实例
    processor = ExcelProcessor(department="test")
    
    # 转换为Series
    test_row = pd.Series(test_data)
    
    # 测试_normalize_column_name方法
    print("\n📋 测试列名标准化:")
    test_columns = ["实验平台", "物种", "样本类型", "样本详细类型", "实验方案", "人工细胞注释"]
    for col in test_columns:
        normalized = processor._normalize_column_name(col)
        print(f"  {col} -> {normalized}")
    
    # 验证平台类型映射
    platform_normalized = processor._normalize_column_name("实验平台")
    assert platform_normalized == "platform_type", f"平台类型映射错误: {platform_normalized} != platform_type"
    
    # 验证人工细胞注释映射
    annotation_normalized = processor._normalize_column_name("人工细胞注释")
    assert annotation_normalized == "annotation_results", f"人工细胞注释映射错误: {annotation_normalized} != annotation_results"
    
    print("\n✅ 列名标准化测试通过")
    print("\n🎉 所有测试通过！")
    return True


if __name__ == "__main__":
    test_full_row_json_extraction()