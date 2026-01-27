#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试annotation_results字段的处理逻辑
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.tasks.document_tasks_async import parse_annotation_results

# 测试1: 测试parse_annotation_results函数
def test_parse_annotation_results():
    print("📋 测试1: 测试parse_annotation_results函数")
    
    # 正常情况
    annotation_text = "Endothelial(38.6%),Fibroblast(26.34%),Adipocyte(12.52%)"
    cell_types_list, cell_type_details = parse_annotation_results(annotation_text)
    print(f"✅ 正常情况 - 细胞类型列表: {cell_types_list}")
    print(f"✅ 正常情况 - 细胞类型详情: {cell_type_details}")
    
    # 空字符串情况
    cell_types_list, cell_type_details = parse_annotation_results("")
    print(f"✅ 空字符串情况 - 细胞类型列表: {cell_types_list}")
    print(f"✅ 空字符串情况 - 细胞类型详情: {cell_type_details}")
    
    # None情况
    try:
        cell_types_list, cell_type_details = parse_annotation_results(None)
        print(f"✅ None情况 - 细胞类型列表: {cell_types_list}")
        print(f"✅ None情况 - 细胞类型详情: {cell_type_details}")
    except Exception as e:
        print(f"❌ None情况 - 出错: {e}")

# 测试2: 测试字符串字段None值处理逻辑
def test_string_field_handling():
    print("\n📋 测试2: 测试字符串字段None值处理逻辑")
    
    # 模拟standardized_metadata
    standardized_metadata = {
        "department": "market",
        "filename": "test.xlsx",
        "chunk_id": 1,
        "annotation_results": None,
        "tissue_digestion_protocol_name": None,
        "arrival_temp_celsius": None,
        "total_cells_10k": None
    }
    
    # 模拟字符串字段处理
    string_fields = [
        "annotation_results",
        "tissue_digestion_protocol_name",
        "tissue_digestion_protocol_summary",
        "related_article_link",
        "feishu_doc_link",
        "video_stream_link"
    ]
    
    for field_name in string_fields:
        if field_name in standardized_metadata and standardized_metadata[field_name] is None:
            standardized_metadata[field_name] = ""
    
    # 模拟数值字段处理
    numeric_fields = {
        "arrival_temp_celsius": 0.0,
        "total_cells_10k": 0.0,
        "clumping_rate_percent": 0.0,
        "cell_viability_percent": 0.0,
        "nucleated_rate_percent": 0.0,
        "captured_cells": 0,
        "reads_per_cell": 0,
        "median_genes": 0,
        "row_index": 0,
        "chunk_id": 0
    }
    
    for field_name, default_value in numeric_fields.items():
        if field_name in standardized_metadata and standardized_metadata[field_name] is None:
            standardized_metadata[field_name] = default_value
    
    # 打印处理结果
    print(f"✅ 处理后的annotation_results: {standardized_metadata['annotation_results']} (类型: {type(standardized_metadata['annotation_results'])})")
    print(f"✅ 处理后的tissue_digestion_protocol_name: {standardized_metadata['tissue_digestion_protocol_name']} (类型: {type(standardized_metadata['tissue_digestion_protocol_name'])})")
    print(f"✅ 处理后的arrival_temp_celsius: {standardized_metadata['arrival_temp_celsius']} (类型: {type(standardized_metadata['arrival_temp_celsius'])})")
    print(f"✅ 处理后的total_cells_10k: {standardized_metadata['total_cells_10k']} (类型: {type(standardized_metadata['total_cells_10k'])})")

if __name__ == "__main__":
    print("🔍 测试annotation_results字段处理逻辑...")
    test_parse_annotation_results()
    test_string_field_handling()
    print("\n🎉 所有测试完成！")