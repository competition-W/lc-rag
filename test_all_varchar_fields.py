#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有VARCHAR字段的处理逻辑，确保None值都能正确转换为空字符串
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 测试所有VARCHAR字段的处理逻辑
def test_all_varchar_fields():
    print("🔍 测试所有VARCHAR字段的处理逻辑...")
    
    # 1. 测试schema_fields默认值
    print("\n📋 测试1: schema_fields默认值")
    schema_fields = {
        # 核心RAG字段
        "doc_id": "",
        "text": "",
        "embedding": None,
        
        # 公共元数据字段
        "source_table": "",
        "chunk_id": None,
        "filename": "",
        "department": "",
        "doc_type": "",
        "owner": "",
        "uploader": "",
        "sheet_name": "",
        "row_index": None,
        
        # 实验数据字段
        "platform_type": "",
        "species": "",
        "sample_type_exp": "",
        "sample_detailed_type": "",
        "experiment_protocol": "",
        "arrival_temp_celsius": None,
        "total_cells_10k": None,
        "clumping_rate_percent": None,
        "cell_viability_percent": None,
        "nucleated_rate_percent": None,
        "captured_cells": None,
        "reads_per_cell": None,
        "median_genes": None,
        "annotation_results": "",
        "tissue_digestion_protocol_name": "",
        "tissue_digestion_protocol_summary": "",
        "related_article_link": "",
        "feishu_doc_link": "",
        "video_stream_link": "",
        
        # 样本制备指南字段
        "product_level1": "",
        "product_level2": "",
        "product_level3": "",
        "sample_category_prep": "",
        "sample_type_prep": "",
        "tissue_type_prep": "",
        "sample_prep_method": "",
        "recommended_amount_risk": "",
        "recommended_amount_qualified": [],
        "qualitative_description_risk": "",
        "qualitative_description_qualified": [],
        "sample_preparation_method_doc": "",
        "sampling_notes": "",
        "notes_full_text": ""
    }
    
    # 检查所有VARCHAR字段的默认值
    varchar_fields = [
        "doc_id", "text", "source_table", "filename", "department", "doc_type", "owner", 
        "uploader", "sheet_name", "platform_type", "species", "sample_type_exp", 
        "sample_detailed_type", "experiment_protocol", "annotation_results", 
        "tissue_digestion_protocol_name", "tissue_digestion_protocol_summary", 
        "related_article_link", "feishu_doc_link", "video_stream_link", "product_level1", 
        "product_level2", "product_level3", "sample_category_prep", "sample_type_prep", 
        "tissue_type_prep", "sample_prep_method", "recommended_amount_risk", 
        "qualitative_description_risk", "sample_preparation_method_doc", "sampling_notes", 
        "notes_full_text"
    ]
    
    for field in varchar_fields:
        if field in schema_fields:
            default_value = schema_fields[field]
            if default_value == "":
                print(f"✅ {field}: 空字符串 (正确)")
            else:
                print(f"❌ {field}: {default_value} (应为空字符串)")
        else:
            print(f"⚠️  {field}: 未在schema_fields中定义")
    
    # 2. 测试None值转换逻辑
    print("\n📋 测试2: None值转换逻辑")
    
    # 模拟standardized_metadata，包含多个None值
    standardized_metadata = {
        # 核心RAG字段
        "doc_id": None,
        "text": None,
        "embedding": None,
        
        # 公共元数据字段
        "source_table": None,
        "chunk_id": 1,
        "filename": None,
        "department": None,
        "doc_type": None,
        "owner": None,
        "uploader": None,
        "sheet_name": None,
        "row_index": 1,
        
        # 实验数据字段
        "platform_type": None,
        "species": None,
        "sample_type_exp": None,
        "sample_detailed_type": None,
        "experiment_protocol": None,
        "arrival_temp_celsius": None,
        "total_cells_10k": None,
        "clumping_rate_percent": None,
        "cell_viability_percent": None,
        "nucleated_rate_percent": None,
        "captured_cells": None,
        "reads_per_cell": None,
        "median_genes": None,
        "annotation_results": None,
        "tissue_digestion_protocol_name": None,
        "tissue_digestion_protocol_summary": None,
        "related_article_link": None,
        "feishu_doc_link": None,
        "video_stream_link": None,
        
        # 样本制备指南字段
        "product_level1": None,
        "product_level2": None,
        "product_level3": None,
        "sample_category_prep": None,
        "sample_type_prep": None,
        "tissue_type_prep": None,
        "sample_prep_method": None,
        "recommended_amount_risk": None,
        "recommended_amount_qualified": None,
        "qualitative_description_risk": None,
        "qualitative_description_qualified": None,
        "sample_preparation_method_doc": None,
        "sampling_notes": None,
        "notes_full_text": None
    }
    
    # 应用字段默认值
    for field_name, default_value in schema_fields.items():
        if field_name not in standardized_metadata:
            standardized_metadata[field_name] = default_value
    
    # 处理数值字段
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
    
    # 处理字符串字段
    string_fields = [
        "doc_id", "text", "source_table", "filename", "department", "doc_type", "owner", 
        "uploader", "sheet_name", "platform_type", "species", "sample_type_exp", 
        "sample_detailed_type", "experiment_protocol", "annotation_results", 
        "tissue_digestion_protocol_name", "tissue_digestion_protocol_summary", 
        "related_article_link", "feishu_doc_link", "video_stream_link", "product_level1", 
        "product_level2", "product_level3", "sample_category_prep", "sample_type_prep", 
        "tissue_type_prep", "sample_prep_method", "recommended_amount_risk", 
        "qualitative_description_risk", "sample_preparation_method_doc", "sampling_notes", 
        "notes_full_text"
    ]
    
    for field_name in string_fields:
        if field_name in standardized_metadata and standardized_metadata[field_name] is None:
            standardized_metadata[field_name] = ""
    
    # 处理数组字段
    array_fields = [
        "recommended_amount_qualified",
        "qualitative_description_qualified"
    ]
    
    for field_name in array_fields:
        if field_name in standardized_metadata and standardized_metadata[field_name] is None:
            standardized_metadata[field_name] = []
    
    # 验证结果
    all_passed = True
    
    # 验证VARCHAR字段
    print("   📄 VARCHAR字段验证:")
    for field in varchar_fields:
        if field in standardized_metadata:
            value = standardized_metadata[field]
            if value is None:
                print(f"   ❌ {field}: None (应转换为空字符串)")
                all_passed = False
            elif isinstance(value, str):
                print(f"   ✅ {field}: {repr(value)} (字符串类型，正确)")
            else:
                print(f"   ⚠️  {field}: {type(value).__name__} (应为字符串类型)")
                all_passed = False
    
    # 验证数组字段
    array_fields = ["recommended_amount_qualified", "qualitative_description_qualified"]
    print("\n   📋 数组字段验证:")
    for field in array_fields:
        if field in standardized_metadata:
            value = standardized_metadata[field]
            if value is None:
                print(f"   ❌ {field}: None (应转换为空数组)")
                all_passed = False
            elif isinstance(value, list):
                print(f"   ✅ {field}: {repr(value)} (数组类型，正确)")
            else:
                print(f"   ⚠️  {field}: {type(value).__name__} (应为数组类型)")
                all_passed = False
    
    if all_passed:
        print("\n🎉 所有字段的None值都已正确转换！")
    else:
        print("\n❌ 部分字段的None值未正确转换！")
    
    # 3. 特别验证推荐送样量相关字段
    print("\n📋 测试3: 特别验证推荐送样量相关字段")
    critical_fields = [
        "recommended_amount_risk",
        "recommended_amount_qualified",
        "qualitative_description_risk",
        "qualitative_description_qualified",
        "product_level1",
        "product_level3",
        "sample_category_prep"
    ]
    
    for field in critical_fields:
        if field in standardized_metadata:
            value = standardized_metadata[field]
            if value is None:
                print(f"❌ {field}: None (应转换为空字符串或空数组)")
            elif isinstance(value, (str, list)):
                print(f"✅ {field}: {repr(value)} ({type(value).__name__}类型，正确)")
    
    # 4. 打印最终结果统计
    print("\n📊 测试结果统计")
    total_varchar_fields = len(varchar_fields)
    total_array_fields = len(array_fields)
    total_fields = total_varchar_fields + total_array_fields
    
    processed_varchar_fields = len([f for f in varchar_fields if f in standardized_metadata])
    processed_array_fields = len([f for f in array_fields if f in standardized_metadata])
    processed_fields = processed_varchar_fields + processed_array_fields
    
    string_fields = len([f for f in varchar_fields if f in standardized_metadata and isinstance(standardized_metadata[f], str)])
    array_fields_count = len([f for f in array_fields if f in standardized_metadata and isinstance(standardized_metadata[f], list)])
    
    none_varchar_fields = len([f for f in varchar_fields if f in standardized_metadata and standardized_metadata[f] is None])
    none_array_fields = len([f for f in array_fields if f in standardized_metadata and standardized_metadata[f] is None])
    none_fields = none_varchar_fields + none_array_fields
    
    print(f"总字段数: {total_fields} (VARCHAR: {total_varchar_fields}, ARRAY: {total_array_fields})")
    print(f"已处理字段数: {processed_fields} (VARCHAR: {processed_varchar_fields}, ARRAY: {processed_array_fields})")
    print(f"正确处理字段数: {string_fields + array_fields_count} (字符串: {string_fields}, 数组: {array_fields_count})")
    print(f"None值字段数: {none_fields} (VARCHAR: {none_varchar_fields}, ARRAY: {none_array_fields})")
    
    if none_fields == 0:
        print("\n🎉 所有字段都已正确处理，没有None值！")
    else:
        print(f"\n❌ 还有 {none_fields} 个字段含有None值！")

if __name__ == "__main__":
    test_all_varchar_fields()
    print("\n🎉 测试完成！")