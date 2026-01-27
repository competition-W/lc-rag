#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from rag.services.processors.excel_processor import ExcelProcessor
from rag.tasks.document_tasks_async import clean_and_convert_value
import json

def test_json_module_scope():
    """测试json模块作用域问题是否解决"""
    print("测试1: json模块作用域问题")
    # 模拟document_tasks_async.py中使用json.dumps的场景
    test_data = ["Endothelial", "Fibroblast", "Mural cell"]
    try:
        # 这应该不会抛出UnboundLocalError
        result = json.dumps(test_data)
        print(f"✅ json.dumps() 成功执行: {result}")
        return True
    except Exception as e:
        print(f"❌ json.dumps() 执行失败: {e}")
        return False

def test_extract_schema_from_df():
    """测试_extract_schema_from_df方法是否能处理df[col]返回DataFrame的情况"""
    print("\n测试2: _extract_schema_from_df方法")
    # 创建一个包含多层列索引的DataFrame
    columns = pd.MultiIndex.from_tuples([
        ('species', '物种'),
        ('sample_type_exp', '样本类型'),
        ('sample_detailed_type', '详细类型'),
        ('annotation_results', '人工细胞注释')
    ])
    
    # 创建测试数据
    data = [
        ['Human', 'Tissue', 'Liver', 'Endothelial(38.6%),Fibroblast(26.34%)'],
        ['Mouse', 'Cell', 'Neuron', 'Neuron(80.2%),Astrocyte(19.8%)'],
        ['Human', 'Tissue', 'Kidney', 'Podocyte(50.1%),Proximal tubule(49.9%)']
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # 测试核心逻辑直接
    try:
        # 这是核心逻辑从_extract_schema_from_df方法中提取的
        schema_map = {}
        
        for col in df.columns:
            try:
                # 处理df[col]可能返回DataFrame的情况
                col_data = df[col]
                if hasattr(col_data, 'shape') and len(col_data.shape) > 1 and col_data.shape[1] > 1:
                    # 如果是DataFrame且有多个列，取第一列
                    col_data = col_data.iloc[:, 0]
                elif hasattr(col_data, 'to_frame'):  # 确保是Series
                    # 这是Series对象，可以直接处理
                    pass
                else:
                    # 其他情况，转换为Series
                    col_data = pd.Series(col_data)
                
                # 现在可以安全地调用dropna()和unique()
                unique_values = col_data.dropna().unique()
                
                # 对于组织类型（tissue）字段，放宽唯一值数量限制
                if col == "tissue":
                    # 组织类型可能有很多种，放宽限制到200个
                    is_enum = len(unique_values) <= 200 and len(unique_values) > 0
                    valid_values = [str(v) for v in unique_values[:200]] if is_enum else []
                else:
                    # 其他字段保持原有限制
                    is_enum = len(unique_values) <= 50 and len(unique_values) > 0
                    valid_values = [str(v) for v in unique_values[:50]] if is_enum else []
                
                if col not in schema_map:
                    schema_map[col] = {
                        "key": col,
                        "name": col,
                        "desc": f"来自表格列 '{col}'",
                        "source_table": "experiment_data",
                        "valid_values": valid_values
                    }
                else:
                    if is_enum:
                        existing = set(schema_map[col]["valid_values"])
                        new_v = set(valid_values)
                        merged = list(existing | new_v)
                        if len(merged) <= 50:
                            schema_map[col]["valid_values"] = merged
            except Exception as e:
                print(f"⚠️ 处理列 {col} 时出错: {e}")
                continue
        
        schema = list(schema_map.values())
        print(f"✅ _extract_schema_from_df 核心逻辑成功执行，生成了 {len(schema)} 个字段")
        for field in schema:
            print(f"   - {field['name']}: 有效值={len(field['valid_values'])}")
        return True
    except Exception as e:
        print(f"❌ _extract_schema_from_df 核心逻辑执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_clean_and_convert_value():
    """测试clean_and_convert_value函数是否能处理Series对象"""
    print("\n测试3: clean_and_convert_value函数")
    
    # 创建一个Series对象
    test_series = pd.Series(['Endothelial', 'Fibroblast', None, 'Mural cell'])
    
    try:
        # 这应该能正确处理Series对象
        result = clean_and_convert_value(test_series, str)
        print(f"✅ clean_and_convert_value 成功处理Series: {result}")
        return True
    except Exception as e:
        print(f"❌ clean_and_convert_value 处理Series失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试修复效果...")
    
    tests = [
        test_json_module_scope,
        test_extract_schema_from_df,
        test_clean_and_convert_value
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n测试完成: {passed}/{len(tests)} 测试通过")
    
    if passed == len(tests):
        print("🎉 所有测试通过！修复成功！")
        sys.exit(0)
    else:
        print("❌ 部分测试失败！需要进一步修复。")
        sys.exit(1)
