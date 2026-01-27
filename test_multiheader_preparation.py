#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试多级表头下的样本制备字段提取
"""

import io
import pandas as pd
from rag.services.processors.excel_processor import ExcelProcessor

async def test_multiheader_preparation_fields():
    """测试多级表头下的样本制备字段提取"""
    print("\n🔍 测试多级表头下的样本制备字段提取")
    
    # 创建测试用的多级表头DataFrame
    # 模拟真实的"单细胞样本类型细分保存方式"表结构
    # 第一级表头: 产品一级目录, 产品二级目录, 产品三级目录, 样本准备, 样本准备, 样本准备, 样本准备, 注意事项, 备注
    # 第二级表头: 产品一级目录, 产品二级目录, 产品三级目录, 建议送样量, 建议送样量, 定性描述, 定性描述, 取样送样的注意事项, 备注
    
    # 创建多级表头
    columns = pd.MultiIndex.from_tuples([
        ('产品一级目录', '产品一级目录'),
        ('产品二级目录', '产品二级目录'),
        ('产品三级目录', '产品三级目录'),
        ('样本准备', '建议送样量'),  # 风险送样量
        ('样本准备', '建议送样量'),  # 合格送样量
        ('样本准备', '定性描述'),    # 风险描述
        ('样本准备', '定性描述'),    # 合格描述
        ('样本准备', '样本准备方法'),
        ('注意事项', '取样送样的注意事项'),
        ('备注', '备注'),
    ])
    
    # 测试数据
    data = [
        ['单细胞产品', '组织样本', '新鲜组织', '20mg', '50mg', '约指甲盖大小', '约花生米大小', '参考文档A', '取样后立即冷冻', '测试数据'],
        ['单细胞产品', '组织样本', '冻存组织', '10mg', '30mg', '约米粒大小', '约绿豆大小', '参考文档B', '取样后-80℃保存', '测试数据2'],
    ]
    
    # 创建DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # 输出测试数据结构
    print("\n📊 测试数据结构:")
    print("多级表头:")
    print(df.columns)
    print("\n数据:")
    print(df)
    
    # 创建ExcelProcessor实例
    processor = ExcelProcessor(department="test")
    
    # 测试_process_sheet方法，这是真实处理流程会调用的方法
    print("\n📋 测试_process_sheet方法")
    
    # 模拟表格类型检测
    table_type = "sample_preparation"
    
    # 处理多级表头
    flat_columns = []
    column_mapping = {}
    category_mapping = {}
    col_name_counts = {}
    
    # 扁平化多级表头，处理重复列名
    for col in df.columns:
        main_col = str(col[1]) if len(col) > 1 and pd.notna(col[1]) else str(col[0])
        
        if main_col in col_name_counts:
            col_name_counts[main_col] += 1
            unique_col = f"{main_col}_{col_name_counts[main_col]}"
            combined_col = f"{str(col[0]) if pd.notna(col[0]) else ''}_{unique_col}"
        else:
            col_name_counts[main_col] = 0
            combined_col = f"{str(col[0]) if pd.notna(col[0]) else ''}_{main_col}"
        
        flat_columns.append(combined_col)
        column_mapping[col] = combined_col
        category_mapping[combined_col] = {
            "level_1": str(col[0]) if pd.notna(col[0]) else "",
            "level_2": main_col,
            "main_col": main_col,
            "group_index": col_name_counts[main_col] if main_col in col_name_counts else 0
        }
    
    # 重命名列
    df.columns = flat_columns
    
    print("\n🔄 扁平化后的列名:")
    print(df.columns)
    print("\n扁平化后的数据:")
    print(df)
    
    # 遍历数据行，测试处理
    print("\n🔍 处理每行数据:")
    for idx, row in df.iterrows():
        print(f"\n行 {idx+1}:")
        print(row)
        
        # 调用处理方法
        nodes = processor._process_preparation_data(row, idx, "test.xlsx", "测试表", "test_user")
        
        for i, node in enumerate(nodes):
            print(f"\n   生成节点 {i+1}:")
            # 检查关键字段
            key_fields = [
                "product_level1", "product_level2", "product_level3",
                "recommended_amount_risk", "recommended_amount_qualified",
                "qualitative_description_risk", "qualitative_description_qualified",
                "sample_preparation_method_doc"
            ]
            
            for field in key_fields:
                value = node.metadata.get(field)
                status = "✅" if value is not None else "❌"
                print(f"      {status} {field}: {value}")
    
    print("\n🎉 多级表头测试完成！")
    return True

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_multiheader_preparation_fields())
    exit(0 if success else 1)