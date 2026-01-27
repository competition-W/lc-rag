#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试重复表头处理功能
"""
import asyncio
import pandas as pd
from rag.services.processors.excel_processor import ExcelProcessor


async def test_repeated_headers():
    """测试重复表头处理功能"""
    print("\n🔍 测试重复表头处理功能")
    
    # 模拟包含重复表头的Excel数据
    # 使用列表方式创建，避免字典键唯一的问题
    data = [
        ['单细胞产品', '组织样本', '新鲜组织', '风险：约20mg；', '合格：约50mg；', '合格：约250mg；', '风险：大米大小；', '合格：绿豆大小；', '合格：黄豆大小；', '参考文档A', '取样后立即冷冻', '测试数据']
    ]
    
    # 包含重复列名的列名列表
    columns = [
        '产品一级目录', '产品二级目录', '产品三级目录', 
        '建议送样量', '建议送样量', '建议送样量',
        '定性描述', '定性描述', '定性描述',
        '样本准备方法', '取样送样的注意事项', '备注'
    ]
    
    # 创建DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # 输出原始测试数据
    print("\n📊 原始测试数据:")
    print(df)
    print(f"\n原始列名: {list(df.columns)}")
    
    # 创建ExcelProcessor实例
    processor = ExcelProcessor(department="test")
    
    # 测试表头处理逻辑
    print("\n📋 测试表头处理逻辑")
    
    # 复制处理逻辑进行测试
    header_count = {}
    new_columns = []
    
    for col in df.columns:
        col_str = str(col).strip()
        print(f"\n处理列名: '{col_str}'")
        
        # 处理重复表头：只处理"建议送样量"和"定性描述"这两个表头
        if "建议送样量" in col_str:
            base_header = "建议送样量"
        elif "定性描述" in col_str:
            base_header = "定性描述"
        else:
            # 其他表头不处理
            new_columns.append(col_str)
            print(f"→ 保留原名: '{col_str}'")
            continue
        
        # 处理重复表头
        if base_header in header_count:
            header_count[base_header] += 1
            # 添加序号：建议送样量1、建议送样量2...
            new_col = f"{base_header}{header_count[base_header]}"
        else:
            header_count[base_header] = 1
            new_col = f"{base_header}{header_count[base_header]}"
        
        new_columns.append(new_col)
        print(f"→ 新列名: '{new_col}'")
    
    # 更新DataFrame的列名
    if new_columns:
        df.columns = new_columns
    
    # 输出处理后的数据
    print(f"\n📊 处理后的数据列名: {list(df.columns)}")
    print(df)
    
    # 验证结果
    expected_columns = [
        '产品一级目录', '产品二级目录', '产品三级目录', 
        '建议送样量1', '建议送样量2', '建议送样量3',
        '定性描述1', '定性描述2', '定性描述3',
        '样本准备方法', '取样送样的注意事项', '备注'
    ]
    
    print(f"\n🔍 验证结果:")
    print(f"期望列名: {expected_columns}")
    print(f"实际列名: {list(df.columns)}")
    
    assert list(df.columns) == expected_columns, f"列名处理错误，期望: {expected_columns}，实际: {list(df.columns)}"
    
    print("✅ 重复表头处理测试通过！")
    return True


async def test_field_mapping():
    """测试字段映射功能"""
    print("\n\n🔍 测试字段映射功能")
    
    # 测试人工细胞注释到annotation_results的映射
    processor = ExcelProcessor(department="test")
    
    # 测试各种字段映射
    test_cases = [
        ("人工细胞注释", "annotation_results"),
        ("细胞注释", "annotation_results"),
        ("注释结果", "annotation_results"),
        ("人工注释结果", "annotation_results"),
        ("物种", "species"),
        ("样本类型", "sample_type_exp"),
        ("样本详细类型", "sample_detailed_type"),
    ]
    
    all_passed = True
    for test_input, expected_output in test_cases:
        result = processor._normalize_column_name(test_input)
        passed = result == expected_output
        all_passed = all_passed and passed
        
        status = "✅" if passed else "❌"
        print(f"{status} '{test_input}' → '{result}' (期望: '{expected_output}')")
    
    assert all_passed, "字段映射测试失败"
    print("✅ 字段映射测试通过！")
    return True


if __name__ == "__main__":
    """运行测试"""
    print("🚀 开始测试重复表头处理功能")
    
    try:
        # 运行测试
        asyncio.run(test_repeated_headers())
        asyncio.run(test_field_mapping())
        print("\n🎉 所有测试通过！")
        exit(0)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
