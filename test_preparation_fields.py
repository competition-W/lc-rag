#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试样本制备字段提取，特别是多级表头下的样本准备字段
"""

import io
import pandas as pd
from rag.services.processors.excel_processor import ExcelProcessor

async def test_preparation_fields_extraction():
    """测试样本制备字段提取"""
    print("\n🔍 测试样本制备字段提取修复")
    
    # 创建测试用的多级表头DataFrame
    # 模拟"单细胞样本类型细分保存方式"表的结构
    data = {
        ('产品一级目录', '产品一级目录'): ['单细胞产品', '单细胞产品'],
        ('产品二级目录', '产品二级目录'): ['组织样本', '组织样本'],
        ('产品三级目录', '产品三级目录'): ['新鲜组织', '新鲜组织'],
        ('样本信息', '组织类型'): ['肝脏', '肝脏'],
        # 模拟一级表头"样本准备"下的重复列名
        ('样本准备', '建议送样量'): ['20mg', '50mg'],  # 风险送样量
        ('样本准备', '建议送样量'): ['50mg', '100mg'],  # 合格送样量
        ('样本准备', '定性描述'): ['约指甲盖大小', '约花生米大小'],  # 风险描述
        ('样本准备', '定性描述'): ['约花生米大小', '约黄豆大小'],  # 合格描述
        ('样本准备', '样本准备方法'): ['参考文档A', '参考文档A'],
        ('注意事项', '取样送样的注意事项'): ['取样后立即冷冻', '取样后立即冷冻'],
        ('备注', '备注'): ['测试数据', '测试数据'],
    }
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 测试数据：添加带"风险"和"合格"关键字的列
    test_data = {
        '产品一级目录': '单细胞产品',
        '产品二级目录': '组织样本',
        '产品三级目录': '新鲜组织',
        '组织类型': '肝脏',
        '风险送样量': '20mg',
        '合格送样量': '50mg',
        '风险定性描述': '约指甲盖大小',
        '合格定性描述': '约花生米大小',
        '样本准备方法': '参考文档A',
        '取样送样的注意事项': '取样后立即冷冻',
        '备注': '测试数据',
    }
    
    # 创建单行测试DataFrame
    test_df = pd.DataFrame([test_data])
    
    # 创建ExcelProcessor实例
    processor = ExcelProcessor(department="test")
    
    # 测试_process_preparation_data方法
    print("\n📋 测试_process_preparation_data方法")
    from pandas import Series
    
    # 将DataFrame行转换为Series
    test_row = test_df.iloc[0]
    
    # 调用处理方法
    nodes = processor._process_preparation_data(test_row, 0, "test.xlsx", "测试表", "test_user")
    
    # 打印结果
    print(f"处理结果: 生成了 {len(nodes)} 个节点")
    print()
    
    for i, node in enumerate(nodes):
        print(f"节点 {i+1} 元数据:")
        # 检查关键字段
        key_fields = [
            "product_level1", "product_level2", "product_level3",
            "tissue_type_prep",
            "recommended_amount_risk", "recommended_amount_qualified",
            "qualitative_description_risk", "qualitative_description_qualified",
            "sample_preparation_method_doc", "sampling_notes", "notes_full_text"
        ]
        
        for field in key_fields:
            value = node.metadata.get(field)
            status = "✅" if value is not None else "❌"
            print(f"   {status} {field}: {value}")
        print()
    
    # 检查是否所有关键字段都被提取
    all_fields_found = True
    for node in nodes:
        for field in key_fields:
            if node.metadata.get(field) is None:
                all_fields_found = False
                break
        if not all_fields_found:
            break
    
    if all_fields_found:
        print("🎉 所有关键字段都已正确提取！")
    else:
        print("⚠️  部分关键字段未提取到！")
    
    return all_fields_found

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_preparation_fields_extraction())
    exit(0 if success else 1)