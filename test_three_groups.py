#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试三组数据（风险+两组合格）的处理
"""

import pandas as pd
from rag.services.processors.excel_processor import ExcelProcessor

async def test_three_groups_data():
    """测试三组数据的处理"""
    print("\n🔍 测试三组数据（风险+两组合格）的处理")
    
    # 模拟用户描述的场景：
    # 建议送样量 建议送样量 建议送样量 定性描述 定性描述 定性描述 
    # 风险：约20mg； 合格：约50mg； 合格：约250mg； 风险：大米大小； 合格：绿豆大小； 合格：黄豆大小；
    
    # 创建测试数据
    test_data = {
        '产品一级目录': '单细胞产品',
        '产品二级目录': '组织样本',
        '产品三级目录': '新鲜组织',
        '组织类型': '肝脏',
        # 三列建议送样量
        '建议送样量1': '风险：约20mg；',
        '建议送样量2': '合格：约50mg；',
        '建议送样量3': '合格：约250mg；',
        # 三列定性描述
        '定性描述1': '风险：大米大小；',
        '定性描述2': '合格：绿豆大小；',
        '定性描述3': '合格：黄豆大小；',
        '样本准备方法': '参考文档A',
        '取样送样的注意事项': '取样后立即冷冻',
        '备注': '测试数据',
    }
    
    # 创建DataFrame
    df = pd.DataFrame([test_data])
    
    # 输出测试数据
    print("\n📊 测试数据:")
    print(df)
    
    # 创建ExcelProcessor实例
    processor = ExcelProcessor(department="test")
    
    # 测试处理
    print("\n📋 测试_process_preparation_data方法")
    
    # 获取第一行数据
    test_row = df.iloc[0]
    
    # 调用处理方法
    nodes = processor._process_preparation_data(test_row, 0, "test.xlsx", "测试表", "test_user")
    
    # 打印结果
    print(f"\n处理结果: 生成了 {len(nodes)} 个节点")
    
    for i, node in enumerate(nodes):
        print(f"\n节点 {i+1} 元数据:")
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
    
    # 验证结果
    print("\n🔍 验证结果:")
    node = nodes[0]
    metadata = node.metadata
    
    # 预期结果
    expected_results = {
        "recommended_amount_risk": 20.0,
        "recommended_amount_qualified": 250.0,  # 取最大值
        "qualitative_description_risk": "大米大小；",
        "qualitative_description_qualified": "绿豆大小；; 黄豆大小；"  # 合并所有合格描述
    }
    
    # 检查结果
    all_passed = True
    for field, expected in expected_results.items():
        actual = metadata.get(field)
        if actual == expected:
            print(f"✅ {field}: {actual} (符合预期)")
        else:
            print(f"❌ {field}: 预期 {expected}，实际 {actual}")
            all_passed = False
    
    if all_passed:
        print("\n🎉 测试通过！三组数据处理正确。")
    else:
        print("\n⚠️  测试失败！三组数据处理有误。")
    
    return all_passed

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_three_groups_data())
    exit(0 if success else 1)