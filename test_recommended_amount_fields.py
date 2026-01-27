#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试带序号的建议送样量和定性描述字段处理
"""
import sys
import os
import pandas as pd

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.services.processors.excel_processor import ExcelProcessor


async def test_recommended_amount_fields():
    """测试带序号的建议送样量和定性描述字段处理"""
    print("\n🔍 测试带序号的建议送样量和定性描述字段处理")
    
    # 模拟用户描述的场景：
    # 建议送样量1    建议送样量2    建议送样量3    定性描述1    定性描述2    定性描述3
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
    print(f"\n列名: {list(df.columns)}")

    # 创建ExcelProcessor实例
    processor = ExcelProcessor(department="test")

    # 测试字段映射
    print("\n📋 测试字段映射")
    
    test_columns = ['建议送样量1', '建议送样量2', '建议送样量3', '定性描述1', '定性描述2', '定性描述3']
    
    for col in test_columns:
        mapped_field = processor._normalize_column_name(col)
        print(f"✅ '{col}' -> '{mapped_field}'")
        
    # 验证映射结果
    expected_mappings = {
        '建议送样量1': 'recommended_amount_1',
        '建议送样量2': 'recommended_amount_2',
        '建议送样量3': 'recommended_amount_3',
        '定性描述1': 'qualitative_description_1',
        '定性描述2': 'qualitative_description_2',
        '定性描述3': 'qualitative_description_3'
    }
    
    all_passed = True
    for col, expected in expected_mappings.items():
        result = processor._normalize_column_name(col)
        if result == expected:
            print(f"✅ 验证通过: '{col}' -> '{result}'")
        else:
            print(f"❌ 验证失败: '{col}' -> '{result}' (期望: '{expected}')")
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有字段映射验证通过！")
        return True
    else:
        print("\n❌ 字段映射验证失败！")
        return False


if __name__ == "__main__":
    """运行测试"""
    import asyncio
    
    try:
        success = asyncio.run(test_recommended_amount_fields())
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
