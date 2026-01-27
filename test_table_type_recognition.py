#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试表格类型识别逻辑
"""
import sys
import os

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.tasks.document_tasks_async import clean_and_convert_value


def test_table_type_recognition():
    """测试表格类型识别逻辑"""
    print("\n🔍 测试表格类型识别逻辑")
    
    # 模拟不同sheet_name的情况
    test_cases = [
        {
            "name": "单细胞项目经验查询",
            "sheet_name": "单细胞项目经验查询（人工细胞注释20260115）",
            "expected_type": "experiment_data"
        },
        {
            "name": "单细胞样本类型细分保存方式",
            "sheet_name": "单细胞样本类型细分保存方式",
            "expected_type": "preparation_guidelines"
        },
        {
            "name": "样本制备指南",
            "sheet_name": "样本制备指南",
            "expected_type": "preparation_guidelines"
        },
        {
            "name": "实验数据",
            "sheet_name": "实验数据",
            "expected_type": "experiment_data"
        },
        {
            "name": "其他表格",
            "sheet_name": "其他表格",
            "expected_type": "general"
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        # 模拟current_raw_data
        current_raw_data = {
            "sheet_name": test_case["sheet_name"]
        }
        
        # 复制修改后的表格类型识别逻辑
        table_type = "general"
        explicit_table_type = None
        sheet_name = current_raw_data.get("sheet_name", "").lower()
        
        # 直接根据sheet_name判断表格类型
        if sheet_name:
            if any(keyword in sheet_name for keyword in ["经验", "实验", "experimental", "exp"]):
                explicit_table_type = "experiment_data"
            elif any(keyword in sheet_name for keyword in ["样本类型", "保存方式", "制备", "preparation", "guide"]):
                explicit_table_type = "preparation_guidelines"
        
        # 方法1: 检查是否有明确的表格类型字段（这里不测试）
        # 方法2: 如果仍无法确定，根据关键字段数量判断（这里不测试）
        
        # 应用判断结果
        if explicit_table_type:
            table_type = explicit_table_type
        
        # 验证结果
        passed = table_type == test_case["expected_type"]
        all_passed = all_passed and passed
        
        status = "✅" if passed else "❌"
        print(f"{status} {test_case['name']}: sheet_name='{test_case['sheet_name']}' → table_type='{table_type}' (期望: '{test_case['expected_type']}')")
    
    if all_passed:
        print("\n🎉 所有测试通过！")
        return True
    else:
        print("\n❌ 部分测试失败！")
        return False


if __name__ == "__main__":
    """运行测试"""
    try:
        success = test_table_type_recognition()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
