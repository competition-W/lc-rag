#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.services.processors.excel_processor import ExcelProcessor

def test_sheet_type_detection():
    """测试表格类型检测功能"""
    print("测试1: 表格类型检测")
    
    # 创建ExcelProcessor实例
    processor = ExcelProcessor(department="test_department")
    
    # 测试用例：不同的工作表名称
    test_cases = [
        # 应该检测为project_experience的情况
        ("项目经验汇总.xlsx", "实验数据", ["实验平台", "物种", "样本大类"], "project_experience"),
        ("project_data.xlsx", "Sheet1", ["细胞总量", "结团率", "细胞活率"], "project_experience"),
        
        # 应该检测为sample_preparation的情况
        ("样本制备指南.xlsx", "Sheet1", ["产品一级目录", "样本大类", "组织类型"], "sample_preparation"),
        ("sample_guide.xlsx", "Sheet1", ["建议送样量", "定性描述"], "sample_preparation"),
        
        # 修复后的测试用例：单细胞样本类型细分保存方式
        ("test_file.xlsx", "单细胞样本类型细分保存方式", ["样本类型", "保存方式"], "sample_preparation"),
        ("test_file.xlsx", "样本类型细分", ["样本类型", "组织类型"], "sample_preparation"),
        ("test_file.xlsx", "保存方式指南", ["保存方式", "样本类型"], "sample_preparation"),
        
        # 应该检测为general的情况
        ("general_data.xlsx", "Sheet1", ["列1", "列2", "列3"], "general"),
    ]
    
    passed = 0
    for i, (filename, sheet_name, columns, expected) in enumerate(test_cases):
        result = processor._detect_table_type(filename, sheet_name, columns)
        status = "✅" if result == expected else "❌"
        print(f"{status} 测试用例 {i+1}: {filename} - {sheet_name}")
        print(f"   预期: {expected}, 实际: {result}")
        if result == expected:
            passed += 1
    
    print(f"\n测试完成: {passed}/{len(test_cases)} 测试通过")
    
    if passed == len(test_cases):
        print("🎉 所有测试通过！表格类型检测功能正常。")
        return True
    else:
        print("❌ 部分测试失败！请检查表格类型检测逻辑。")
        return False

if __name__ == "__main__":
    test_sheet_type_detection()
