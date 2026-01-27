#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试样本制备指南表格的字段提取功能
"""
import json
import pandas as pd
from rag.services.processors.excel_processor import ExcelProcessor


def test_sample_preparation_fields_extraction():
    """测试样本制备指南表格的字段提取"""
    print("🔍 测试样本制备指南表格字段提取")
    
    # 创建测试数据，模拟用户提供的示例
    test_data = {
        "产品一级目录": "单细胞空间组学",
        "产品二级目录": "单细胞测序",
        "产品三级目录": "单细胞Flex（10x）",
        "样本大类": "实体组织类样本",
        "样本类型": "石蜡包埋组织",
        "组织类型": "皮肤",
        "样本处理方式": "抽核",
        "建议送样量1": "不合格：约250mg；",
        "建议送样量2": "风险：约350mg；",
        "建议送样量3": "合格：约800mg；",
        "定性描述1": "不合格：一个食指指甲盖大小（1×1cm）；",
        "定性描述2": "风险：一个大拇指指甲盖大小（1.5×1.5cm）；",
        "定性描述3": "合格：两个大拇指指甲盖大小（2×2cm）；",
        "取样送样的注意事项": "",
        "样本准备方法": "石蜡组织送样方法sop",
        "备注": "不合格：可以尝试，细胞量接近上机捕获极限，坏死类组织需根据坏死部位占比来提高送样量；\n风险：可以尝试，细胞量一般可满足上机要求，坏死类组织需根据坏死部位占比来提高送样量；\n合格：细胞量完全足够，满足上机捕获要求，坏死类组织需根据坏死部位占比来提高送样量；"
    }
    
    # 创建ExcelProcessor实例
    processor = ExcelProcessor(department="test")
    
    # 转换为Series
    test_row = pd.Series(test_data)
    
    # 测试_normalize_column_name方法
    print("\n📋 测试列名标准化:")
    test_columns = [
        "产品一级目录", "产品二级目录", "产品三级目录",
        "样本大类", "样本类型", "组织类型", "样本处理方式",
        "建议送样量1", "建议送样量2", "建议送样量3",
        "定性描述1", "定性描述2", "定性描述3",
        "样本准备方法", "备注"
    ]
    
    for col in test_columns:
        normalized = processor._normalize_column_name(col)
        print(f"  {col} -> {normalized}")
    
    # 验证产品信息映射
    assert processor._normalize_column_name("产品一级目录") == "product_level1", "产品一级目录映射错误"
    assert processor._normalize_column_name("产品二级目录") == "product_level2", "产品二级目录映射错误"
    assert processor._normalize_column_name("产品三级目录") == "product_level3", "产品三级目录映射错误"
    
    # 验证样本信息映射
    assert processor._normalize_column_name("样本大类") == "sample_category_prep", "样本大类映射错误"
    assert processor._normalize_column_name("样本类型") == "sample_type_prep", "样本类型映射错误"
    assert processor._normalize_column_name("组织类型") == "tissue_type_prep", "组织类型映射错误"
    assert processor._normalize_column_name("样本处理方式") == "sample_prep_method", "样本处理方式映射错误"
    
    # 验证送样要求映射
    assert processor._normalize_column_name("建议送样量1") == "recommended_amount_1", "建议送样量1映射错误"
    assert processor._normalize_column_name("定性描述1") == "qualitative_description_1", "定性描述1映射错误"
    
    print("\n✅ 列名标准化测试通过")
    print("\n🎉 所有测试通过！")
    return True


if __name__ == "__main__":
    test_sample_preparation_fields_extraction()