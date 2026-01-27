#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from rag.services.processors.excel_processor import ExcelProcessor
from rag.tasks.document_tasks_async import process_document_task_async

def test_comprehensive_preparation_fix():
    """综合测试样本制备字段提取修复"""
    print("综合测试样本制备字段提取修复")
    
    # 创建ExcelProcessor实例
    processor = ExcelProcessor(department="test_department")
    
    # 模拟一个包含多级表头的样本制备行
    # 假设列名已经被扁平化处理
    test_row_data = {
        # 扁平化的多级表头列名
        "产品一级目录_产品一级": "单细胞产品",
        "产品二级目录_产品二级": "组织样本",
        "产品三级目录_产品三级": "新鲜组织",
        "样本大类_样本大类": "组织样本",
        "样本类型_样本类型细分": "新鲜组织",
        "组织类型_组织": "肝脏",
        "样本处理方式_解离": "解离",
        "建议送样量1_风险": "20mg",
        "定性描述1_风险": "约指甲盖大小",
        "建议送样量2_合格": "50mg",
        "定性描述2_合格": "约花生米大小",
        "样本准备方法_方法": "参考文档A",
        "取样送样的注意事项_注意": "取样后立即冷冻",
        "备注_说明": "测试数据",
        
        # 直接列名
        "产品一级目录": "单细胞产品",
        "产品二级目录": "组织样本",
        "产品三级目录": "新鲜组织",
        "样本类型": "新鲜组织",
        "组织类型": "肝脏",
    }
    
    # 转换为Series
    row = pd.Series(test_row_data)
    
    # 处理数据
    nodes = processor._process_preparation_data(
        row=row,
        row_index=1,
        filename="test_file.xlsx",
        sheet_name="单细胞样本类型细分保存方式",
        user_id="test_user"
    )
    
    # 检查结果
    print(f"\n处理结果: 生成了 {len(nodes)} 个节点")
    
    # 检查所有关键制备字段
    required_fields = [
        "product_level1", "product_level2", "product_level3", "tissue_type_prep",
        "recommended_amount_risk", "recommended_amount_qualified",
        "qualitative_description_risk", "qualitative_description_qualified",
        "sample_preparation_method_doc", "sampling_notes", "notes_full_text"
    ]
    
    all_fields_found = set()
    for i, node in enumerate(nodes):
        print(f"\n节点 {i+1} 元数据:")
        for field in required_fields:
            if field in node.metadata:
                value = node.metadata[field]
                status = "✅" if value is not None and value != "" and value != [] else "❌"
                print(f"   {status} {field}: {value}")
                if value is not None and value != "" and value != []:
                    all_fields_found.add(field)
    
    print(f"\n综合检查结果:")
    print(f"共找到 {len(all_fields_found)} / {len(required_fields)} 个必填字段")
    
    # 检查缺失的字段
    missing_fields = [field for field in required_fields if field not in all_fields_found]
    if missing_fields:
        print(f"\n❌ 缺失字段: {missing_fields}")
        return False
    else:
        print(f"\n✅ 所有必填字段都已找到！")
        return True

if __name__ == "__main__":
    if test_comprehensive_preparation_fix():
        print("\n🎉 综合测试通过！所有修复都已生效。")
        sys.exit(0)
    else:
        print("\n❌ 综合测试失败！")
        sys.exit(1)
