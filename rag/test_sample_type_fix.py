#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试样本类型字段修复和实验指标显示
验证样本类型字段是否能正确映射到动态字段中，以及实验指标是否能正确显示
"""
import asyncio
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.processors.excel_processor import ExcelProcessor

async def test_sample_type_field():
    """测试样本类型字段是否能正确映射到动态字段，以及实验指标是否能正确显示"""
    try:
        # 读取测试文件
        test_file_path = r"D:\LC-BIO\lc-rag-V2\rag\test_data\20260127-单细胞时空组学标准化材料表格梳理v1.xlsx"
        
        if not os.path.exists(test_file_path):
            print(f"测试文件不存在: {test_file_path}")
            return
        
        # 读取文件数据
        with open(test_file_path, 'rb') as f:
            file_data = f.read()
        
        # 创建Excel处理器
        processor = ExcelProcessor(department="market")
        
        # 处理文件 - 指定处理单细胞项目经验查询表格
        result = await processor.process(
            file_data=file_data,
            filename="test.xlsx",
            user_id="system",
            sheet_name="单细胞项目经验查询（人工细胞注释20260115）"
        )
        
        print(f"处理结果: 成功={result.success}, 总节点数={result.total_chunks}")
        
        # 检查节点中的metadata和文本内容
        for i, node in enumerate(result.chunks[:10]):  # 检查前10个节点
            print(f"\n节点 {i+1}:")
            print(f"文本内容: {node.text}")
            print(f"Metadata 键值:")
            # 检查样本类型相关字段
            for key, value in node.metadata.items():
                if key in ['sample_type_exp', 'sample_type_prep', 'species', 'sample_detailed_type']:
                    print(f"  {key}: {value}")
            # 检查实验指标相关字段
            print(f"实验指标相关字段:")
            for key, value in node.metadata.items():
                if key in ['sampling_temperature', 'arrival_temp_celsius', 'cell_count', 'total_cells_10k', 'clustering_rate', 'clumping_rate_percent', 'captured_cells', 'captured_cell_count', 'median_genes', 'median_genes_per_cell']:
                    print(f"  {key}: {value}")
            
            # 检查full_row_json中的相关字段
            if 'full_row_json' in node.metadata:
                import json
                try:
                    full_row = json.loads(node.metadata['full_row_json'])
                    print(f"Full row中的相关字段:")
                    for key in ['到样温度', '细胞总量', '结团率']:
                        if key in full_row:
                            print(f"  {key}: {full_row[key]}")
                except Exception as e:
                    print(f"解析full_row_json失败: {e}")
                    
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_sample_type_field())
