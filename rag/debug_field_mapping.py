#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试字段映射脚本
用于检查列名映射和字段值
"""
import pandas as pd
import os
from services.processors.excel_processor import ExcelProcessor

async def debug_field_mapping():
    """调试字段映射"""
    print("开始调试字段映射...")
    
    # 测试文件路径
    test_file_path = "test_data/20260127-单细胞时空组学标准化材料表格梳理v1.xlsx"
    
    if not os.path.exists(test_file_path):
        print(f"错误: 测试文件不存在: {test_file_path}")
        return
    
    print(f"调试文件: {test_file_path}")
    
    try:
        # 初始化处理器
        processor = ExcelProcessor(department="market")
        
        # 读取Excel文件
        xl = pd.ExcelFile(test_file_path)
        print(f"\n文件包含的sheet: {xl.sheet_names}")
        
        for sheet_name in xl.sheet_names:
            print(f"\n=== 调试sheet: {sheet_name} ===")
            
            # 读取sheet数据，使用第2行作为列名
            df = pd.read_excel(test_file_path, sheet_name=sheet_name, header=1, nrows=5)
            
            # 打印前2行数据
            print(f"\n前2行数据:")
            print(df.head(2))
            
            # 调试列名映射
            print(f"\n列名映射:")
            for col in df.columns:
                if col in ['样本类型', '到样温度', '细胞总量', '结团率']:
                    # 模拟处理器的列名标准化
                    milvus_key = processor._normalize_column_name(col)
                    print(f"  '{col}' -> '{milvus_key}'")
                    
                    # 检查字段值
                    for i, value in enumerate(df[col].head(2)):
                        print(f"    行 {i+1}: {value} (类型: {type(value).__name__})")
    
    except Exception as e:
        print(f"❌ 调试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(debug_field_mapping())
