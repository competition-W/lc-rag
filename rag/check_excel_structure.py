#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查Excel表格结构脚本
用于查看实际的列名和样本类型字段内容
"""
import pandas as pd
import os

def check_excel_structure():
    """检查Excel表格结构"""
    print("开始检查Excel表格结构...")
    
    # 测试文件路径
    test_file_path = "test_data/20260127-单细胞时空组学标准化材料表格梳理v1.xlsx"
    
    if not os.path.exists(test_file_path):
        print(f"错误: 测试文件不存在: {test_file_path}")
        return
    
    print(f"检查文件: {test_file_path}")
    
    try:
        # 读取Excel文件的所有sheet
        xl = pd.ExcelFile(test_file_path)
        print(f"\n文件包含的sheet: {xl.sheet_names}")
        
        for sheet_name in xl.sheet_names:
            print(f"\n=== 检查sheet: {sheet_name} ===")
            
            # 读取sheet数据，使用第2行作为列名（处理多级表头）
            df = pd.read_excel(test_file_path, sheet_name=sheet_name, header=1, nrows=3)
            
            # 打印列名
            print(f"\n列名 ({len(df.columns)} 列):")
            for i, col in enumerate(df.columns):
                print(f"  {i+1}. '{col}'")
            
            # 打印前几行数据
            print(f"\n前3行数据:")
            print(df.head(3))
    
    except Exception as e:
        print(f"❌ 检查过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_excel_structure()
