#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本，检查"单细胞样本类型细分保存方式"sheet的具体数据
"""
import pandas as pd
import io
import os

# 测试文件路径
file_path = r'rag\test_data\20260114-单细胞时空组学标准化材料表格梳理.xlsx'

def test_sheet_data():
    print(f"🔍 测试文件: {file_path}")
    print(f"📊 文件存在: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        print("❌ 文件不存在！")
        return
    
    try:
        # 读取所有sheet名称
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        print(f"📋 所有sheet名称: {sheet_names}")
        
        # 检查目标sheet
        target_sheet = "单细胞样本类型细分保存方式"
        if target_sheet not in sheet_names:
            print(f"❌ 目标sheet '{target_sheet}'不存在！")
            return
        
        print(f"\n📋 开始检查sheet: '{target_sheet}'")
        
        # 读取目标sheet数据
        df = pd.read_excel(file_path, sheet_name=target_sheet)
        print(f"📊 sheet形状: {df.shape} (行: {df.shape[0]}, 列: {df.shape[1]})")
        
        # 打印列名
        print(f"📋 列名: {list(df.columns)}")
        
        # 打印前20行数据
        print("\n📋 前20行数据:")
        print(df.head(20))
        
        # 检查每行是否都是空值
        empty_rows = df.isna().all(axis=1)
        print(f"\n📊 空行数量: {empty_rows.sum()} / {len(df)}")
        print(f"📊 非空行数量: {len(df) - empty_rows.sum()}")
        
        # 检查数据类型检测逻辑
        print("\n📋 表格类型检测模拟:")
        
        # 模拟表格类型检测
        def detect_table_type(sheet_name, columns):
            sheet_name_lower = sheet_name.lower()
            columns_lower = [col.lower() for col in columns]
            
            project_keywords = ["实验平台", "物种", "样本大类", "样本详细类型", "细胞总量", "结团率", "细胞活率", "基因中位数", "单细胞", "转录本", "测序", "分析", "样本信息", "实验指标", "数据指标"]
            sample_keywords = ["产品", "样本大类", "组织类型", "样本处理方式", "建议送样量", "样本准备方法", "样本保存", "样本采集"]
            
            project_keywords_lower = [kw.lower() for kw in project_keywords]
            sample_keywords_lower = [kw.lower() for kw in sample_keywords]
            
            project_score = 0
            sample_score = 0
            
            for col in columns_lower:
                for kw in project_keywords_lower:
                    if kw in col:
                        project_score += 1
                        break
                for kw in sample_keywords_lower:
                    if kw in col:
                        sample_score += 1
                        break
            
            print(f"   项目经验关键词匹配: {project_score}个")
            print(f"   样本准备关键词匹配: {sample_score}个")
            
            if project_score > sample_score:
                return "project_experience"
            elif sample_score > project_score:
                return "sample_preparation"
            else:
                return "general"
        
        detected_type = detect_table_type(target_sheet, df.columns)
        print(f"   检测到的表格类型: {detected_type}")
        
        # 模拟skip_rows计算
        if detected_type == "project_experience":
            skip_rows = 2
        else:
            skip_rows = 0
        print(f"   预计跳过行数: {skip_rows}")
        
        # 检查实际处理的数据行
        print(f"\n📋 实际处理数据行模拟 (只显示前10行非空数据):")
        processed_rows = 0
        max_display = 10
        display_count = 0
        for idx, row in df.iterrows():
            if idx < skip_rows:
                continue
            if not row.isna().all():
                processed_rows += 1
                if display_count < max_display:
                    print(f"   处理行 {idx}: {row.dropna().to_dict()}")
                    display_count += 1
        print(f"   预计处理行数: {processed_rows}")
        
        print("\n🎉 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sheet_data()