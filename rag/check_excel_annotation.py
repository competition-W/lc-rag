import pandas as pd
import re

# 读取Excel文件
file_path = r'd:\LC-BIO\lc-rag-V2\rag\test_data\20260114-单细胞时空组学标准化材料表格梳理.xlsx'

print(f"🔍 正在读取Excel文件: {file_path}")

try:
    # 获取所有工作表名称
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names
    print(f"  工作表名称: {sheet_names}")
    
    # 遍历所有工作表
    for sheet_name in sheet_names:
        print(f"\n📋 处理工作表: {sheet_name}")
        
        # 读取工作表
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"  工作表形状: {df.shape}")
        
        # 检查是否包含人工注释相关的列
        columns = df.columns.tolist()
        print(f"  列名: {columns}")
        
        # 查找包含注释的列
        annotation_columns = []
        for col in columns:
            col_str = str(col).lower()
            if any(keyword in col_str for keyword in ["人工注释", "注释结果"]):
                annotation_columns.append(col)
        
        if annotation_columns:
            print(f"  找到注释列: {annotation_columns}")
            
            # 检查这些列中是否有值
            for col in annotation_columns:
                # 统计非空值数量
                non_empty_count = df[col].notna().sum()
                print(f"  列 {col} 非空值数量: {non_empty_count}")
                
                # 查看前5个非空值
                non_empty_values = df[col].dropna().head(5)
                if not non_empty_values.empty:
                    print(f"  前5个值: {non_empty_values.tolist()}")
                    
                    # 检查是否包含注释格式
                    for val in non_empty_values:
                        if re.search(r'[A-Za-z]+\(\d+\.\d+%\)', str(val)):
                            print(f"    ✅ 值 '{val}' 包含注释格式！")
                        else:
                            print(f"    ❌ 值 '{val}' 不包含注释格式")
        else:
            print("  未找到注释相关的列")
            
            # 尝试遍历所有列，查看是否有包含注释格式的单元格
            print("  🔍 尝试查找包含注释格式的单元格...")
            found = False
            for col in columns:
                for val in df[col].dropna():
                    if re.search(r'[A-Za-z]+\(\d+\.\d+%\)', str(val)):
                        print(f"    ✅ 在列 {col} 中找到注释格式: '{val}'")
                        found = True
                        break
                if found:
                    break
            if not found:
                print("    ❌ 未找到包含注释格式的单元格")
                
    print("\n✅ Excel文件检查完成！")
    
except Exception as e:
    print(f"❌ 读取Excel文件时出错: {e}")
    import traceback
    traceback.print_exc()
