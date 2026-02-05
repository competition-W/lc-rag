#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excel处理器测试脚本
用于验证Excel文件处理和字段映射功能
"""
import io
import os
import sys
from services.processors.excel_processor import ExcelProcessor

async def test_excel_processor():
    """测试Excel处理器功能"""
    print("开始测试Excel处理器...")
    
    # 初始化处理器
    processor = ExcelProcessor(department="market")
    
    # 测试文件路径
    test_file_path = "test_data/20260127-单细胞时空组学标准化材料表格梳理v1.xlsx"
    
    if not os.path.exists(test_file_path):
        print(f"错误: 测试文件不存在: {test_file_path}")
        return False
    
    print(f"使用测试文件: {test_file_path}")
    
    try:
        # 读取文件数据
        with open(test_file_path, 'rb') as f:
            file_data = f.read()
        
        # 处理文件
        result = await processor.process(
            file_data=file_data,
            filename=os.path.basename(test_file_path),
            user_id="test_user"
        )
        
        if result.success:
            print(f"✅ 处理成功! 生成了 {result.total_chunks} 个节点")
            
            # 检查前3个节点的metadata
            if result.chunks:
                print("\n查看前3个节点的metadata:")
                for i, node in enumerate(result.chunks[:3]):
                    print(f"\n节点 {i+1}:")
                    print(f"文本内容: {node.text[:200]}...")
                    print(f"Metadata键: {list(node.metadata.keys())[:10]}...")
                    
                    # 检查关键字段
                    key_fields = ['species', 'sample_type', 'platform_type', 'prep_method']
                    for field in key_fields:
                        if field in node.metadata:
                            print(f"  {field}: {node.metadata[field]}")
            
            return True
        else:
            print(f"❌ 处理失败: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_excel_processor())
