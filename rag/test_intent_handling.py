#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试查询意图处理逻辑
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.prompt_manager import prompt_manager

def test_intent_handling(query_text):
    """
    模拟查询服务中的意图识别和提示词选择逻辑
    """
    lower_query = query_text.lower()
    
    # 定义问题类型关键词
    numeric_query_keywords = ["数据指标", "实验指标", "指标", "统计", "统计分析", "数值", "数值型", "数值指标"]
    non_numeric_query_keywords = ["注释结果", "细胞注释", "注释", "结果", "如何", "怎么样", "解离", "抽核", 
                                  "鉴定", "有没有", "存在", "T细胞", "细胞类型", "细胞", "鉴定到", "检测到", 
                                  "包含", "含有", "有哪些", "有什么"]
    
    # 确定查询类型
    is_numeric_query = any(keyword in lower_query for keyword in numeric_query_keywords)
    is_annotation_query = any(keyword in lower_query for keyword in non_numeric_query_keywords)
    
    # 额外检查：如果查询包含细胞类型相关关键词，强制标记为注释查询
    cell_type_keywords = ["T细胞", "B细胞", "巨噬细胞", "心肌细胞", "内皮细胞", "免疫细胞", 
                         "细胞", "细胞类型", "细胞亚型", "亚群"]
    if any(keyword in lower_query for keyword in cell_type_keywords):
        is_annotation_query = True
        is_numeric_query = False  # 优先使用注释查询类型
    
    # 根据查询类型选择提示词
    if is_annotation_query:
        prompt_config = prompt_manager.get_prompt("annotation_query")
        prompt_type = "annotation_query"
    else:
        prompt_config = prompt_manager.get_prompt("general_query")
        prompt_type = "general_query"
    
    return {
        "query": query_text,
        "is_numeric_query": is_numeric_query,
        "is_annotation_query": is_annotation_query,
        "prompt_type": prompt_type,
        "prompt_name": prompt_config["name"]
    }

# 测试不同类型的查询
print("=== 测试意图处理逻辑 ===")

# 测试用例
test_cases = [
    "小鼠心脏有没有鉴定到T细胞",
    "小鼠心脏的实验指标如何",
    "小鼠心脏的细胞类型分布",
    "实验的结团率是多少",
    "心肌细胞的数量统计",
    "如何进行细胞解离",
    "样本的基因中位数是多少"
]

for test_case in test_cases:
    result = test_intent_handling(test_case)
    print(f"\n查询: {result['query']}")
    print(f"- 是否数值型查询: {result['is_numeric_query']}")
    print(f"- 是否注释查询: {result['is_annotation_query']}")
    print(f"- 选择的提示词类型: {result['prompt_type']}")
    print(f"- 提示词名称: {result['prompt_name']}")

print("\n✅ 意图处理逻辑测试完成！")
