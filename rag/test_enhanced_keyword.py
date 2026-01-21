#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强关键词提取测试
测试增强后的关键词提取功能，特别是针对复杂查询
"""

import logging
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from services.query_parser import parse_user_query
from utils.auth import AuthContext

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_enhanced_keyword_extraction():
    """测试增强后的关键词提取功能"""
    print("=" * 80)
    print("增强关键词提取测试")
    print("=" * 80)
    
    # 创建一个模拟的 AuthContext
    auth_context = AuthContext(
        user_id="test_user",
        department="market",
        user_role=None
    )
    
    # 测试查询列表，重点测试复杂查询
    test_queries = [
        "小鼠心脏抽核实验方案的细胞活率数据",
        "人外周血PBMC流式分选实验的活性指标",
        "猪肝脏新鲜组织解离方案的数据量",
        "大鼠肺冻存组织的基因中位数",
        "牛脾脏细胞悬液的捕获细胞数"
    ]
    
    for query in test_queries:
        print(f"\n\n{'='*60}")
        print(f"测试查询: {query}")
        print(f"{'='*60}")
        
        try:
            # 解析查询
            search_term, filters, intent = await parse_user_query(query, auth_context)
            
            print(f"1. 意图识别结果:")
            print(f"   - 识别意图: {intent}")
            print(f"   - 提取关键词: {search_term}")
            print(f"   - 提取过滤条件: {filters}")
            print(f"   - 过滤条件数量: {len(filters)}")
            
            # 验证是否提取到了关键信息
            if len(filters) >= 3:
                print(f"\n✅ 测试通过: 成功提取到了 {len(filters)} 个过滤条件")
            else:
                print(f"\n⚠️  测试警告: 只提取到了 {len(filters)} 个过滤条件")
                
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n\n{'='*80}")
    print("所有测试完成")
    print(f"{'='*80}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_enhanced_keyword_extraction())
