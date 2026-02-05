#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保存查询检索到的原始条目脚本

此脚本用于执行查询并将检索到的原始条目保存到文件中，以便后续分析和查看。
"""

import asyncio
import json
import os
import sys
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from rag.services.query_service import unified_query_service
from rag.utils.auth import AuthContext
from rag.utils.logger import logger


async def save_retrieved_items(query_text, department="market", output_file=None):
    """
    执行查询并保存检索到的原始条目
    
    Args:
        query_text: 查询文本
        department: 部门名称
        output_file: 输出文件路径，如果为None则自动生成
    """
    # 创建认证上下文
    auth = AuthContext(
        user_id="system",
        department=department,
        user_role="admin"
    )
    
    logger.info(f"开始执行查询: '{query_text}' (部门: {department})")
    
    # 执行查询
    result = await unified_query_service(
        auth=auth,
        query_text=query_text,
        column_filters=None,
        llm_top_k=8,
        semantic_top_k=15,
        intent="unknown",
        websocket=None,
        parsed_query=None
    )
    
    logger.info(f"查询完成，模式: {result.get('mode', 'unknown')}")
    logger.info(f"生成的回答: {result.get('answer', '')[:100]}...")
    logger.info(f"检索到的源数量: {len(result.get('sources', []))}")
    logger.info(f"检索到的所有行数量: {len(result.get('all_rows', []))}")
    
    # 准备保存的数据
    save_data = {
        "query": query_text,
        "department": department,
        "timestamp": datetime.now().isoformat(),
        "result_mode": result.get('mode', 'unknown'),
        "answer": result.get('answer', ''),
        "sources": result.get('sources', []),
        "all_rows": result.get('all_rows', []),
        "token_stats": result.get('token_stats', {})
    }
    
    # 生成输出文件路径
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_sanitized = "".join(c for c in query_text if c.isalnum() or c in "_ -").strip()[:50]
        output_file = os.path.join(
            os.path.dirname(__file__),
            "output",
            f"retrieved_items_{timestamp}_{query_sanitized}.json"
        )
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存数据
    logger.info(f"保存检索结果到: {output_file}")
    
    # 自定义JSON编码器，处理可能的对象
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif hasattr(obj, 'node'):
                return {
                    "node": self.default(obj.node),
                    "score": getattr(obj, 'score', None)
                }
            elif hasattr(obj, 'metadata'):
                return {
                    "text": getattr(obj, 'text', ''),
                    "metadata": getattr(obj, 'metadata', {}),
                    "id": getattr(obj, 'id', None)
                }
            elif isinstance(obj, set):
                return list(obj)
            try:
                return super().default(obj)
            except:
                return str(obj)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2, cls=CustomEncoder)
    
    logger.info(f"保存完成，文件大小: {os.path.getsize(output_file)} 字节")
    return output_file


async def main():
    # 示例查询
    sample_queries = [
        "帮我查询小鼠心脏组织的实验指标",
        "帮我查询肝脏组织的样本制备指南",
        "帮我查询肺组织的实验数据"
    ]
    
    for query in sample_queries:
        try:
            output_file = await save_retrieved_items(query)
            print(f"查询 '{query}' 已保存到: {output_file}")
        except Exception as e:
            logger.error(f"执行查询时出错: {e}")
            print(f"查询 '{query}' 执行失败: {e}")
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(main())
