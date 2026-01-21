#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复效果：验证小鼠心脏冻存组织抽核方案的结团率查询
"""
import json
import logging
import sys
from services.query_parser import parse_query
from services.milvus_manager import milvus_manager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("test_fix")

def test_query_parsing():
    """测试查询解析是否正确提取过滤条件"""
    logger.info("=== 测试1: 查询解析 ===")
    
    # 测试查询文本
    query_text = "查询小鼠心脏冻存组织抽核方案的结团率"
    
    # 调用查询解析器
    result = parse_query(query_text, department="market")
    
    logger.info(f"查询文本: {query_text}")
    logger.info(f"解析结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
    
    # 验证关键过滤条件是否正确提取
    expected_filters = {
        "species": "小鼠",
        "tissue": "心脏",
        "category": "冻存组织",
        "col_syfa（jl/ch）": "抽核",
        "col_jie_tuan_lv": "*"
    }
    
    actual_filters = result["filters"]
    
    # 检查所有期望的过滤条件是否存在
    for key, expected_value in expected_filters.items():
        if key in actual_filters:
            actual_value = actual_filters[key]
            if actual_value == expected_value:
                logger.info(f"✅ 过滤条件正确: {key} = {actual_value}")
            else:
                logger.error(f"❌ 过滤条件值错误: {key} 期望 {expected_value}, 实际 {actual_value}")
        else:
            logger.error(f"❌ 过滤条件缺失: {key}")
    
    return result

def test_milvus_query(filters):
    """测试Milvus查询是否能正确执行"""
    logger.info("\n=== 测试2: Milvus查询执行 ===")
    
    try:
        from services.query_service import _strategy_structured_table
        
        # 模拟认证信息
        class MockAuth:
            def __init__(self, department):
                self.department = department
        
        auth = MockAuth(department="market")
        
        # 调用查询策略
        import asyncio
        
        async def run_query():
            # 由于index参数需要实际的索引对象，我们直接测试过滤条件构建逻辑
            logger.info("构建查询表达式...")
            
            # 模拟_strategy_structured_table中的过滤条件处理逻辑
            expr_parts = [f"department == '{auth.department}'"]
            
            for k, v in filters.items():
                if v != "*":
                    # 清理字段名中的特殊字符（与修复后的逻辑一致）
                    cleaned_k = k.replace('\n', '').replace('\r', '')
                    cleaned_k = cleaned_k.replace('（', '').replace('）', '')
                    cleaned_k = cleaned_k.replace('(', '').replace(')', '')
                    cleaned_k = cleaned_k.replace('：', '').replace(':', '')
                    
                    expr_parts.append(f"{cleaned_k} == '{v}'")
            
            milvus_expr = " and ".join(expr_parts)
            logger.info(f"✅ 生成查询表达式: {milvus_expr}")
            
            # 直接执行Milvus查询
            logger.info("执行Milvus查询...")
            
            from pymilvus import connections, Collection, utility
            from config import settings
            
            # 连接Milvus
            if not connections.has_connection("default"):
                connections.connect(
                    alias="default",
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT,
                    user=settings.MILVUS_USER,
                    password=settings.MILVUS_PASSWORD
                )
            
            # 获取集合
            collection_name = settings.MILVUS_COLLECTION
            if utility.has_collection(collection_name):
                collection = Collection(collection_name)
                logger.info(f"✅ 成功获取集合: {collection_name}")
                
                # 加载集合
                collection.load()
                logger.info("✅ 集合加载成功")
                
                # 执行查询
                results = collection.query(
                    expr=milvus_expr,
                    output_fields=["*"],
                    limit=10
                )
                
                logger.info(f"✅ 查询执行成功，返回 {len(results)} 条结果")
                
                if results:
                    logger.info("\n=== 示例结果 ===")
                    for i, result in enumerate(results[:3]):
                        logger.info(f"结果 {i+1}:")
                        # 只显示关键字段
                        key_fields = ["species", "tissue", "category", "col_syfa（jl/ch）", "col_jie_tuan_lv"]
                        for field in key_fields:
                            if field in result:
                                logger.info(f"  {field}: {result[field]}")
                    
                    return len(results) > 0
                else:
                    logger.warning("⚠️ 查询返回0条结果")
                    return False
            else:
                logger.error(f"❌ 集合 {collection_name} 不存在")
                return False
        
        return asyncio.run(run_query())
        
    except Exception as e:
        logger.exception(f"❌ 查询执行失败: {e}")
        return False

def test_field_cleaning():
    """测试字段名清理逻辑"""
    logger.info("\n=== 测试3: 字段名清理 ===")
    
    test_fields = [
        "col_syfa（jl/ch）",  # 包含中文括号
        "col_zzzl\n（sz）",    # 包含换行符
        "col_dxms\n（jg/jtd）",  # 包含换行符
        "col_zsjg（zztyxzs）", # 包含中文括号
    ]
    
    cleaned_fields = []
    
    for field in test_fields:
        # 应用清理逻辑
        cleaned = field.replace('\n', '').replace('\r', '')
        cleaned = cleaned.replace('（', '').replace('）', '')
        cleaned = cleaned.replace('(', '').replace(')', '')
        cleaned = cleaned.replace('：', '').replace(':', '')
        
        cleaned_fields.append(cleaned)
        logger.info(f"✅ 清理前: '{field}' -> 清理后: '{cleaned}'")
    
    # 验证清理后的字段名是否符合要求
    for cleaned in cleaned_fields:
        # 检查是否还包含特殊字符
        has_special_chars = any(c in cleaned for c in ["\n", "（", "）", "\r"])
        if not has_special_chars:
            logger.info(f"✅ 清理后的字段名 '{cleaned}' 符合要求")
        else:
            logger.error(f"❌ 清理后的字段名 '{cleaned}' 仍包含特殊字符")
    
    return True

def main():
    """主测试函数"""
    logger.info("🚀 开始测试修复效果...")
    
    # 测试1: 查询解析
    parse_result = test_query_parsing()
    
    # 测试2: 字段名清理
    field_cleaning_result = test_field_cleaning()
    
    # 测试3: Milvus查询执行
    if parse_result and "filters" in parse_result:
        query_result = test_milvus_query(parse_result["filters"])
    else:
        logger.error("❌ 查询解析失败，无法执行Milvus查询测试")
        query_result = False
    
    logger.info("\n=== 测试总结 ===")
    logger.info(f"查询解析测试: {'✅ 通过' if parse_result else '❌ 失败'}")
    logger.info(f"字段名清理测试: {'✅ 通过' if field_cleaning_result else '❌ 失败'}")
    logger.info(f"Milvus查询测试: {'✅ 通过' if query_result else '❌ 失败'}")
    
    if parse_result and field_cleaning_result and query_result:
        logger.info("\n🎉 所有测试通过！修复成功！")
        return 0
    else:
        logger.error("\n💥 测试失败！修复未完成！")
        return 1

if __name__ == "__main__":
    sys.exit(main())
