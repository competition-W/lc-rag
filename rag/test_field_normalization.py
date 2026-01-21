#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证字段名规范化效果
"""

import logging
from services.processors.excel_processor import ExcelProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_field_normalization():
    """测试字段名规范化效果"""
    logger.info("=" * 50)
    logger.info("开始测试：字段名规范化效果")
    logger.info("=" * 50)
    
    # 创建Excel处理器实例
    processor = ExcelProcessor(department="market")
    
    # 测试包含特殊字符的字段名
    test_columns = [
        "实验方案\n（解离/抽核）",  # 包含换行符和中文括号
        "组织重量\n（数值）",  # 包含换行符和中文括号
        "定性描述\n（几根/几条等）",  # 包含换行符和中文括号
        "注释结果（组织特异性注释）",  # 包含中文括号
        "细胞总量\n（万）",  # 包含换行符和中文括号
    ]
    
    logger.info("测试字段名列表：")
    for col in test_columns:
        logger.info(f"  - '{col}'")
    
    logger.info("\n" + "-" * 30)
    logger.info("规范化结果：")
    
    for col in test_columns:
        normalized = processor._normalize_column_name(col)
        logger.info(f"  '{col}' -> '{normalized}'")
    
    logger.info("\n" + "=" * 50)
    logger.info("测试结束")
    logger.info("=" * 50)

if __name__ == "__main__":
    test_field_normalization()
