#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证提示词管理器
"""

import logging
from services.prompt_manager import prompt_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_prompt_manager():
    """测试提示词管理器"""
    logger.info("=" * 50)
    logger.info("开始测试：提示词管理器")
    logger.info("=" * 50)
    
    # 测试获取所有支持的意图
    intents = prompt_manager.list_intents()
    logger.info(f"支持的意图列表: {intents}")
    
    # 测试获取提示词模板
    for intent in intents:
        prompt = prompt_manager.get_prompt(intent)
        logger.info(f"\n--- 意图: {intent} ---")
        logger.info(f"名称: {prompt['name']}")
        logger.info(f"提示词长度: {len(prompt['system_prompt'])} 字符")
        logger.info(f"提示词前100字符: {prompt['system_prompt'][:100]}...")
    
    # 测试添加新的提示词模板
    test_intent = "test_intent"
    test_name = "测试意图"
    test_prompt = "这是一个测试提示词模板"
    prompt_manager.add_prompt(test_intent, test_name, test_prompt)
    
    # 验证新添加的提示词
    new_prompt = prompt_manager.get_prompt(test_intent)
    logger.info(f"\n--- 新添加的意图: {test_intent} ---")
    logger.info(f"名称: {new_prompt['name']}")
    logger.info(f"提示词: {new_prompt['system_prompt']}")
    
    logger.info("\n" + "=" * 50)
    logger.info("测试结束")
    logger.info("=" * 50)

if __name__ == "__main__":
    test_prompt_manager()
