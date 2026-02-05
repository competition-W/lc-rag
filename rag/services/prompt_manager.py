#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提示词管理器
负责管理不同意图的提示词模板
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class PromptManager:
    """
    提示词管理器类
    管理不同意图的提示词模板
    """
    def __init__(self):
        """
        初始化提示词管理器
        """
        self.prompts = {}
        self._load_prompts()
    
    def _load_prompts(self):
        """
        从Python配置文件加载提示词
        """
        try:
            # 从Python配置文件导入提示词
            import sys
            import os
            # 添加config目录到Python路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(current_dir, "..", "config")
            config_dir = os.path.normpath(config_dir)
            sys.path.insert(0, config_dir)
            from prompts_config import PROMPTS_CONFIG
            
            logger.info("✅ 成功加载提示词配置文件: prompts_config.py")
            logger.info(f"✅ 加载的提示词类别: {list(PROMPTS_CONFIG.keys())}")
            
            # 加载响应生成提示词
            if 'response_generation' in PROMPTS_CONFIG:
                self.prompts = PROMPTS_CONFIG['response_generation']
                response_templates = list(self.prompts.keys())
                logger.info(f"✅ 加载了 {len(response_templates)} 个响应生成提示词模板: {response_templates}")
            else:
                # 使用默认提示词
                self._use_default_prompts()
            
        except Exception as e:
            logger.error(f"❌ 加载提示词配置文件失败: {e}")
            # 使用默认提示词
            self._use_default_prompts()
    
    def _use_default_prompts(self):
        """
        使用默认提示词
        """
        logger.warning("⚠️ 使用默认提示词模板")
        self.prompts = {
            "query_experiment_data": {
                "name": "实验记录查询",
                "system_prompt": "你是一位资深的生物数据分析专家。你的任务是根据检索到的数据库记录，回答用户关于生物实验数据的问题。"
            },
            "query_preparation_guidelines": {
                "name": "样本制备指南查询",
                "system_prompt": "你是一位资深的生物实验技术专家。你的任务是根据用户的查询，提供详细的实验准备指南。"
            }
        }
    
    def get_intent_detection_prompt(self) -> Optional[str]:
        """
        获取意图识别提示词
        
        Returns:
            Optional[str]: 意图识别提示词，如果不存在则返回None
        """
        try:
            # 从Python配置文件导入提示词
            import sys
            import os
            # 添加config目录到Python路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(current_dir, "..", "config")
            config_dir = os.path.normpath(config_dir)
            sys.path.insert(0, config_dir)
            from prompts_config import PROMPTS_CONFIG
            
            return PROMPTS_CONFIG.get('intent_detection', {}).get('system_prompt')
        except Exception as e:
            logger.error(f"❌ 获取意图识别提示词失败: {e}")
            return None
    
    def get_response_prompt(self, intent: str) -> Optional[Dict]:
        """
        获取指定意图的响应提示词
        
        Args:
            intent: 意图名称
        
        Returns:
            Optional[Dict]: 包含name和system_prompt的字典，如果不存在则返回None
        """
        try:
            return self.prompts.get(intent)
        except Exception as e:
            logger.error(f"❌ 获取响应提示词失败: {e}")
            return None
    
    def list_response_intents(self) -> List[str]:
        """
        获取所有支持的响应意图列表
        
        Returns:
            List[str]: 支持的响应意图名称列表
        """
        try:
            return list(self.prompts.keys())
        except Exception as e:
            logger.error(f"❌ 获取响应意图列表失败: {e}")
            return []
    
    def list_intents(self) -> List[str]:
        """
        获取所有支持的意图列表
        
        Returns:
            List[str]: 支持的意图名称列表
        """
        return list(self.prompts.keys())
    
    def get_prompt(self, intent: str) -> Dict:
        """
        获取指定意图的提示词模板
        
        Args:
            intent (str): 意图名称
            
        Returns:
            Dict: 包含name和system_prompt的字典
        """
        # 如果意图不存在，返回默认提示词
        if intent not in self.prompts:
            logger.warning(f"意图 '{intent}' 不存在，使用默认提示词")
            return self.prompts.get("query_experiment_data", {})
        
        return self.prompts[intent]
    
    def add_prompt(self, intent: str, name: str, system_prompt: str) -> None:
        """
        添加新的提示词模板
        
        Args:
            intent (str): 意图名称
            name (str): 意图中文名称
            system_prompt (str): 系统提示词模板
        """
        self.prompts[intent] = {
            "name": name,
            "system_prompt": system_prompt
        }
        logger.info(f"添加新提示词模板: {intent} - {name}")
    
    def reload_prompts(self):
        """
        重新加载提示词配置文件
        """
        logger.info("🔄 重新加载提示词配置文件")
        # 清除模块缓存，确保重新导入最新的配置
        import importlib
        try:
            import sys
            import os
            # 添加config目录到Python路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(current_dir, "..", "config")
            config_dir = os.path.normpath(config_dir)
            sys.path.insert(0, config_dir)
            import prompts_config
            importlib.reload(prompts_config)
        except Exception as e:
            logger.warning(f"⚠️ 清除模块缓存失败: {e}")
        self._load_prompts()


# 创建全局实例
prompt_manager = PromptManager()