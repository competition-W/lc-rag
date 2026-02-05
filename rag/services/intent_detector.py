#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图检测器
统一处理用户查询的意图识别、实体抽取和查询解析
"""

import json
import os
import re
from typing import Tuple, Dict, Any, List
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from utils.auth import AuthContext
from utils.logger import logger
from utils.token_counter import token_counter

# 导入公共字段映射
from services.common.field_mapping import build_complete_field_mapping
# 导入提示词管理器
from .prompt_manager import prompt_manager


class IntentDetector:
    """
    意图检测器
    负责处理用户查询的意图识别、实体抽取和查询解析
    """
    
    def __init__(self):
        """
        初始化意图检测器
        """
        # 可识别的意图类型
        self.intent_types = {
            "query_experiment_data": "查询实验记录相关信息",
            "query_preparation_guidelines": "查询样本制备指南相关信息"
        }
        
        # 加载 Schema 注册表
        self.schema_registry = self._load_schema_registry()
        
        # 构建字段映射
        self.field_mapping = self._build_field_mapping()
        
        # 加载 Schema 上下文
        self.schema_context = self._load_schema_context()
        
        # 加载LLM提示词模板
        self.llm_prompt_template = prompt_manager.get_intent_detection_prompt()
        
        # 如果配置文件中没有提示词，使用默认提示词
        if not self.llm_prompt_template:
            logger.warning("⚠️ 从配置文件加载提示词失败，使用默认提示词")
            self.llm_prompt_template = ""
        else:
            logger.info("✅ 成功从配置文件加载意图识别提示词")

    
    def _load_schema_registry(self, json_path: str = None) -> Dict[str, Dict[str, str]]:
        """
        加载 Schema 注册表
        """
        if json_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, "..", "data", "schema_registry.json")
            json_path = os.path.normpath(json_path)

        registry = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    registry = json.load(f)
                logger.info(f"✅ 成功加载 Schema 注册表: {json_path}")
            except Exception as e:
                logger.error(f"❌ 读取 Schema 注册表出错: {e}")
        return registry
    
    def _build_field_mapping(self) -> Dict[str, str]:
        """
        构建字段映射
        从公共字段映射模块获取统一的字段映射关系
        """
        # 使用公共字段映射模块中的完整字段映射
        mapping = build_complete_field_mapping()
        logger.info(f"✅ 成功加载公共字段映射，共 {len(mapping)} 个映射关系")
        return mapping
    
    def _load_schema_context(self) -> str:
        """
        加载 Schema 上下文
        """
        columns = self.schema_registry.get("market", [])
        schema_lines = []
        added_keys = set()

        # 优先处理 JSON 文件里的定义
        for col in columns:
            key = col.get("key")
            if not key:
                continue
                
            name = col.get("name", key)
            valid_values = col.get("valid_values", [])

            # 过滤有效值
            clean_values = [str(v) for v in valid_values if v and str(v).strip() not in ["/", "nan"]]
            
            if clean_values:
                # 限制显示数量，防止 Token 溢出
                values_str = ", ".join(clean_values[:50]) 
                schema_lines.append(f"- 字段: `{key}` ({name})\n  可选值: [{values_str}]")
                added_keys.add(key)

        return "\n".join(schema_lines)
    
    def _llm_based_intent_recognition(self, user_query: str) -> Dict[str, Any]:
        """
        基于LLM的意图识别和实体抽取
        
        Args:
            user_query: 用户的自然语言查询
            
        Returns:
            包含意图和实体信息的字典
        """
        logger.info(f"🧠 使用基于LLM的意图识别处理查询: {user_query}")
        
        try:
            # 尝试使用新的tongyi-intent-detect-v3模型
            try:
                from llm.get_intent_model import get_intent_model, create_intent_messages
                
                # 获取意图识别模型
                intent_model = get_intent_model()
                if intent_model:
                    # 构建提示词
                    prompt = self.llm_prompt_template.replace("{{schema_context}}", self.schema_context)
                    prompt = prompt.replace("{{user_query}}", user_query)
                    
                    # 创建消息列表
                    messages = create_intent_messages(prompt, user_query)
                    
                    # 调用意图识别模型
                    response_text = intent_model(messages)
                    
                    if response_text:
                        logger.info(f"📝 意图识别模型返回结果: {response_text[:200]}...")
                        # 统计意图识别模型的tokens消耗
                        prompt_text = "\n".join([msg["content"] for msg in messages])
                        token_counter.count_llm_tokens(prompt_text, response_text, "intent-detection")
                        
                        # 解析返回的JSON
                        import json
                        # 清理返回的文本，移除可能的Markdown代码块标记
                        clean_text = response_text.strip()
                        if clean_text.startswith('```json'):
                            clean_text = clean_text[7:]
                        if clean_text.endswith('```'):
                            clean_text = clean_text[:-3]
                        clean_text = clean_text.strip()
                        result = json.loads(clean_text)
                        
                        # 验证结果结构
                        if not isinstance(result, dict):
                            raise ValueError("意图识别模型返回结果格式错误")
                        
                        # 确保返回的字段完整，处理不同模型返回的字段名
                        # 映射模型返回的字段名到内部使用的字段名
                        if "intent" in result:
                            result["query_intent"] = result.pop("intent")
                        elif "query_intent" not in result:
                            result["query_intent"] = "general_query"
                        
                        if "extracted_entities" in result:
                            result["milvus_filters"] = result.pop("extracted_entities")
                        elif "milvus_filters" not in result:
                            result["milvus_filters"] = {}
                        
                        if "requested_fields" not in result:
                            result["requested_fields"] = []
                        
                        # 验证字段名是否正确
                        valid_fields = {
                            "species",                  # 物种
                            "sample_type_exp",          # 样本类型（项目经验表）
                            "sample_type_prep",          # 样本类型（样本准备表）
                            "tissue_type_prep",          # 组织类型
                            "sample_category_prep",      # 样本大类
                            "sample_detailed_type",     # 详细样本类型
                            "experiment_protocol",      # 实验方案（项目经验表）
                            "sample_prep_method",       # 样本处理方式（样本准备表）
                            "platform_type",            # 实验平台
                            "product_level3",            # 产品类型
                            "source_table",             # 数据源表
                            "department"                # 部门
                        }
                        
                        # 过滤无效字段
                        filtered_filters = {}
                        for field, value in result["milvus_filters"].items():
                            if field in valid_fields:
                                filtered_filters[field] = value
                            else:
                                logger.warning(f"⚠️ 过滤掉无效字段: {field}")
                        
                        result["milvus_filters"] = filtered_filters
                        
                        logger.info(f"✅ 意图识别模型成功: {result['query_intent']}")
                        logger.info(f"✅ 实体抽取成功: {result['milvus_filters']}")
                        
                        # 特殊情况：组织消化方法属于项目经验查询
                        user_query_lower = user_query.lower()
                        if "组织消化方法" in user_query_lower:
                            result["query_intent"] = "query_experiment_data"
                            logger.info(f"🔧 特殊处理：组织消化方法查询 -> 改为 query_experiment_data")
                        
                        return result
            except ImportError as e:
                logger.warning(f"⚠️ 无法导入意图识别模型: {e}")
            except Exception as e:
                logger.warning(f"⚠️ 使用意图识别模型失败: {e}")
            
            # 回退到使用Settings.llm
            if not Settings.llm:
                logger.warning("⚠️ LLM未初始化，回退到基于规则的意图识别")
                return {
                    "query_intent": self.detect_intent(user_query),
                    "milvus_filters": self.extract_entities(user_query),
                    "requested_fields": []
                }
            
            # 构建提示词
            prompt = self.llm_prompt_template.replace("{{schema_context}}", self.schema_context)
            prompt = prompt.replace("{{user_query}}", user_query)
            
            # 调用LLM (使用同步方法)
            response = Settings.llm.complete(prompt)
            response_text = str(response.text)
            
            logger.info(f"📝 LLM返回结果: {response_text[:200]}...")
            # 统计回退LLM的tokens消耗
            token_counter.count_llm_tokens(prompt, response_text, "fallback-llm")
            
            # 解析LLM返回的JSON
            import json
            # 清理返回的文本，移除可能的Markdown代码块标记
            clean_text = response_text.strip()
            if clean_text.startswith('```json'):
                clean_text = clean_text[7:]
            if clean_text.endswith('```'):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()
            result = json.loads(clean_text)
            
            # 验证结果结构
            if not isinstance(result, dict):
                raise ValueError("LLM返回结果格式错误")
            
            # 适配不同的返回结构
            # 处理常见的字段名变体
            if "intent" in result and "query_intent" not in result:
                result["query_intent"] = result.pop("intent")
            if "extracted_entities" in result and "milvus_filters" not in result:
                result["milvus_filters"] = result.pop("extracted_entities")
            if "query_fields" in result and "requested_fields" not in result:
                result["requested_fields"] = result.pop("query_fields")
            
            # 确保返回的字段完整
            if "query_intent" not in result:
                result["query_intent"] = "general_query"
            if "milvus_filters" not in result:
                result["milvus_filters"] = {}
            if "requested_fields" not in result:
                result["requested_fields"] = []
            
            # 验证字段名是否正确
            valid_fields = {
                "species",                  # 物种
                "sample_type_exp",          # 样本类型（项目经验表）
                "sample_type_prep",          # 样本类型（样本准备表）
                "tissue_type_prep",          # 组织类型
                "sample_category_prep",      # 样本大类
                "sample_detailed_type",     # 详细样本类型
                "experiment_protocol",      # 实验方案（项目经验表）
                "sample_prep_method",       # 样本处理方式（样本准备表）
                "platform_type",            # 实验平台
                "product_level3",            # 产品类型
                "source_table",             # 数据源表
                "department"                # 部门
            }
            
            # 过滤无效字段并修正字段名映射
            filtered_filters = {}
            
            # 组织类型关键词列表
            tissue_keywords = ["心脏", "肝脏", "肺", "脑", "脾脏", "肾", "皮肤", "肠道", "肌肉", "全血", "骨髓", "外周血", "PBMC"]
            
            # 样本类型关键词列表
            sample_type_keywords = ["冻存组织", "新鲜实体组织", "液体类样本", "细胞类样本", "类器官类样本", "石蜡包埋组织"]
            
            # 首先收集所有组织类型值
            tissue_values = []
            # 组织类型字段列表
            tissue_fields = ["sample_type_exp", "sample_type_prep", "tissue_type_prep", "sample_detailed_type"]
            
            for field, value in result["milvus_filters"].items():
                if value and isinstance(value, str):
                    clean_value = value.strip()
                    # 检查是否是组织类型
                    if any(keyword in clean_value for keyword in tissue_keywords):
                        tissue_values.append(clean_value)
                        logger.info(f"🔍 识别到组织类型值: '{clean_value}' 来自字段 '{field}'")
            
            # 然后处理所有字段
            for field, value in result["milvus_filters"].items():
                if field in valid_fields:
                    if value and isinstance(value, str):
                        clean_value = value.strip()
                        
                        # 情况1：如果是样本类型字段但值是组织类型，确保存储到正确的字段
                        if field in ["sample_type_exp", "sample_type_prep"] and any(keyword in clean_value for keyword in tissue_keywords):
                            logger.info(f"🔍 修正字段映射: 跳过组织类型 '{clean_value}' 存储到 '{field}'，将在后续处理中存储到正确字段")
                            continue
                        
                        # 情况2：如果是样本类型，确保存储到正确的字段
                        elif any(keyword in clean_value for keyword in sample_type_keywords):
                            logger.info(f"🔍 修正字段映射: 将样本类型 '{clean_value}' 存储到 'sample_type_exp' 和 'sample_type_prep'")
                            filtered_filters["sample_type_exp"] = clean_value
                            filtered_filters["sample_type_prep"] = clean_value
                            continue
                        
                        # 情况3：其他情况，保持原样
                        else:
                            filtered_filters[field] = clean_value
                    elif value is not None:
                        # 非字符串值且不为None，保持原样
                        filtered_filters[field] = value
                else:
                    logger.warning(f"⚠️ 过滤掉无效字段: {field}")
            
            # 处理收集到的组织类型值
            for tissue_value in tissue_values:
                # 根据意图类型决定存储到哪个字段
                if result.get("query_intent") == "query_experiment_data":
                    logger.info(f"🔍 修正字段映射: 将组织类型 '{tissue_value}' 存储到 'sample_detailed_type' (实验数据意图)")
                    filtered_filters["sample_detailed_type"] = tissue_value
                elif result.get("query_intent") == "query_preparation_guidelines":
                    logger.info(f"🔍 修正字段映射: 将组织类型 '{tissue_value}' 存储到 'tissue_type_prep' (样本准备意图)")
                    filtered_filters["tissue_type_prep"] = tissue_value
                else:
                    # 默认为两种字段都存储
                    logger.info(f"🔍 修正字段映射: 将组织类型 '{tissue_value}' 存储到 'sample_detailed_type' 和 'tissue_type_prep'")
                    filtered_filters["sample_detailed_type"] = tissue_value
                    filtered_filters["tissue_type_prep"] = tissue_value
            
            result["milvus_filters"] = filtered_filters
            
            logger.info(f"✅ LLM-based意图识别成功: {result['query_intent']}")
            logger.info(f"✅ LLM-based实体抽取成功: {result['milvus_filters']}")
            
            # 特殊情况：组织消化方法属于项目经验查询
            user_query_lower = user_query.lower()
            if "组织消化方法" in user_query_lower:
                result["query_intent"] = "query_experiment_data"
                logger.info(f"🔧 特殊处理：组织消化方法查询 -> 改为 query_experiment_data")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ LLM-based意图识别失败: {e}")
            # 回退到基于规则的方法
            intent = self.detect_intent(user_query)
            return {
                "query_intent": intent,
                "milvus_filters": self.extract_entities(user_query, intent),
                "requested_fields": []
            }
    
    def detect_intent(self, user_query: str) -> str:
        """
        基于规则的意图识别
        
        Args:
            user_query: 用户的自然语言查询
            
        Returns:
            意图类型字符串
        """
        logger.info(f"🔧 使用基于规则的意图识别处理查询: {user_query}")
        
        # 转换为小写，便于关键词匹配
        query_lower = user_query.lower()
        
        # 特殊情况：组织消化方法属于项目经验查询
        if "组织消化方法" in query_lower:
            return "query_experiment_data"
        
        # 实验数据查询关键词
        experiment_keywords = ["做过", "做了", "数据", "指标", "结果", "实验", "单细胞", "记录", "经验"]
        
        # 样本制备指南查询关键词
        preparation_keywords = ["需要", "注意事项", "送样量", "保存", "风险", "指南", "SOP", "制备"]
        
        # 检查是否包含实验数据查询关键词
        if any(keyword in query_lower for keyword in experiment_keywords):
            return "query_experiment_data"
        
        # 检查是否包含样本制备指南查询关键词
        if any(keyword in query_lower for keyword in preparation_keywords):
            return "query_preparation_guidelines"
        
        # 默认返回实验数据查询
        return "query_experiment_data"
    
    def extract_entities(self, user_query: str, intent: str = None) -> Dict[str, Any]:
        """
        从用户查询中提取实体和过滤条件
        
        Args:
            user_query: 用户的自然语言查询
            intent: 意图类型，可选
            
        Returns:
            包含实体和过滤条件的字典
        """
        entities = {}
        
        # 提取物种
        species = None
        if "小鼠" in user_query:
            species = "小鼠"
        elif "人" in user_query:
            species = "人"
        elif "大鼠" in user_query:
            species = "大鼠"
        elif "猪" in user_query:
            species = "猪"
        elif "牛" in user_query:
            species = "牛"
        elif "绵羊" in user_query or "羊" in user_query:
            species = "绵羊"
        elif "蝙蝠" in user_query:
            species = "蝙蝠"
        elif "山羊" in user_query:
            species = "山羊"
        elif "猕猴" in user_query:
            species = "猕猴"
        if species:
            entities["species"] = species
        
        # 提取组织类型
        tissue_type = None
        if "心脏" in user_query and "心肌" not in user_query:
            tissue_type = "心脏"
        elif "心肌" in user_query:
            tissue_type = "心肌"
        elif "肿瘤" in user_query:
            tissue_type = "肿瘤"
        elif "脑" in user_query or "脑组织" in user_query:
            tissue_type = "脑"
        elif "肺" in user_query or "肺组织" in user_query:
            tissue_type = "肺"
        elif "肝" in user_query or "肝组织" in user_query:
            tissue_type = "肝脏"
        elif "脾" in user_query or "脾脏" in user_query:
            tissue_type = "脾脏"
        elif "肾" in user_query or "肾脏" in user_query:
            tissue_type = "肾"
        elif "皮肤" in user_query:
            tissue_type = "皮肤"
        elif "肠道" in user_query:
            tissue_type = "肠道"
        elif "肌肉" in user_query or "背最长肌" in user_query:
            tissue_type = "肌肉"
        elif "全血" in user_query:
            tissue_type = "全血"
        elif "骨髓血" in user_query or "骨髓" in user_query:
            tissue_type = "骨髓"
        elif "外周血" in user_query or "PBMC" in user_query:
            tissue_type = "外周血/PBMC"
        if tissue_type:
            # 根据意图类型决定存储到哪个字段
            if intent == "query_experiment_data":
                logger.info(f"🔍 修正字段映射: 将组织类型 '{tissue_type}' 存储到 'sample_detailed_type' (实验数据意图)")
                entities["sample_detailed_type"] = tissue_type
            elif intent == "query_preparation_guidelines":
                logger.info(f"🔍 修正字段映射: 将组织类型 '{tissue_type}' 存储到 'tissue_type_prep' (样本准备意图)")
                entities["tissue_type_prep"] = tissue_type
            else:
                # 默认为两种字段都存储
                logger.info(f"🔍 修正字段映射: 将组织类型 '{tissue_type}' 存储到 'sample_detailed_type' 和 'tissue_type_prep'")
                entities["tissue_type_prep"] = tissue_type
                entities["sample_detailed_type"] = tissue_type
            # 不再添加到样本类型字段，因为 sample_type_exp 和 sample_type_prep 应该存储样本类型，不是组织类型
        
        # 提取样本大类
        sample_category = None
        if any(keyword in user_query for keyword in ["实体组织", "组织样本"]):
            sample_category = "实体组织类样本"
        elif any(keyword in user_query for keyword in ["液体", "血液", "腹水", "胸水"]):
            sample_category = "液体类样本"
        elif any(keyword in user_query for keyword in ["细胞", "PBMC"]):
            sample_category = "细胞类样本"
        elif "类器官" in user_query:
            sample_category = "类器官类样本"
        elif "植物" in user_query:
            sample_category = "植物类样本"
        if sample_category:
            entities["sample_category_prep"] = sample_category
        
        # 提取详细样本类型
        sample_detailed_type = None
        if "新鲜实体组织" in user_query:
            sample_detailed_type = "新鲜实体组织"
        elif "液体类样本" in user_query:
            sample_detailed_type = "液体类样本"
        elif "细胞类样本" in user_query:
            sample_detailed_type = "细胞类样本"
        elif "类器官" in user_query:
            sample_detailed_type = "类器官类样本"
        elif "冻存组织" in user_query:
            sample_detailed_type = "冻存组织"
        elif "石蜡" in user_query:
            sample_detailed_type = "石蜡包埋组织"
        if sample_detailed_type:
            entities["sample_detailed_type"] = sample_detailed_type
        
        # 提取实验方案
        experiment_protocol = None
        if "解离" in user_query:
            experiment_protocol = "解离"
        elif "抽核" in user_query:
            experiment_protocol = "抽核"
        if experiment_protocol:
            entities["experiment_protocol"] = experiment_protocol
            entities["sample_prep_method"] = experiment_protocol
        
        # 提取实验平台
        platform_type = None
        if "10X" in user_query:
            platform_type = "10X单细胞"
        elif "华大" in user_query:
            platform_type = "华大单细胞"
        elif "墨卓" in user_query:
            platform_type = "墨卓单细胞"
        if platform_type:
            entities["platform_type"] = platform_type
        
        # 提取产品类型
        product_type = None
        if "单细胞转录组" in user_query:
            product_type = "单细胞转录组"
        elif "单细胞免疫组" in user_query:
            product_type = "单细胞免疫组"
        elif "单细胞ATAC" in user_query or "单细胞 ATAC" in user_query:
            product_type = "单细胞 ATAC"
        elif "单细胞Flex" in user_query or "单细胞 Flex" in user_query:
            product_type = "单细胞Flex"
        if product_type:
            entities["product_level3"] = product_type
        
        return entities
    
    def parse_query(self, user_text: str, auth: AuthContext = None, use_llm: bool = True) -> Tuple[str, Dict[str, Any], str, Dict[str, Any]]:
        """
        统一的查询解析入口
        
        Args:
            user_text: 用户的自然语言查询
            auth: 认证上下文
            use_llm: 是否使用基于LLM的意图识别
            
        Returns:
            (搜索词, 过滤条件字典, 意图类型, 解析后的查询结果)
        """
        logger.info(f"📥 接收查询请求: {user_text}")
        
        try:
            # 1. 意图识别和实体抽取
            if use_llm:
                # 使用基于LLM的方法
                llm_result = self._llm_based_intent_recognition(user_text)
                intent = llm_result.get("query_intent", "general_query")
                entities = llm_result.get("milvus_filters", {})
                requested_fields = llm_result.get("requested_fields", [])
            else:
                # 使用基于规则的方法
                intent = self.detect_intent(user_text)
                entities = self.extract_entities(user_text, intent)
                requested_fields = []
            
            logger.info(f"🎯 检测到意图: {intent}")
            logger.info(f"🔍 提取到实体: {entities}")
            
            # 2. 根据意图设置数据源表
            if intent == "query_experiment_data":
                entities["source_table"] = "experiment_data"
            elif intent == "query_preparation_guidelines":
                entities["source_table"] = "preparation_guidelines"
            
            # 3. 添加部门过滤条件
            if auth:
                entities["department"] = auth.department
            
            # 4. 构建搜索词
            search_term = self._extract_keywords(user_text)
            logger.info(f"🔑 提取关键词: {search_term}")
            
            # 5. 构建解析结果
            parsed_result = {
                "query_intent": intent,
                "milvus_filters": entities,
                "requested_fields": requested_fields
            }
            
            # 6. 构建过滤条件字典，兼容现有系统
            filters = {}
            for field, value in entities.items():
                if value is not None:
                    filters[field] = value
            
            logger.info(f"✅ 查询解析完成: 搜索词='{search_term}', 过滤条件={filters}, 意图={intent}")
            
            return search_term, filters, intent, parsed_result
            
        except Exception as e:
            logger.error(f"❌ 查询解析异常: {e}")
            # 降级：返回原文本，不做过滤
            return user_text, {}, "general_query", {"query_intent": "general_query", "milvus_filters": {}, "requested_fields": []}
    
    def _extract_keywords(self, query_text: str) -> str:
        """
        从查询文本中提取关键词
        
        Args:
            query_text: 用户的自然语言查询
            
        Returns:
            提取的关键词
        """
        # 简单实现：去除停用词，提取核心词
        stop_words = ["的", "了", "和", "是", "在", "有", "我", "你", "他", "她", "它", "们", "请", "帮", "我", "查询", "找", "一下"]
        words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', query_text)
        keywords = [word for word in words if word not in stop_words]
        return " ".join(keywords)
    
    def _expand_tissue_variants(self, field: str, value: str) -> list:
        """
        为组织类型字段生成变体
        
        Args:
            field: 字段名
            value: 字段值
            
        Returns:
            包含变体的列表
        """
        # 定义需要处理变体的组织类型字段
        tissue_fields = [
            "sample_detailed_type",  # 样本详细类型（项目经验查询）
            "tissue_type_prep"         # 组织类型（样本准备查询）
        ]
        
        # 只处理组织类型字段
        if field not in tissue_fields:
            return [value]
        
        variants = [value]
        
        # 生成变体
        if isinstance(value, str):
            # 如果值包含"组织"，添加不包含"组织"的版本
            if "组织" in value:
                variants.append(value.replace("组织", ""))
            # 如果值不包含"组织"，添加包含"组织"的版本
            elif "组织" not in value:
                variants.append(value + "组织")
        
        # 去重并返回
        return list(set(variants))
    
    def build_milvus_expr(self, filters: Dict[str, Any]) -> str:
        """
        根据解析出的过滤器构建Milvus的查询表达式
        
        Args:
            filters: 包含过滤条件的字典
            
        Returns:
            Milvus查询表达式字符串
        """

        conditions = []
        for field, value in filters.items():
            if value is None:
                continue
            
            # 处理列表类型（多个值）
            if isinstance(value, list):
                # 过滤空值
                non_empty_values = [v for v in value if v is not None and str(v).strip() and str(v).strip().lower() not in ["null", "none", "空"]]
                if non_empty_values:
                    # 为组织类型字段构建变体
                    expanded_values = []
                    for v in non_empty_values:
                        if isinstance(v, str):
                            expanded_values.extend(self._expand_tissue_variants(field, v))
                    
                    # 去重
                    unique_values = list(set(expanded_values))
                    if unique_values:
                        # 构建 in 条件
                        values_str = ", ".join([f"'{v}'" for v in unique_values])
                        conditions.append(f"{field} in [{values_str}]")
            
            # 处理字符串类型
            elif isinstance(value, str):
                # 跳过表示空值的字符串
                if value.lower() in ["null", "none", "空", ""]:
                    continue
                
                # 为组织类型字段构建变体
                tissue_variants = self._expand_tissue_variants(field, value)
                if len(tissue_variants) > 1:
                    # 构建 in 条件
                    values_str = ", ".join([f"'{v}'" for v in tissue_variants])
                    conditions.append(f"{field} in [{values_str}]")
                else:
                    # 构建等于条件
                    conditions.append(f"{field} == '{value}'")
            
            # 处理布尔值
            elif isinstance(value, bool):
                conditions.append(f"{field} == {str(value).lower()}")
            
            # 处理数值类型
            elif isinstance(value, (int, float)):
                conditions.append(f"{field} == {value}")
        
        if not conditions:
            return ""
        return " and ".join(conditions)
    
    def get_intent_description(self, intent_key: str) -> str:
        """
        获取意图类型的描述
        
        Args:
            intent_key: 意图键名
            
        Returns:
            意图描述
        """
        return self.intent_types.get(intent_key, "未知意图")


# 全局意图检测器实例
intent_detector = IntentDetector()
