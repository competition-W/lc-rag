#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强型意图识别器
用于识别用户查询的意图类型、提取Milvus过滤条件和用户请求字段
"""

import json
from typing import Dict, Any, Optional, List
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from utils.logger import logger


class IntentRecognizer:
    """
    增强型意图识别器
    负责识别用户查询的意图类型、提取Milvus过滤条件和用户请求字段
    """
    
    def __init__(self):
        # 可识别的意图类型
        self.intent_types = {
            "query_experiment_data": "查询实验记录相关信息",
            "query_preparation_guidelines": "查询样本制备指南相关信息",
            "unknown": "无法明确区分上述两种意图的通用查询"
        }
    
    async def recognize_intent_and_filters(self, user_query: str) -> Dict[str, Any]:
        """
        增强型意图识别与实体抽取
        使用LLM解析用户查询，提取意图、Milvus过滤器和请求字段
        
        Args:
            user_query: 用户的自然语言查询
            
        Returns:
            包含意图、过滤条件和请求字段的字典
        """
        try:
            # 构建增强型意图识别Prompt
            prompt = self._build_intent_recognition_prompt(user_query)
            
            # 调用LLM进行意图识别
            from llama_index.core.llms import ChatMessage, MessageRole
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content="你是一个能够精确分析用户查询的助手。请严格按照指定格式输出结果。"),
                ChatMessage(role=MessageRole.USER, content=prompt)
            ]
            logger.info(f"📝 系统提示词: {messages[0].content}")
            logger.info(f"📝 用户提示词: {messages[1].content[:200]}...")  # 只显示前200个字符
            logger.info(f"🔧 LLM方法调用: Settings.llm.chat")
            
            # DashScopeLLM有chat方法，使用chat同步方法，不需要await
            response = Settings.llm.chat(messages=messages, temperature=0.0)
            
            # 详细日志记录
            logger.info(f"📋 LLM响应对象: {response}")
            logger.info(f"📋 LLM响应类型: {type(response)}")
            
            # 解析LLM响应 - ChatResponse对象有message属性
            llm_response_str = response.message.content.strip() if hasattr(response, 'message') and hasattr(response.message, 'content') else ''
            logger.info(f"🧠 LLM意图识别原始响应: '{llm_response_str}'")
            logger.info(f"📏 响应长度: {len(llm_response_str)}")
            
            # 检查响应是否为空
            if not llm_response_str:
                logger.error(f"❌ LLM返回空响应，使用基于规则的意图识别作为备选")
                return self._rule_based_intent_recognition(user_query)
            
            # 提取JSON部分
            json_start = llm_response_str.find("{")
            json_end = llm_response_str.rfind("}") + 1
            if json_start != -1 and json_end != 0:
                json_str = llm_response_str[json_start:json_end]
                try:
                    intent_result = json.loads(json_str)
                    
                    # 兼容处理数值范围查询
                    self._process_numeric_filters(intent_result)
                    
                    logger.info(f"✅ 意图识别结果: {json.dumps(intent_result, ensure_ascii=False)}")
                    return intent_result
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON解析失败: {e}, 原始JSON: {json_str}, 使用基于规则的意图识别作为备选")
                    return self._rule_based_intent_recognition(user_query)
            else:
                logger.error(f"❌ 无法从LLM响应中提取JSON: {llm_response_str}, 使用基于规则的意图识别作为备选")
                return self._rule_based_intent_recognition(user_query)
                
        except Exception as e:
            logger.error(f"❌ 意图识别失败: {e}, 使用基于规则的意图识别作为备选")
            import traceback
            logger.error(f"📋 错误堆栈: {traceback.format_exc()}")
            return self._rule_based_intent_recognition(user_query)
    
    def _rule_based_intent_recognition(self, user_query: str) -> Dict[str, Any]:
        """
        基于规则的意图识别，当LLM无法正常工作时作为备选
        
        Args:
            user_query: 用户的自然语言查询
            
        Returns:
            包含意图、过滤条件和请求字段的字典
        """
        logger.info(f"🔧 使用基于规则的意图识别处理查询: {user_query}")
        
        # 构建默认结果
        intent_result = self._get_default_intent_result()
        
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
        intent_result["milvus_filters"]["species"] = species
        
        # 提取样本类型
        sample_type = None
        if "心脏" in user_query:
            sample_type = "心脏组织"
        elif "心肌" in user_query:
            sample_type = "心肌组织"
        elif "肿瘤" in user_query:
            sample_type = "肿瘤"
        elif "脑组织" in user_query:
            sample_type = "脑组织"
        elif "肺" in user_query:
            sample_type = "肺组织"
        elif "肝" in user_query:
            sample_type = "肝组织"
        intent_result["milvus_filters"]["sample_detailed_type"] = sample_type
        
        # 识别意图
        query_lower = user_query.lower()
        if any(keyword in query_lower for keyword in ["做过", "做了", "数据", "指标", "结果", "实验", "单细胞"]):
            intent_result["query_intent"] = "query_experiment_data"
            intent_result["milvus_filters"]["source_table"] = "experiment_data"
        elif any(keyword in query_lower for keyword in ["需要", "注意事项", "送样量", "保存", "风险", "指南", "SOP"]):
            intent_result["query_intent"] = "query_preparation_guidelines"
            intent_result["milvus_filters"]["source_table"] = "preparation_guidelines"
        
        logger.info(f"✅ 基于规则的意图识别结果: {json.dumps(intent_result, ensure_ascii=False)}")
        return intent_result
    
    # def _build_intent_recognition_prompt(self, user_query: str) -> str:
    #     """
    #     构建意图识别Prompt
    #     """
    #     return f"""
    #     请从以下用户问题中提取查询意图、用于Milvus精确过滤的条件，以及用户最终想要获取的具体信息字段。

    #     请严格按照以下JSON格式输出。如果某个字段在查询中未明确提及，请将其值设为 null 或空列表（根据字段类型）。

    #     可识别的意图：
    #     - "query_experiment_data": 查询实验记录相关信息。
    #     - "query_preparation_guidelines": 查询样本制备指南相关信息。
    #     - "general_query": 如果无法明确区分上述两种意图。

    #     可用于过滤的实验数据字段 (来自 'experiment_data' 源):
    #     - `source_table`: "experiment_data" (如果明确指代实验数据)
    #     - `project_library_type` (e.g., "10X单细胞3‘转录本-抽核(V3试剂)")
    #     - `species` (e.g., "小鼠", "人")
    #     - `sample_type` (e.g., "冻存组织", "新鲜组织")
    #     - `sample_detailed_type` (e.g., "心肌组织", "肿瘤")
    #     - `prep_method` (e.g., "解离", "抽核")
    #     - `arrival_temp_celsius` (数值, e.g., "-80")
    #     - `total_cells_10k` (数值, 细胞总量(万))
    #     - `clumping_rate_percent` (数值, 结团率(%))
    #     - `cell_viability_percent` (数值, 细胞活率(%))
    #     - `nucleated_rate_percent` (数值, 有核率(%))
    #     - `captured_cells` (数值, 捕获细胞数)
    #     - `reads_per_cell` (数值, reads/cell)
    #     - `median_genes` (数值, 基因中位数)
    #     - `target_cell_types`: (字符串列表, e.g., ["巨噬细胞", "内皮细胞"], 如果用户询问“有没有鉴定到巨噬细胞？”则为 ["巨噬细胞"])
    #     - `tissue_digestion_protocol` (e.g., "温和机械解离")

    #     可用于过滤的样本制备指南字段 (来自 'preparation_guidelines' 源):
    #     - `source_table`: "preparation_guidelines" (如果明确指代制备指南)
    #     - `product_level1` (e.g., "单细胞空间组学")
    #     - `product_level2` (e.g., "单细胞测序")
    #     - `product_level3` (e.g., "单细胞转录组")
    #     - `sample_category` (e.g., "实体组织类样本")
    #     - `sample_type` (e.g., "新鲜实体组织")
    #     - `tissue_type` (e.g., "肿瘤", "脑组织")
    #     - `prep_method` (e.g., "解离", "抽核")
    #     - `preparation_method_doc` (SOP文档名)
    #     - `handling_notes` (注意事项)
    #     - `risk_level` (e.g., "风险", "合格")
    #     - `recommended_amount_mg` (数值, 建议送样量(mg))
    #     - `qualitative_description_text` (e.g., "大米大小", "绿豆大小")
    #     - `risk_query`: (布尔值, 如果用户询问“有没有风险？”则设置为 true)

    #     用户明确想获取的信息字段 (请使用预处理后的字段名):
    #     - `requested_fields`: 字符串列表 (e.g., ["cell_viability_percent", "annotation_results_full", "notes_full"])

    #     用户问题: "{user_query}"

    #     输出JSON格式:
    #     ```json
    #     {{
    #     "query_intent": "query_experiment_data",
    #     "milvus_filters": {{
    #         "source_table": null,
    #         "species": null,
    #         "sample_detailed_type": null,
    #         "prep_method": null,
    #         "cell_viability_percent": null,
    #         "target_cell_types": [],
    #         "risk_query": false,
    #         "risk_level": null,
    #         "recommended_amount_mg": null
    #     }},
    #     "requested_fields": []
    #     }}
    #     ```
    #     请注意：对于数值字段，如果用户查询是范围（例如“活率大于90%”），请将其解析为 {{"op": "gt", "value": 90}} 这样的结构。
    #     """
    def _build_intent_recognition_prompt(self, user_query: str) -> str:
        """
        构建意图识别Prompt
        """
        return f"""
        请根据用户问题，严格按照以下要求提取查询意图、过滤条件和请求字段，并输出JSON格式结果：

        ## 1. 查询意图 (query_intent):
        必须从以下选项中选择一个：
        - "query_experiment_data": 用户查询单细胞项目经验相关数据、指标、注释结果、方案细节等
        - "query_preparation_guidelines": 用户查询样本制备指南、送样量、风险、注意事项等
        - "general_query": 无法明确归类到上述两种意图

        ## 2. 过滤字段 (milvus_filters):
        - source_table: 根据选择的意图自动填充为"experiment_data"或"preparation_guidelines"
        - 从用户问题中提取：物种、样本类型、组织类型、实验方案等
        - 数值范围用{{"op": "gt/lt/eq/gte/lte", "value": 数值}}格式

        ## 3. 请求字段 (requested_fields):
        提取用户明确或隐含需要的信息字段，如数据指标、送样量等

        用户问题: "{user_query}"

        输出示例：
        ```json
        {{
        "query_intent": "query_experiment_data",
        "milvus_filters": {{
            "source_table": "experiment_data",
            "species": "小鼠",
            "sample_detailed_type": "心脏组织",
            "prep_method": null,
            "target_cell_types": [],
            "risk_query": false,
            "risk_level": null
        }},
        "requested_fields": ["cell_viability_percent", "annotation_results_full"]
        }}
        ```
        请严格按照上述格式输出，未提及字段设为null或空列表，不要添加任何额外解释。
        """
    
    def _process_numeric_filters(self, intent_result: Dict[str, Any]) -> None:
        """
        处理数值范围过滤器
        将LLM返回的范围表达式转换为标准格式
        """
        import re
        
        filters = intent_result.get("milvus_filters", {})
        for field, value in filters.items():
            if isinstance(value, str):
                # 处理如 ">90" 这样的范围表达式
                match = re.match(r'([<>]=?|!=|==?)\s*(\d+\.?\d*)', value)
                if match:
                    op = match.group(1)
                    val = float(match.group(2))
                    
                    # 转换为标准操作符
                    op_map = {
                        ">": "gt", "<": "lt", "=": "eq", "==": "eq", "!=": "ne",
                        ">=": "gte", "<=": "lte"
                    }
                    filters[field] = {
                        "op": op_map.get(op, "eq"),
                        "value": val
                    }
                    logger.info(f"🔧 转换数值范围过滤器: {field} {op} {val} -> {filters[field]}")
    
    def _get_default_intent_result(self) -> Dict[str, Any]:
        """
        获取默认意图识别结果
        """
        return {
            "query_intent": "unknown",
            "milvus_filters": {
                "source_table": None,
                "species": None,
                "sample_detailed_type": None,
                "prep_method": None,
                "target_cell_types": [],
                "risk_query": False,
                "risk_level": None,
                "recommended_amount_mg": None
            },
            "requested_fields": []
        }
    
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
            if value is None or (isinstance(value, (list, dict)) and not value):
                continue
            
            # 处理数值范围查询
            if isinstance(value, dict) and 'op' in value and 'value' in value:
                op_map = {
                    "gt": ">", "lt": "<", "eq": "==", "ne": "!=",
                    "gte": ">=", "lte": "<="
                }
                op = op_map.get(value['op'], "==")
                conditions.append(f"{field} {op} {value['value']}")
            
            # 处理目标细胞类型 (JSON array_contains)
            elif field == "target_cell_types" and isinstance(value, list) and value:
                for cell_type in value:
                    conditions.append(f"annotation_cell_types_list array_contains '{cell_type}'")
            
            # 处理风险查询
            elif field == "risk_query" and value is True:
                conditions.append("risk_level == '风险'")
            
            # 处理一般字符串/数值精确匹配
            elif isinstance(value, str):
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
    
    def get_intent_type(self, intent_key: str) -> str:
        """
        获取意图类型的描述
        
        Args:
            intent_key: 意图键名
            
        Returns:
            意图描述
        """
        return self.intent_types.get(intent_key, "未知意图")
    
    async def recognize_intent(self, user_query: str) -> str:
        """
        识别用户查询的意图
        
        Args:
            user_query: 用户的自然语言查询
            
        Returns:
            新的意图类型字符串
        """
        try:
            # 直接返回新意图，不再映射为旧意图
            intent_result = await self.recognize_intent_and_filters(user_query)
            query_intent = intent_result.get("query_intent", "unknown")
            
            # 对unknown意图做单独处理，记录日志
            if query_intent == "unknown":
                logger.warning(f"⚠️ 意图识别失败，返回未知意图: {user_query}")
            else:
                logger.info(f"🔍 意图识别结果: {query_intent}")
                
            return query_intent
        except Exception as e:
            logger.error(f"❌ 意图识别失败: {e}")
            return "unknown"


# 全局意图识别器实例
intent_recognizer = IntentRecognizer()
