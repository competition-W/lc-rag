import asyncio
import json
import os
import re
from datetime import datetime
from typing import List, Dict, Optional, Any

from llama_index.core import Settings
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.core.schema import NodeWithScore
from llama_index.core.llms import ChatMessage, MessageRole

from .milvus_manager import milvus_manager
from .prompt_manager import prompt_manager
from .intent_detector import intent_detector
from utils.auth import AuthContext
from utils.logger import logger
from utils.context_formatter import ContextFormatter
from utils.data_exporter import DataExporter

# Rerank 依赖 (可选)
try:
    from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
except ImportError:
    DashScopeRerank = None

# Embedding 依赖
try:
    from llm.get_embedding import get_embed_model
except ImportError:
    # 兜底：如果没有这个文件，就用 Settings 里的
    get_embed_model = lambda **kwargs: Settings.embed_model

# 用于token计数
try:
    import tiktoken
except ImportError:
    tiktoken = None



_EMBED_MODEL_INIT = False

def _ensure_embed_model():
    """确保 Embedding 模型初始化"""
    global _EMBED_MODEL_INIT
    if not _EMBED_MODEL_INIT:
        if not Settings.embed_model:
            Settings.embed_model = get_embed_model() 
        _EMBED_MODEL_INIT = True

# ==========================================
# 1. 统一检索入口 - 三阶段RAG流程
# ==========================================
async def unified_query_service(
    auth: AuthContext,
    query_text: str,
    column_filters: Optional[Dict[str, str]] = None,
    llm_top_k: int = 8,
    semantic_top_k: int = 15,
    intent: str = "unknown",
    websocket = None,
    parsed_query: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    智能检索主入口 - 实现三阶段RAG流程
    """
    # 重置token统计，确保每次请求的token统计都是独立的
    from utils.token_counter import token_counter
    token_counter.reset_stats()
    
    _ensure_embed_model()
    
    # 1. 获取索引
    index = milvus_manager.get_existing_index(auth.department)
    if not index:
        return {
            "mode": "empty", 
            "answer": "抱歉，当前部门暂无数据索引。", 
            "sources": [], 
            "all_rows": []
        }

    # 🔍 调试日志：看看 Service 到底收到了什么
    logger.info(f"🛡️ [Service] 收到请求 -> Query: '{query_text}' | Filters: {column_filters} | Intent: {intent}")

    # 2. 阶段一：增强型意图识别与实体抽取 (LLM-based)
    logger.info("🚀 进入阶段一：增强型意图识别与实体抽取")
    if parsed_query is None:
        # 使用新的intent_detector获取解析结果
        _, _, _, parsed_query = intent_detector.parse_query(query_text)
        logger.info(f"📋 意图识别结果: {json.dumps(parsed_query, ensure_ascii=False)}")
    else:
        logger.info(f"📋 使用已解析的意图识别结果: {json.dumps(parsed_query, ensure_ascii=False)}")

    # ✅ 新增:检查是否需要澄清
    if parsed_query.get("clarification_needed", False):
        clarifying_questions = parsed_query.get("clarifying_questions", [])
        clarification_message = "为了给您提供更准确的信息,请补充以下信息:\n" + "\n".join(
            [f"- {q}" for q in clarifying_questions]
        )
        
        logger.info(f"💬 需要用户澄清: {clarification_message}")
        
        # 获取token统计信息(此时尚未调用LLM,token消耗极低)
        from utils.token_counter import token_counter
        token_stats = token_counter.get_total_stats()
        
        return {
            "mode": "clarification",
            "answer": clarification_message,
            "sources": [],
            "all_rows": [],
            "downloadable_data": [],
            "token_stats": token_stats
        }

    # 3. 阶段二：Milvus 精确检索
    logger.info("🚀 进入阶段二：Milvus 精确检索")
    search_result = await _perform_milvus_retrieval(
        index, query_text, auth, parsed_query, column_filters, llm_top_k
    )

    # 4. 阶段三：上下文构建与LLM响应生成
    logger.info("🚀 进入阶段三：上下文构建与LLM响应生成")
    all_rows = search_result.get("all_rows", [])
    best_rows = search_result.get("best_rows", [])
    
    # 获取检索到的总条数（all_rows包含所有符合条件的结果）
    total_results = len(all_rows) if all_rows else len(best_rows)
    
    logger.info(f"📊 检索结果总数: {total_results} 条")
    logger.info(f"🔝 best_rows数量: {len(best_rows)}")
    logger.info(f"🔍 查询文本: {query_text}")
    logger.info(f"🎯 意图: {intent}")
    
    # 直接生成回答，不考虑30条数据的限制
    # 调用_generate_summary函数，根据查询意图提取相关指标数据
    # 使用all_rows（所有检索结果）进行指标聚合，使用best_rows进行上下文生成
    logger.info("🚀 准备调用_generate_summary函数")
    llm_answer = await _generate_summary(
        query_text=query_text if query_text else str(column_filters), # 如果query为空，用filters做上下文
        context_nodes=best_rows,
        intent=intent,
        all_retrieved_rows=all_rows,  # 传递所有检索结果用于指标聚合
        parsed_query=parsed_query,  # 传递解析后的查询用于上下文构建
        websocket=websocket  # 传递WebSocket连接用于流式输出
    )
    
    logger.info(f"📝 生成的LLM回答: {llm_answer[:100]}...")
    logger.info(f"📝 生成的LLM回答长度: {len(llm_answer)} 字符")

    # 获取token统计信息
    from utils.token_counter import token_counter
    token_stats = token_counter.get_total_stats()
    
    # 获取意图
    intent = parsed_query.get("query_intent", "unknown") if parsed_query else intent

    # 准备前端下载数据（完整版，包含所有字段，中文表头）
    downloadable_data = DataExporter.export_for_frontend(
        docs=search_result.get("all_rows", []),
        intent=intent
    )

    result = {
        "mode": search_result["mode"],
        "answer": llm_answer,
        "sources": best_rows,
        "all_rows": search_result["all_rows"],
        "downloadable_data": downloadable_data,  # ✅ 新增：前端下载用完整数据（中文表头）
        "token_stats": token_stats
    }

    logger.info(f"✅ 返回数据包含 {len(downloadable_data)} 条可下载记录")
    logger.info(f"✅ unified_query_service返回结果: {result.keys()}")
    logger.info(f"✅ answer字段长度: {len(result['answer'])} 字符")
    
    return result

# ==========================================
# 2. 阶段二：Milvus 精确检索实现
# ==========================================
async def _perform_milvus_retrieval(
    index, query_text, auth, parsed_query, column_filters, llm_top_k
):
    """
    执行Milvus精确检索
    """
    # ✅ 简化:直接使用意图识别结果
    query_intent = parsed_query.get("query_intent", "unknown")
    milvus_filters = parsed_query.get("milvus_filters", {}).copy()  # 复制一份,避免修改原始数据
    requested_fields = parsed_query.get("requested_fields", [])
    
    # 添加部门过滤条件(这是唯一需要在这里添加的)
    milvus_filters["department"] = auth.department
    
    # ⚠️ 向后兼容:如果传入了column_filters,合并进去
    if column_filters and len(column_filters) > 0:
        logger.warning(f"⚠️ 检测到传统column_filters参数,建议使用parsed_query: {column_filters}")
        milvus_filters.update(column_filters)
    
    logger.info(f"📋 合并后的Milvus过滤条件: {json.dumps(milvus_filters, ensure_ascii=False)}")
    
    # 构建Milvus查询表达式
    milvus_expr = intent_detector.build_milvus_expr(milvus_filters)
    logger.info(f"🔍 构建的Milvus查询表达式: {milvus_expr}")
    
    # 使用结构化表格检索策略执行检索
    return await _strategy_structured_table(
        index, query_text, auth, milvus_filters, llm_top_k, milvus_expr
    )

# # ==========================================
# # 2. LLM 生成模块
# # ==========================================
# async def _generate_summary(query_text: str, context_nodes: List[Dict], intent: str = "sample_query", websocket=None) -> str:
#     if not context_nodes:
#         return "抱歉，未找到匹配的数据。"
#     
#     # 数据扁平化
#     context_str_list = []
#     for idx, item in enumerate(context_nodes):
#         meta = item.get("metadata", {}).copy()
#         # 清理无关字段
#         for ignore_key in ['chunk_id', 'owner', 'uploader', 'full_row_json', 'doc_type', 'vector_id']:
#             meta.pop(ignore_key, None)
#         context_str_list.append(f"[{idx+1}] {json.dumps(meta, ensure_ascii=False)}")
#     
#     context_text = "\n".join(context_str_list)

#     system_prompt = (
#         "你是一个生物数据助手。请基于【参考数据】回答问题。\n"
#         "原则：\n1. 实事求是，没有数据就说没有。\n2. 语言简洁专业。\n"
#     )
#     user_prompt = f"用户查询: {query_text}\n\n【参考数据】:\n{context_text}"

#     try:
#         if not Settings.llm:
#             return "LLM 未初始化"
#         response = await Settings.llm.achat(
#             messages=[
#                 ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
#                 ChatMessage(role=MessageRole.USER, content=user_prompt)
#             ]
#         )
#         return str(response.message.content)
#     except Exception as e:
#         logger.error(f"LLM 生成失败: {e}")
#         return "已检索到数据，请查看列表。"


def _format_node_for_llm(node_data: dict, query_text: str = "") -> str:
    """
    将单条元数据字典格式化为 LLM 易读的字符串。
    根据用户查询意图，只提取相关类型的指标数据，降低上下文输入。
    """
    if not node_data:
        return ""
    
    # 定义指标分类
    experimental_metrics = [
        "total_cells_10k", "clumping_rate_percent", "cell_viability_percent", 
        "nucleated_rate_percent", "captured_cells", 
        "tissue_weight", "tissue_weight_unit",
        "qualitative_description", "rin_score",
        "antibody_info", "streaming_protocol", "is_lysis", "is_dead_removal",
        "storage_method", "is_streaming", "experiment_protocol"
    ]
    
    data_metrics = [
        "reads_per_cell", "median_genes", "annotation_results"
    ]
    
    annotation_metrics = [
        "cell_annotation_result", "annotation_results", "注释结果", "细胞注释", "cell_type", "cell_types"
    ]
    
    sample_info = [
        "platform", "species", "category", "tissue", "filename", "sheet_name"
    ]
    
    reference_materials = [
        "digestion_protocol", "related_articles", "feishu_doc_link", "video_live_link"
    ]
    
    # 定义不需要传给 LLM 的内部技术字段
    exclude_keys = [
        "source_id", "doc_id", "embedding", "window", 
        "original_text", "_node_content", "relationships", "department",
        "uploader", "doc_type", "row_index", "chunk_id", "owner"
    ]
    
    # 根据查询意图确定需要提取的指标类型
    extract_all = False
    need_experimental = False
    need_data = False
    need_annotation = False
    
    if not query_text or query_text.strip() == "":
        extract_all = True
    else:
        lower_query = query_text.lower()
        # 实验指标查询
        if "实验指标" in lower_query or "实验数据" in lower_query:
            need_experimental = True
        # 数据指标查询
        elif "数据指标" in lower_query or "统计" in lower_query or "数值" in lower_query:
            need_data = True
        # 注释结果查询
        elif "注释结果" in lower_query or "细胞注释" in lower_query or "注释" in lower_query or "细胞类型" in lower_query:
            need_annotation = True
        # 综合查询
        elif "数据汇总" in lower_query or "汇总" in lower_query or "如何" in lower_query or "怎么样" in lower_query:
            extract_all = True
    
    # 如果没有指定提取类型，默认提取所有
    if not extract_all and not need_experimental and not need_data and not need_annotation:
        extract_all = True
    
    # 结构化数据
    structured_data = {
        "单细胞平台": {},
        "样本信息": {},
        "实验指标": {},
        "数据指标": {},
        "细胞注释结果": {},
        "参考资料": {}
    }
    
    for k, v in node_data.items():
        # 排除系统字段
        if k in exclude_keys:
            continue
        # 排除空值
        if v is None:
            continue
        # 排除空字符串
        if isinstance(v, str) and not v.strip():
            continue
        
        # 将值转换为字符串
        v_str = str(v).strip()
        if not v_str:
            continue
        
        # 清理字段名，移除换行符和特殊字符，使输出更美观
        clean_k = k.replace('\n', ' ')
        
        # 根据字段类型分类
        if k == "platform":
            structured_data["单细胞平台"][clean_k] = v_str
        elif k in sample_info:
            structured_data["样本信息"][clean_k] = v_str
        
        # 提取实验指标
        if (extract_all or need_experimental) and k in experimental_metrics:
            structured_data["实验指标"][clean_k] = v_str
        
        # 提取数据指标
        if (extract_all or need_data) and k in data_metrics:
            structured_data["数据指标"][clean_k] = v_str
        
        # 提取细胞注释结果
        if (extract_all or need_annotation) and (k in annotation_metrics or "annotation" in k.lower() or "注释" in k.lower() or "cell" in k.lower()):
            structured_data["细胞注释结果"][clean_k] = v_str
        
        # 提取参考资料
        if k in reference_materials:
            structured_data["参考资料"][clean_k] = v_str
    
    # 格式化输出
    output_lines = []
    
    for section, items in structured_data.items():
        if items:
            output_lines.append(f"**{section}**")
            for k, v in items.items():
                output_lines.append(f"{k}：{v}")
            output_lines.append("")
    
    return "\n".join(output_lines).strip()


# ==========================================
# 2. LLM 生成模块 (优化版)
# ==========================================
def _aggregate_numeric_metrics(context_nodes: List[Dict], query_text: str = "") -> str:
    """
    聚合处理数字指标，计算平均值、中位数等统计信息
    根据查询文本区分数据指标和实验指标
    """
    if not context_nodes:
        return ""
    
    # 定义指标类型分类
    experimental_metrics = [
        "tissue_weight", "total_cells_10k", "clumping_rate_percent", 
        "cell_viability_percent", "nucleated_rate_percent", "captured_cells", 
        "rin_score"
    ]
    
    data_metrics = [
        "reads_per_cell", "median_genes", "annotation_results"
    ]
    
    # 定义指标名称映射，将英文列名转换为友好的中文名称
    metric_name_mapping = {
        "tissue_weight": "组织重量",
        "reads_per_cell": "数据量",
        "median_genes": "基因中位数",
        "total_cells_10k": "细胞总量",
        "clumping_rate_percent": "结团率",
        "cell_viability_percent": "细胞活率",
        "nucleated_rate_percent": "有效核率",
        "captured_cells": "捕获细胞数",
        "rin_score": "核碎片率",
        "annotation_results": "注释结果-组织特异性指标"
    }
    
    # 分析查询类型
    lower_query = query_text.lower() if query_text else ""
    need_data_metrics = "数据指标" in lower_query
    need_experimental_metrics = "实验指标" in lower_query
    
    # 如果没有明确指定，默认聚合所有数字指标
    if not need_data_metrics and not need_experimental_metrics:
        need_data_metrics = True
        need_experimental_metrics = True
    
    # 收集指定类型的数字指标
    numeric_metrics = {}
    
    for node in context_nodes:
        # 正确处理TextNode对象和字典对象
        if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
            meta = node.node.metadata
        elif hasattr(node, 'metadata'):
            meta = node.metadata
        elif isinstance(node, dict):
            meta = node.get("metadata", {})
        else:
            meta = {}
        for key, value in meta.items():
            # 根据查询类型过滤指标
            is_data_metric = key in data_metrics
            is_experimental_metric = key in experimental_metrics
            
            # 检查是否需要当前指标类型
            should_process = False
            if need_data_metrics and is_data_metric:
                should_process = True
            elif need_experimental_metrics and is_experimental_metric:
                should_process = True
            elif not is_data_metric and not is_experimental_metric:  # 未分类的指标默认处理
                should_process = True
            
            if should_process:
                # 尝试转换为数字
                try:
                    # 清理字符串，去除空格和特殊字符
                    clean_value = str(value).strip()
                    if clean_value and clean_value != "/" and clean_value != "(空)":
                        # 处理带千分位的数字，如12,033
                        clean_value = clean_value.replace(",", "")
                        # 转换为浮点数
                        num_value = float(clean_value)
                        if key not in numeric_metrics:
                            numeric_metrics[key] = []
                        numeric_metrics[key].append(num_value)
                except (ValueError, TypeError):
                    # 不是数字，跳过
                    continue
    
    if not numeric_metrics:
        return ""
    
    # 生成聚合结果
    aggregate_result = "\n\n**数字指标统计汇总**\n"
    aggregate_result += "| 指标名称 | 样本数量 | 平均值 | 最小值 | 最大值 | 中位数 |\n"
    aggregate_result += "|----------|----------|--------|--------|--------|--------|\n"
    
    for metric_name, values in numeric_metrics.items():
        # 计算统计值
        sample_count = len(values)
        avg_value = sum(values) / sample_count
        min_value = min(values)
        max_value = max(values)
        # 计算中位数
        sorted_values = sorted(values)
        mid_index = sample_count // 2
        median_value = sorted_values[mid_index] if sample_count % 2 == 1 else (sorted_values[mid_index - 1] + sorted_values[mid_index]) / 2
        
        # 使用友好的中文名称，如果没有映射则使用原名称
        display_name = metric_name_mapping.get(metric_name, metric_name)
        
        # 添加到结果表格
        aggregate_result += f"| {display_name} | {sample_count} | {avg_value:.2f} | {min_value:.2f} | {max_value:.2f} | {median_value:.2f} |\n"
    
    return aggregate_result

def get_token_count(text, model_name="gpt-4"):
    """使用tiktoken估算文本的token数量"""
    if not tiktoken:
        # 兜底：如果没有tiktoken，使用粗略估算（1个token≈0.75个汉字或1.5个英文单词）
        return len(text) // 2
    
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"❌ Token计数失败: {e}，使用粗略估算")
        return len(text) // 2


def format_single_document(doc, parsed_query):
    """
    根据文档内容和用户请求，格式化单条文档的显示。
    基于source_table和requested_fields定制输出格式。
    """
    if not doc:
        return ""
    
    doc_info = []
    
    # 获取文档ID
    doc_id = "N/A"
    if hasattr(doc, 'node_id'):
        doc_id = doc.node_id
    elif hasattr(doc, 'id'):
        doc_id = doc.id
    elif isinstance(doc, dict):
        doc_id = doc.get("node_id", doc.get("id", "N/A"))
    doc_info.append(f"--- 记录 ID: {doc_id} ---")
    
    # 获取元数据
    metadata = {}
    if hasattr(doc, 'node') and hasattr(doc.node, 'metadata'):
        metadata = doc.node.metadata
    elif hasattr(doc, 'metadata'):
        metadata = doc.metadata
    elif isinstance(doc, dict):
        metadata = doc.get("metadata", {})
    
    source_table = metadata.get("source_table", "unknown")
    requested_fields = parsed_query.get("requested_fields", []) if parsed_query else []
    milvus_filters = parsed_query.get("milvus_filters", {}) if parsed_query else {}
    
    if source_table == "experiment_data":
        doc_info.append(f"来源: 实验数据")
        doc_info.append(f"项目建库类型: {metadata.get('project_library_type', 'N/A')}")
        doc_info.append(f"物种: {metadata.get('species', 'N/A')}")
        doc_info.append(f"样本详细类型: {metadata.get('sample_detailed_type', 'N/A')}")
        doc_info.append(f"实验方案: {metadata.get('prep_method', 'N/A')}")
        
        # 处理特定请求字段
        for field in requested_fields:
            value = metadata.get(field, "N/A")
            if field == "annotation_results_full" or "annotation" in field.lower():
                # 处理注释结果
                doc_info.append(f"人工注释结果: {value}")
            elif field == "cell_viability_percent":
                doc_info.append(f"细胞活率: {value}%")
            elif field == "related_article_link":
                doc_info.append(f"相关文章链接: {value}")
            else:
                doc_info.append(f"{field.replace('_', ' ').title()}: {value}")
        
        # 如果没有requested_fields，添加一些默认的核心指标
        if not requested_fields:
            doc_info.append(f"细胞活率: {metadata.get('cell_viability_percent', 'N/A')}%")
            doc_info.append(f"捕获细胞数: {metadata.get('captured_cells', 'N/A')}")
            annotation = metadata.get('cell_annotation_result', '') or metadata.get('注释结果', '')
            if annotation:
                doc_info.append(f"人工注释结果: {annotation[:100]}{'...' if len(annotation) > 100 else ''}")
    
    elif source_table == "preparation_guidelines":
        doc_info.append(f"来源: 样本制备指南")
        doc_info.append(f"产品类型: {metadata.get('product_level1', 'N/A')} > {metadata.get('product_level2', 'N/A')} > {metadata.get('product_level3', 'N/A')}")
        doc_info.append(f"样本大类: {metadata.get('sample_category', 'N/A')}")
        doc_info.append(f"样本类型: {metadata.get('sample_type', 'N/A')}")
        doc_info.append(f"组织类型: {metadata.get('tissue_type', 'N/A')}")
        doc_info.append(f"制备方案: {metadata.get('prep_method', 'N/A')}")
        doc_info.append(f"建议送样量: {metadata.get('recommended_amount_mg', 'N/A')}mg (约{metadata.get('qualitative_description_text', 'N/A')})")
        doc_info.append(f"风险级别: {metadata.get('risk_level', 'N/A')}")
        
        # 处理特定请求字段
        for field in requested_fields:
            value = metadata.get(field, "N/A")
            if field == "notes_full":
                doc_info.append(f"备注: {value}")
            elif field == "handling_notes":
                doc_info.append(f"取样送样的注意事项: {value}")
            elif field == "preparation_method_doc":
                doc_info.append(f"样本准备方法SOP: {value}")
            else:
                doc_info.append(f"{field.replace('_', ' ').title()}: {value}")
        
        # 如果没有requested_fields，添加一些默认的核心信息
        if not requested_fields:
            doc_info.append(f"风险级别: {metadata.get('risk_level', 'N/A')}")
            notes = metadata.get('notes_full', '') or metadata.get('备注', '')
            if notes:
                doc_info.append(f"备注: {notes[:100]}{'...' if len(notes) > 100 else ''}")
    
    else:
        # 通用格式
        doc_info.append(f"来源: {source_table}")
        doc_info.append(f"样本类型: {metadata.get('sample_type', 'N/A')}")
        doc_info.append(f"组织类型: {metadata.get('sample_detailed_type', 'N/A')}")
        
        # 添加核心指标
        if "cell_viability_percent" in metadata:
            doc_info.append(f"细胞活率: {metadata['cell_viability_percent']}%")
        if "captured_cells" in metadata:
            doc_info.append(f"捕获细胞数: {metadata['captured_cells']}")
    
    return "\n".join(doc_info)


async def _generate_summary(query_text: str, context_nodes: List[Dict], intent: str = "unknown", all_retrieved_rows: List[Dict] = None, parsed_query: Optional[Dict] = None, websocket=None) -> str:
    logger.info(f"🚀 进入_generate_summary函数")
    logger.info(f"📊 context_nodes数量: {len(context_nodes)}")
    logger.info(f"📊 all_retrieved_rows数量: {len(all_retrieved_rows) if all_retrieved_rows else 0}")
    logger.info(f"🎯 意图: {intent}")
    logger.info(f"🔍 查询文本: {query_text}")
    logger.info(f"🔌 WebSocket: {websocket is not None}")
    logger.info(f"📋 parsed_query: {json.dumps(parsed_query, ensure_ascii=False) if parsed_query else 'None'}")
    
    # 确定用于分析的数据 - 必须使用all_retrieved_rows，不能使用筛选后的context_nodes
    all_data = all_retrieved_rows if all_retrieved_rows else context_nodes
    total_samples = len(all_data)
    
    if total_samples == 0:
        logger.info("⚠️ 没有数据用于分析，返回默认消息")
        
        # ✅ 提取用户过滤条件用于友好提示
        milvus_filters = parsed_query.get("milvus_filters", {}) if parsed_query else {}
        
        # 构建过滤条件描述
        filter_descriptions = []
        for key, value in milvus_filters.items():
            if key == "species":
                filter_descriptions.append(f"物种: {value}")
            elif key == "sample_detailed_type":
                filter_descriptions.append(f"组织: {value}")
            elif key == "experiment_protocol":
                filter_descriptions.append(f"实验方案: {value}")
            elif key == "source_table":
                if value == "experiment_data":
                    filter_descriptions.append("数据类型: 项目经验")
                elif value == "preparation_guidelines":
                    filter_descriptions.append("数据类型: 样本准备指南")
            # 可以根据需要添加更多字段
        
        filter_text = "、".join(filter_descriptions) if filter_descriptions else "您指定的条件"
        
        result = f"抱歉,数据库中未找到符合【{filter_text}】的数据。\n\n建议您:\n1. 检查物种/组织名称是否准确\n2. 尝试更通用的查询条件\n3. 联系技术支持确认是否有相关数据"
        
        if websocket:
            await websocket.send_json({"type": "content", "value": result})
            await websocket.send_json({"type": "end_of_stream", "status": "success"})
        return result
    
    logger.info(f"📊 用于分析的数据量: {total_samples} 条 (基于所有检索数据，未做任何筛选)")
    
    # 1. 使用 ContextFormatter 格式化所有文档（零截断版）
    logger.info("📝 开始使用 ContextFormatter 格式化检索到的文档")

    # 获取意图
    intent = parsed_query.get("query_intent", "unknown") if parsed_query else "unknown"

    # 调用 ContextFormatter（自动移除冗余字段，保留所有有价值数据）
    full_formatted_context = ContextFormatter.format_for_llm(all_data, intent)

    logger.info(f"📊 格式化后的上下文长度: {len(full_formatted_context)} 字符")
    logger.info(f"✅ ContextFormatter 已完成数据清洗，移除 full_row_json 等冗余字段")
    
    # 2. 上下文长度管理策略
    final_llm_context = ""
    user_guidance_message = ""
    max_context_tokens = 1000000  # 默认使用较大的上下文窗口
    
    # 估计Prompt中其他部分的token数量，给实际数据留出空间
    overhead_tokens = get_token_count(query_text) + 500  # 留一些Buffer给指令和LLM生成答案
    available_context_tokens = max_context_tokens - overhead_tokens
    
    logger.info(f"🧮 Token管理: 最大上下文窗口={max_context_tokens}, 可用空间={available_context_tokens}")
    
    # 如果所有数据能装下
    if get_token_count(full_formatted_context) <= available_context_tokens:
        final_llm_context = full_formatted_context
        user_guidance_message = f"以下是您查询到的全部 {len(all_data)} 条符合条件的记录的详细信息："
    else:
        # 策略：展示前N条详细信息 + 剩余数据的概览
        logger.info("📏 上下文过长，需要进行截断处理")
        
        # 由于使用了 ContextFormatter，我们直接使用完整上下文进行截断
        # 这里简化处理，直接使用 full_formatted_context
        final_llm_context = full_formatted_context[:int(available_context_tokens * 2)]  # 假设每个token平均2个字符
        final_llm_context += "...\n(上下文过长，已截断)"
        
        user_guidance_message = (
            f"检测到 {len(all_data)} 条符合条件的记录。由于数据量庞大，"
            f"以下为您展示了部分记录的详细信息。"
        )
        
        logger.info(f"📋 由于上下文过长，已对 {len(all_data)} 条记录的详细信息进行了截断处理")
    
    # 3. 根据意图识别结果动态选择提示词
    logger.info(f"🎯 选择Prompt模板,当前意图: {parsed_query.get('query_intent') if parsed_query else 'unknown'}")

    if parsed_query:
        detected_intent = parsed_query.get("query_intent", "unknown")
        
        # 优先使用意图识别结果
        if detected_intent == "query_experiment_data":
            prompt_config = prompt_manager.get_prompt("query_experiment_data")
        elif detected_intent == "query_preparation_guidelines":
            prompt_config = prompt_manager.get_prompt("query_preparation_guidelines")
        else:
            # 兜底:使用通用Prompt
            prompt_config = prompt_manager.get_prompt("unknown")
            logger.warning(f"⚠️ 未识别的意图 '{detected_intent}',使用通用Prompt")
    else:
        # 兜底:如果没有parsed_query,使用传入的intent参数
        prompt_config = prompt_manager.get_prompt(intent)
        logger.warning(f"⚠️ 缺少parsed_query,使用传入的intent参数: {intent}")
    
    # 构建用户提示词
    user_prompt = (
        f"用户问题: {query_text}\n"
        f"\n"
        f"【数据库检索结果】:\n"
        f"{final_llm_context}"
    )

    try:
        # 直接使用Settings._llm属性，避免触发默认的resolve_llm逻辑
        llm_instance = getattr(Settings, '_llm', None)
        if not llm_instance:
            logger.error("❌ LLM 未初始化")
            result = "LLM 未初始化，无法生成回答"
            if websocket:
                await websocket.send_json({"type": "content", "value": result})
                await websocket.send_json({"type": "end_of_stream", "status": "success"})
            return result
            
        # 3. 调用 LLM
        logger.info(f"📞 调用 LLM，意图: {intent}")
        logger.info(f"📄 系统提示词: {prompt_config['system_prompt'][:100]}...")
        logger.info(f"👤 用户提示词长度: {len(user_prompt)} 字符")
        
        logger.info(f"🧠 LLM 实例: {llm_instance}")
        logger.info(f"🔑 LLM API_KEY: {llm_instance.api_key if hasattr(llm_instance, 'api_key') else 'No api_key attribute'}")
        
        try:
            # 构建完整的提示词
            full_prompt = f"{prompt_config['system_prompt']}\n\n用户问题：{user_prompt}"
            logger.info(f"📝 完整提示词: {full_prompt[:100]}...")
            
            if websocket:
                # 流式生成模式 - 实时发送每个token，无缓冲
                logger.info("🔄 尝试使用astream_complete方法调用LLM...")
                response_gen = await llm_instance.astream_complete(full_prompt)
                
                answer = ""
                
                async for response in response_gen:
                    chunk = response.delta or response.text
                    if chunk:
                        answer += chunk
                        # 立即发送每个生成的token，不使用缓冲
                        await websocket.send_json({"type": "content", "value": chunk})
                
                # 发送结束标记
                await websocket.send_json({"type": "end_of_stream", "status": "success"})
            else:
                # 非流式模式
                logger.info("🔄 尝试使用complete方法调用LLM...")
                response = llm_instance.complete(full_prompt)
                answer = str(response.text)
            
            logger.info(f"✅ LLM 调用成功")
            logger.info(f"📝 响应内容长度: {len(answer)}")
            
            if not answer:
                logger.warning("⚠️ LLM 生成的回答为空")
                answer = "抱歉，LLM 未能生成有效的回答。"
                if websocket:
                    await websocket.send_json({"type": "content", "value": answer})
                    await websocket.send_json({"type": "end_of_stream", "status": "success"})
            
            # 统计LLM tokens消耗
            from utils.token_counter import token_counter
            token_counter.count_llm_tokens(full_prompt, answer)
            
            # 获取token统计信息
            token_stats = token_counter.get_total_stats()
            
            return answer
        except Exception as e:
            logger.error(f"❌ LLM 调用失败: {e}")
            logger.error(f"❌ LLM 调用失败详情: {str(e)}")
            import traceback
            logger.error(f"❌ LLM 调用失败堆栈信息: {traceback.format_exc()}")
            result = f"生成回答时遇到错误: {str(e)}，请检查LLM配置和API密钥。"
            if websocket:
                await websocket.send_json({"type": "content", "value": result})
                await websocket.send_json({"type": "end_of_stream", "status": "error"})
            return result
    except Exception as e:
        logger.error(f"❌ LLM 生成失败: {e}")
        logger.error(f"❌ LLM 生成失败详情: {str(e)}")
        # 打印更详细的错误信息，方便调试
        import traceback
        logger.error(f"❌ LLM 生成失败堆栈信息: {traceback.format_exc()}")
        result = f"生成回答时遇到错误: {str(e)}，请检查LLM配置和API密钥。"
        if websocket:
            await websocket.send_json({"type": "content", "value": result})
            await websocket.send_json({"type": "end_of_stream", "status": "error"})
        return result

# ==========================================
# 3. 策略 A: 结构化表格检索 (纯过滤)
# ==========================================
async def _strategy_structured_table(index, query_text, auth, column_filters, llm_top_k, milvus_expr=None):
    # 1. 构建 LlamaIndex MetadataFilters
    logger.info(f"🔍 原始过滤条件: {column_filters}")
    
    # 初始化 all_nodes 变量
    all_nodes = []
    
    # 构建过滤条件列表
    filters_list = [
        MetadataFilter(key="department", operator=FilterOperator.EQ, value=auth.department)
    ]
    
    # 添加用户提供的过滤条件
    for k, v in column_filters.items():
        if v != "*":
            logger.info(f"🔍 处理过滤条件: '{k}' = '{v}'")
            
            # 检查值类型，跳过布尔值字段（LlamaIndex的MetadataFilter不支持布尔值）
            if isinstance(v, bool):
                logger.warning(f"⚠️ 过滤值 '{v}' 是布尔类型，LlamaIndex不支持布尔值过滤，跳过此条件")
                continue
            
            # 跳过表示空值的字符串
            if isinstance(v, str) and v.lower() in ["null", "none", "空", ""]:
                logger.info(f"⚠️ 过滤值 '{v}' 表示空值，跳过此条件")
                continue
            
            # 处理样本类型的不同表述方式
            if (k == "sample_type_exp" or k == "sample_type_prep" or k == "tissue_type_prep") and isinstance(v, str):
                # 记录样本类型变体，用于后续的文本匹配
                sample_type_variants = [v]
                # 如果值包含"组织"，添加不包含"组织"的版本
                if "组织" in v:
                    sample_type_variants.append(v.replace("组织", ""))
                # 如果值不包含"组织"，添加包含"组织"的版本
                elif "组织" not in v:
                    sample_type_variants.append(v + "组织")
                logger.info(f"🔍 为样本类型 '{v}' 生成变体: {sample_type_variants}")
                # 注意：由于 LlamaIndex 的 MetadataFilter 不支持 OR 条件，
                # 我们仍然使用原始值，但在后续的文本匹配中处理变体
            
            # 检查字段名是否包含特殊字符（换行符、中文括号等）
            has_special_chars = any(c in k for c in ['\n', '\r', '\t', '（', '）'])
            if has_special_chars:
                logger.warning(f"⚠️ 字段名 '{k}' 包含特殊字符，跳过 LlamaIndex 过滤")
                continue
            
            # 直接使用原始字段名，根据 field_mapping.py 中的定义
            filters_list.append(MetadataFilter(key=k, operator=FilterOperator.EQ, value=v))
            logger.info(f"✅ 添加过滤条件: {k} == {v}")
    
    # 构建 MetadataFilters 对象
    metadata_filters = MetadataFilters(filters=filters_list, condition="and")
    logger.info(f"✅ 构建完成 MetadataFilters")
    
    # 2. 使用 LlamaIndex 索引进行检索
    all_nodes = []
    
    # 第一步：优先使用 PyMilvus API 进行精确查询
    if milvus_expr:
        try:
            logger.info(f"📝 使用 PyMilvus API 进行精确查询")
            
            # 直接使用 PyMilvus API 查询数据
            from pymilvus import connections, Collection
            from config import settings
            
            # ✅ 优化:使用长连接模式,只在连接失效时重连
            try:
                # 检查连接是否存在且有效
                if not connections.has_connection("default"):
                    logger.info(f"🔗 建立 Milvus 长连接: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
                    connections.connect(
                        alias="default",
                        host=settings.MILVUS_HOST,
                        port=settings.MILVUS_PORT,
                        timeout=30
                    )
                    logger.info(f"✅ Milvus 连接成功")
                else:
                    logger.info(f"♻️ 复用现有 Milvus 连接")
            except Exception as e:
                logger.error(f"❌ Milvus 连接失败: {str(e)}")
                # 如果连接失败,尝试重新连接
                try:
                    if connections.has_connection("default"):
                        connections.disconnect("default")
                    connections.connect(
                        alias="default",
                        host=settings.MILVUS_HOST,
                        port=settings.MILVUS_PORT,
                        timeout=30
                    )
                    logger.info(f"✅ Milvus 重连成功")
                except Exception as retry_e:
                    logger.error(f"❌ Milvus 重连也失败: {str(retry_e)}")
                    raise
            
            # 获取集合
            collection = Collection("dept_market")
            logger.info(f"✅ 成功获取集合")
            
            # 执行查询
            logger.info(f"🔍 执行 PyMilvus 查询，表达式: {milvus_expr}")
            
            # 根据 field_mapping.py 构建正确的查询表达式
            # 1. 保持原样，因为 field_mapping.py 中定义的就是这些字段名
            fixed_expr = milvus_expr
            
            # 2. 处理样本类型的不同表述方式
            import re
            # 查找所有样本类型相关的条件
            sample_type_patterns = [
                r'sample_type_exp == \'(.*?)\'',
                r'sample_type_prep == \'(.*?)\'',
                r'tissue_type_prep == \'(.*?)\''
            ]
            
            for pattern in sample_type_patterns:
                matches = re.finditer(pattern, fixed_expr)
                for match in matches:
                    field_name = match.group(0).split(' == ')[0]
                    sample_value = match.group(1)
                    # 生成变体
                    variants = [sample_value]
                    # 如果值包含"组织"，添加不包含"组织"的版本
                    if "组织" in sample_value:
                        variants.append(sample_value.replace("组织", ""))
                    # 如果值不包含"组织"，添加包含"组织"的版本
                    elif "组织" not in sample_value:
                        variants.append(sample_value + "组织")
                    # 构建OR条件
                    if len(variants) > 1:
                        sample_conditions = [f"{field_name} == '{v}'" for v in variants]
                        sample_conditions_str = "(" + " or ".join(sample_conditions) + ")"
                        # 替换原来的条件
                        fixed_expr = fixed_expr.replace(match.group(0), sample_conditions_str)
            
            logger.info(f"🔍 修复后的查询表达式: {fixed_expr}")
            
            # ✅ 记录查询表达式到专门的查询日志
            from datetime import datetime
            logger.info(f"📊 [QUERY_LOG] 执行 Milvus 查询")
            logger.info(f"📊 [QUERY_LOG] 查询表达式: {fixed_expr}")
            logger.info(f"📊 [QUERY_LOG] 查询限制: 2000")
            logger.info(f"📊 [QUERY_LOG] 查询时间: {datetime.now().isoformat()}")
            
            # 执行完整查询(调整限制为2000)
            query_results = collection.query(
                expr=fixed_expr,
                output_fields=["*"],
                limit=2000  # ✅ 从1000调整为2000
            )
            
            logger.info(f"📊 [QUERY_LOG] 查询结果数量: {len(query_results)}")
            logger.info(f"✅ PyMilvus 查询命中: {len(query_results)} 条数据")
            
            # 如果有结果，转换为 LlamaIndex Node 格式
            if query_results:
                # 转换为 LlamaIndex Node 格式
                for result in query_results:
                    # 从查询结果构建 TextNode
                    from llama_index.core.schema import TextNode
                    text_node = TextNode(
                        text=result.get("text", ""),
                        metadata=result,
                        id_=str(result.get("id", ""))
                    )
                    all_nodes.append(text_node)
                logger.info(f"✅ 转换完成 TextNode，共 {len(all_nodes)} 条")
                
                # 如果有结果，直接返回，不执行第二步
                if all_nodes:
                    logger.info(f"✅ 精确查询成功，返回 {len(all_nodes)} 条结果")
                    return {
                        "mode": "structured_table",
                        "best_rows": all_nodes,
                        "all_rows": all_nodes
                    }
            else:
                # ✅ 0结果处理:不再尝试fallback,直接返回空结果
                logger.warning(f"⚠️ PyMilvus 查询返回0条数据")
                logger.warning(f"⚠️ 查询表达式: {fixed_expr}")
                logger.warning(f"⚠️ 过滤条件: {json.dumps(column_filters, ensure_ascii=False)}")
                
                return {
                    "mode": "structured_table",
                    "best_rows": [],
                    "all_rows": []
                }
        except Exception as e:
            logger.error(f"❌ PyMilvus 精确查询失败: {str(e)}")
    
    logger.info(f"📊 [Table] 最终检索命中: {len(all_nodes)} 条数据")

    # 3. 样本去重：优化去重逻辑，避免过度去重
    unique_nodes = []
    seen_rows = set()  # 使用更独特的标识符组合：(filename, sheet_name, row_index, chunk_id)，避免同一文件的多行数据被过度去重
    
    for node in all_nodes:
        if hasattr(node, 'node'):
            metadata = node.node.metadata if hasattr(node.node, 'metadata') else {}
        elif hasattr(node, 'metadata'):
            metadata = node.metadata
        elif isinstance(node, dict):
            metadata = node.get('metadata', {})
        else:
            metadata = {}
        
        # 提取更独特的标识符组合，确保同一文件的不同行能被正确保留
        filename = metadata.get('filename', 'Unknown')
        sheet_name = metadata.get('sheet_name', 'Unknown')
        row_index = str(metadata.get('row_index', 'Unknown'))
        chunk_id = str(metadata.get('chunk_id', 'Unknown'))
        
        # 创建更独特的行标识
        row_key = (filename, sheet_name, row_index, chunk_id)
        
        # 如果是第一次遇到这个行，添加到结果中
        if row_key not in seen_rows:
            seen_rows.add(row_key)
            unique_nodes.append(node)
    
    logger.info(f"📊 去重后剩余: {len(unique_nodes)} 条数据")

    # 4. 挑选 best_rows (用于 LLM)
    best_nodes = []
    
    # 返回所有去重后的结果，不限制数量
    if unique_nodes:
        best_nodes = unique_nodes
        logger.info(f"📝 直接返回去重后的所有 {len(best_nodes)} 条数据")
    else:
        best_nodes = []

    return {
        "mode": "table",
        "all_rows": _format_nodes(unique_nodes, keep_full_json=True),
        "best_rows": _format_nodes(best_nodes, keep_full_json=False)
    }

# ==========================================
# 4. 策略 B: 通用语义检索
# ==========================================
async def _strategy_general_semantic(index, query_text, auth, top_k):
    # 仅部门过滤
    filters = MetadataFilters(filters=[
        MetadataFilter(key="department", operator=FilterOperator.EQ, value=auth.department)
    ])
    
    # 扩大检索范围 (Retrieve) -> 重排序 (Rerank)
    retriever = index.as_retriever(similarity_top_k=top_k*4, filters=filters)
    candidate_nodes = await retriever.aretrieve(query_text)
    
    logger.info(f"🔍 [Semantic] 初筛命中: {len(candidate_nodes)} 条 -> 准备 Rerank")

    final_nodes = await _perform_rerank(query_text, candidate_nodes, top_k)
    
    return {
        "mode": "semantic",
        "all_rows": _format_nodes(candidate_nodes, keep_full_json=False), # 返回所有结果用于统计分析
        "best_rows": _format_nodes(final_nodes, keep_full_json=False)
    }

# ==========================================
# 5. 辅助函数
# ==========================================
async def _perform_rerank(query, nodes, top_n):
    """
    安全 Rerank：处理空列表、空查询、插件缺失等情况
    """
    if not nodes:
        return []
    
    # 如果查询词为空，无法计算相关性，直接返回
    if not query or not query.strip():
        return nodes[:top_n]

    if not DashScopeRerank:
        return nodes[:top_n]

    try:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return nodes[:top_n]

        reranker = DashScopeRerank(
            top_n=top_n, 
            model="gte-rerank-v2", 
            api_key=api_key,
            return_documents=True
        )
        
        from fastapi.concurrency import run_in_threadpool
        results = await run_in_threadpool(reranker.postprocess_nodes, nodes=nodes, query_str=query)
        
        return results
    except Exception as e:
        logger.warning(f"⚠️ Rerank 失败 (降级处理): {e}")
        return nodes[:top_n]

def _format_nodes(nodes, keep_full_json=True):
    results = []
    for node in nodes:
        # 正确处理TextNode对象和字典对象
        if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
            safe_meta = node.node.metadata.copy() if node.node.metadata else {}
        elif hasattr(node, 'metadata'):
            safe_meta = node.metadata.copy() if node.metadata else {}
        elif isinstance(node, dict):
            metadata = node.get('metadata', {})
            safe_meta = metadata.copy() if metadata else {}
        else:
            safe_meta = {}
        
        # 提取 full_row_json
        row_data = {}
        if "full_row_json" in safe_meta:
            try:
                raw_json = safe_meta["full_row_json"]
                row_data = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
            except: pass
            
            if not keep_full_json:
                del safe_meta["full_row_json"]

        # 获取节点属性
        if hasattr(node, 'node'):
            node_id = node.node.node_id if hasattr(node.node, 'node_id') else str(id(node))
            text = node.node.text if hasattr(node.node, 'text') else ""
            score = float(node.score) if hasattr(node, 'score') else 0.0
        else:
            if hasattr(node, 'node_id'):
                node_id = node.node_id
            elif isinstance(node, dict):
                node_id = node.get('node_id', str(id(node)))
            else:
                node_id = str(id(node))
            
            if hasattr(node, 'text'):
                text = node.text
            elif isinstance(node, dict):
                text = node.get('text', "")
            else:
                text = ""
            
            if hasattr(node, 'score'):
                score = float(node.score)
            elif isinstance(node, dict) and 'score' in node:
                score = float(node.get('score', 0.0))
            else:
                score = 0.0
        
        results.append({
            "node_id": node_id,
            "text": text,
            "score": score,
            "metadata": safe_meta,
            "row_data": row_data
        })
    return results