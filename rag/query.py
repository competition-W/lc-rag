import asyncio
import json
import os
import re
from typing import List, Dict, Optional, Any

from llama_index.core import Settings
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.core.schema import NodeWithScore
from llama_index.core.llms import ChatMessage, MessageRole

from .milvus_manager import milvus_manager
from .prompt_manager import prompt_manager
from utils.auth import AuthContext
from utils.logger import logger

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

_EMBED_MODEL_INIT = False

def _ensure_embed_model():
    """确保 Embedding 模型初始化"""
    global _EMBED_MODEL_INIT
    if not _EMBED_MODEL_INIT:
        if not Settings.embed_model:
            Settings.embed_model = get_embed_model() 
        _EMBED_MODEL_INIT = True

# ==========================================
# 1. 统一检索入口
# ==========================================
async def unified_query_service(
    auth: AuthContext,
    query_text: str,
    column_filters: Optional[Dict[str, str]] = None,
    llm_top_k: int = 8,
    semantic_top_k: int = 15,
    intent: str = "unknown",
    websocket = None
) -> Dict[str, Any]:
    """
    智能检索主入口
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

    # 2. 路由策略：根据意图和是否有 Filters 决定走哪条路
    search_result = {}
    
    # 🔍 调试日志：看看 Service 到底收到了什么
    logger.info(f"🛡️ [Service] 收到请求 -> Query: '{query_text}' | Filters: {column_filters} | Intent: {intent}")

    # 基于意图的检索策略路由
    if intent in ["sample_query", "project_query", "query_experiment_data", "query_preparation_guidelines"]:
        # 样本准备查询和项目经验查询：纯关键词检索
        logger.info(f"🛤️ 命中策略: [{intent} - 纯关键词检索]")
        # 检查是否有过滤条件
        if column_filters and len(column_filters) > 0:
            # 有过滤条件，使用结构化表格检索
            search_result = await _strategy_structured_table(
                index, query_text, auth, column_filters, llm_top_k
            )
        else:
            # 没有过滤条件，使用关键词检索（不使用语义检索）
            # 对于这两种意图，我们希望只进行关键词检索，而不是语义检索
            # 这里我们使用一个空的过滤条件来触发结构化表格检索，使用查询文本作为关键词
            logger.info(f"📝 对于 {intent} 意图，没有过滤条件，使用查询文本作为关键词进行检索")
            # 调用结构化表格检索，但不使用语义检索
            search_result = await _strategy_structured_table(
                index, query_text, auth, {}, llm_top_k
            )
    else:
        # 其他意图：根据是否有 Filters 决定走哪条路
        logger.info(f"🛤️ 命中策略: [其他意图 - 通用检索策略]")
        if column_filters and len(column_filters) > 0:
            logger.info("🛤️ 子策略: [结构化表格检索]")
            search_result = await _strategy_structured_table(
                index, query_text, auth, column_filters, llm_top_k
            )
        else:
            logger.info("🛤️ 子策略: [通用语义检索]")
            search_result = await _strategy_general_semantic(
                index, query_text, auth, semantic_top_k
            )

    # 3. 检查检索结果条数，决定是否调用LLM生成回答
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
        websocket=websocket  # 传递WebSocket连接用于流式输出
    )
    
    logger.info(f"📝 生成的LLM回答: {llm_answer[:100]}...")
    logger.info(f"📝 生成的LLM回答长度: {len(llm_answer)} 字符")

    # 获取token统计信息
    from utils.token_counter import token_counter
    token_stats = token_counter.get_total_stats()
    
    result = {
        "mode": search_result["mode"],
        "answer": llm_answer,
        "sources": best_rows,
        "all_rows": search_result["all_rows"],
        "token_stats": token_stats
    }
    
    logger.info(f"✅ unified_query_service返回结果: {result.keys()}")
    logger.info(f"✅ answer字段长度: {len(result['answer'])} 字符")
    
    return result

# # ==========================================
# # 2. LLM 生成模块
# # ==========================================
# async def _generate_summary(query_text: str, context_nodes: List[Dict], intent: str = "sample_query") -> str:
#     if not context_nodes:
#         return "抱歉，未找到匹配的数据。"
    
#     # 数据扁平化
#     context_str_list = []
#     for idx, item in enumerate(context_nodes):
#         meta = item.get("metadata", {}).copy()
#         # 清理无关字段
#         for ignore_key in ['chunk_id', 'owner', 'uploader', 'full_row_json', 'doc_type', 'vector_id']:
#             meta.pop(ignore_key, None)
#         context_str_list.append(f"[{idx+1}] {json.dumps(meta, ensure_ascii=False)}")
    
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
        "col_xbzl\n（w）", "col_jie_tuan_lv", "col_xi_bao_huo_lv", 
        "col_you_he_lv", "col_bu_huo_xi_bao_shu", 
        "col_zu_zhi_zhong_liang_shu_zhi", "col_zu_zhi_zhong_liang_dan_wei",
        "col_ding_xing_miao_shu_ji_gen_ji_tiao_deng", "col_hszl\n（rinz）",
        "col_kang_ti_xin_xi", "col_lsfxfa", "is_lysis", "is_dead_removal",
        "storage_method", "col_liu_shi_yu_fou", "col_syfa\n（jl/ch）"
    ]
    
    data_metrics = [
        "col_shu_ju_liang", "col_ji_yin_zhong_wei_shu", "col_zsjg_zztyxzs"
    ]
    
    annotation_metrics = [
        "cell_annotation_result", "col_zsjg_zztyxzs", "注释结果", "细胞注释", "cell_type", "cell_types"
    ]
    
    sample_info = [
        "platform", "species", "category", "tissue", "filename", "sheet_name"
    ]
    
    reference_materials = [
        "col_zzxhfags", "col_xgzzyhwzlj", "col_fswdlj", "col_spzblj"
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
        "col_zu_zhi_zhong_liang_shu_zhi", "col_xbzl\n（w）", "col_jie_tuan_lv", 
        "col_xi_bao_huo_lv", "col_you_he_lv", "col_bu_huo_xi_bao_shu", 
        "col_hszl\n（rinz）"
    ]
    
    data_metrics = [
        "col_shu_ju_liang", "col_ji_yin_zhong_liang_shu", "col_ji_yin_zhong_wei_shu", 
        "col_zsjg_zztyxzs"
    ]
    
    # 定义指标名称映射，将英文列名转换为友好的中文名称
    metric_name_mapping = {
        "col_zu_zhi_zhong_liang_shu_zhi": "组织重量",
        "col_shu_ju_liang": "数据量",
        "col_ji_yin_zhong_wei_shu": "基因中位数",
        "col_xbzl\n（w）": "细胞总量",
        "col_jie_tuan_lv": "结团率",
        "col_xi_bao_huo_lv": "细胞活率",
        "col_you_he_lv": "有效核率",
        "col_bu_huo_xi_bao_shu": "捕获细胞数",
        "col_hszl\n（rinz）": "核碎片率",
        "col_zsjg_zztyxzs": "注释结果-组织特异性指标"
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
        meta = node.get("metadata", {})
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

async def _generate_summary(query_text: str, context_nodes: List[Dict], intent: str = "unknown", all_retrieved_rows: List[Dict] = None, websocket=None) -> str:
    logger.info(f"🚀 进入_generate_summary函数")
    logger.info(f"📊 context_nodes数量: {len(context_nodes)}")
    logger.info(f"📊 all_retrieved_rows数量: {len(all_retrieved_rows) if all_retrieved_rows else 0}")
    logger.info(f"🎯 意图: {intent}")
    logger.info(f"🔍 查询文本: {query_text}")
    logger.info(f"🔌 WebSocket: {websocket is not None}")
    
    # 确定用于分析的数据 - 必须使用all_retrieved_rows，不能使用筛选后的context_nodes
    all_data = all_retrieved_rows if all_retrieved_rows else context_nodes
    total_samples = len(all_data)
    
    if total_samples == 0:
        logger.info("⚠️ 没有数据用于分析，返回默认消息")
        result = "抱歉，数据库中未找到符合条件的数据。"
        if websocket:
            await websocket.send_json({"type": "content", "value": result})
            await websocket.send_json({"type": "end_of_stream", "status": "success"})
        return result
    
    logger.info(f"📊 用于分析的数据量: {total_samples} 条 (基于所有检索数据，未做任何筛选)")
    
    # 1. 分析用户问题类型
    lower_query = query_text.lower() if query_text else ""
    
    # 定义问题类型关键词
    numeric_query_keywords = ["数据指标", "实验指标", "指标", "统计", "统计分析", "数值", "数值型", "数值指标"]
    # 扩展非数值型查询关键词，增加细胞鉴定相关关键词
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
    
    logger.info(f"🔍 查询类型分析结果: 数值型查询={is_numeric_query}, 注释/结果查询={is_annotation_query}")
    
    # 2. 提取关键词信息
    keywords = []
    if query_text:
        keywords = re.findall(r'[\u4e00-\u9fa5]+', query_text)[:5]  # 提取前5个中文关键词
    
    # 3. 构建上下文
    context_intro = f"您的查询：{query_text}\n\n"
    context_intro += f"⚠️ 重要说明：以下分析基于所有检索到的 {total_samples} 条数据，未做任何筛选。\n"
    
    if keywords:
        context_intro += f"关键词：{', '.join(keywords)}\n"
    
    full_context = context_intro
    
    # 根据查询类型构建不同的上下文
    if is_numeric_query:
        # 数值型查询：生成包含统计分析的表格
        logger.info("📊 数值型查询，生成包含统计分析的表格")
        # 首先获取统计汇总数据，传入query_text用于区分数据指标和实验指标
        aggregate_text = _aggregate_numeric_metrics(all_data, query_text)
        # 然后构建包含所有检索结果的详细信息，用于评价分析
        detailed_info_text = "\n\n**所有检索结果详细信息**\n"
        for idx, item in enumerate(all_data[:10]):  # 最多显示前10个，避免上下文过长
            meta = item.get("metadata", {})
            formatted_text = _format_node_for_llm(meta, query_text)
            detailed_info_text += f"【样本 {idx+1}】\n{formatted_text}\n\n"
        if len(all_data) > 10:
            detailed_info_text += f"... 还有 {len(all_data) - 10} 条数据未显示，完整统计基于所有 {total_samples} 条数据\n"
        full_context += aggregate_text + detailed_info_text
    elif is_annotation_query or "注释" in lower_query or "结果" in lower_query:
        # 非数值型查询：展开所有相关结果
        logger.info("📝 非数值型查询，展开所有相关结果")
        result_str_list = []
        for idx, item in enumerate(all_data):
            meta = item.get("metadata", {})
            # 使用专门的格式化函数，提取相关数据
            formatted_text = _format_node_for_llm(meta, query_text)
            result_str_list.append(f"【结果 {idx+1}】\n{formatted_text}")
        result_text = "\n\n".join(result_str_list)
        full_context += result_text
    else:
        # 默认情况：生成包含统计分析的表格
        logger.info("📊 默认情况，生成包含统计分析的表格")
        aggregate_text = _aggregate_numeric_metrics(all_data, query_text)
        full_context += aggregate_text
    
    # 输出提取的实验指标结果到控制台，方便调试
    logger.info(f"\n🔍 提取的实验指标结果:\n{full_context}")

    # 3. 根据查询类型动态选择提示词意图
    # 告诉模型：怎么读数据、怎么组织语言
    if is_annotation_query:
        # 注释/细胞鉴定查询使用专门的提示词
        prompt_config = prompt_manager.get_prompt("annotation_query")
    else:
        # 其他查询使用原有的意图
        prompt_config = prompt_manager.get_prompt(intent)
    
    # 构建用户提示词
    user_prompt = (
        f"用户问题: {query_text}\n"
        f"\n"
        f"【数据库检索结果】:\n"
        f"{full_context}"
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
            token_counter.count_llm_tokens(
                input_text=full_prompt,
                output_text=answer,
                model_name=llm_instance.model_name
            )
            
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
# async def _strategy_structured_table(index, query_text, auth, column_filters, llm_top_k):
#     # 1. 构建 Filter
#     # 强制加上部门隔离
#     filters_list = [
#         MetadataFilter(key="department", operator=FilterOperator.EQ, value=auth.department)
#     ]
#     # 加上用户的筛选条件
#     for k, v in column_filters.items():
#         # 注意：Milvus 的 EQ 是精确匹配，大小写敏感
#         filters_list.append(MetadataFilter(key=k, operator=FilterOperator.EQ, value=v))
    
#     metadata_filters = MetadataFilters(filters=filters_list, condition="and") # 确保是 AND 关系
    
#     # 2. 如果 query_text 为空（说明用户只说了筛选条件，比如"列出所有小鼠数据"）
#     # 我们使用 "*" 或 " " 进行全量匹配，不依赖向量相似度
#     is_pure_filter = not (query_text and query_text.strip())
#     search_text = query_text if not is_pure_filter else " " 

#     # 3. 执行检索 (Retrieval)
#     # 对于表格模式，我们希望尽可能多地拿回数据，然后让前端分页，所以取 2000
#     retriever = index.as_retriever(similarity_top_k=2000, filters=metadata_filters)
#     all_nodes = await retriever.aretrieve(search_text)
    
#     logger.info(f"📊 [Table] 过滤命中: {len(all_nodes)} 条数据")

#     # 4. 挑选 best_rows (用于 LLM)
#     best_nodes = []
#     if is_pure_filter:
#         # 如果没有文本查询，Rerank 没意义，直接取前 N 条
#         best_nodes = all_nodes[:llm_top_k]
#     else:
#         # 如果有文本意图，才进行 Rerank
#         best_nodes = await _perform_rerank(query_text, all_nodes[:100], llm_top_k)

#     return {
#         "mode": "table",
#         "all_rows": _format_nodes(all_nodes, keep_full_json=True),
#         "best_rows": _format_nodes(best_nodes, keep_full_json=False)
#     }

# 上面注释的那一版对于意图识别的结果太严格了，如果没有检索到完全符合意图识别的结果就把数据丢弃了
# 逻辑变更：如果“带关键词的检索”结果为 0，但只要有 Filters，我们就立刻扔掉关键词，直接把符合 Filter 的所有数据全捞出来（兜底）。
async def _strategy_structured_table(index, query_text, auth, column_filters, llm_top_k):
    # 1. 构建 LlamaIndex MetadataFilters
    logger.info(f"🔍 原始过滤条件: {column_filters}")
    
    # 构建过滤条件列表
    filters_list = [
        MetadataFilter(key="department", operator=FilterOperator.EQ, value=auth.department)
    ]
    
    # 添加用户提供的过滤条件
    for k, v in column_filters.items():
        if v != "*":
            logger.info(f"🔍 处理过滤条件: '{k}' = '{v}'")
            # 检查字段名是否包含特殊字符（换行符、中文括号等）
            has_special_chars = any(c in k for c in ['\n', '\r', '\t', '（', '）'])
            if has_special_chars:
                logger.warning(f"⚠️ 字段名 '{k}' 包含特殊字符，跳过 LlamaIndex 过滤")
                continue
            filters_list.append(MetadataFilter(key=k, operator=FilterOperator.EQ, value=v))
            logger.info(f"✅ 添加过滤条件: {k} == {v}")
    
    # 构建 MetadataFilters 对象
    metadata_filters = MetadataFilters(filters=filters_list, condition="and")
    logger.info(f"✅ 构建完成 MetadataFilters")
    
    # 2. 使用 LlamaIndex 索引进行检索
    all_nodes = []
    try:
        # 创建检索器 - 移除数量限制，获取所有匹配结果
        retriever = index.as_retriever(
            similarity_top_k=10000,  # 设置一个很大的值，确保获取所有匹配结果
            filters=metadata_filters
        )
        
        # 执行检索
        # 如果 query_text 为空，使用空格避免向量检索干扰
        search_text = query_text if query_text and query_text.strip() else " "
        logger.info(f"🔍 执行 LlamaIndex 检索，搜索词: '{search_text}'")
        
        all_nodes = await retriever.aretrieve(search_text)
        logger.info(f"✅ LlamaIndex 检索命中: {len(all_nodes)} 条数据")
        
        # 3. 如果检索结果为空，尝试移除可能有问题的字段
        if not all_nodes and len(filters_list) > 1:
            logger.info(f"⚠️ LlamaIndex 检索命中 0 条，尝试移除可能有问题的字段")
            
            # 构建移除特殊字段后的过滤条件
            simplified_filters = [
                MetadataFilter(key="department", operator=FilterOperator.EQ, value=auth.department)
            ]
            
            for k, v in column_filters.items():
                if v != "*" and 'syfa' not in k.lower():
                    # 检查字段名是否包含特殊字符（换行符、中文括号等）
                    has_special_chars = any(c in k for c in ['\n', '\r', '\t', '（', '）'])
                    if has_special_chars:
                        logger.warning(f"⚠️ 简化检索：字段名 '{k}' 包含特殊字符，跳过")
                        continue
                    # 跳过实验方案字段和包含特殊字符的字段
                    simplified_filters.append(MetadataFilter(key=k, operator=FilterOperator.EQ, value=v))
            
            simplified_metadata_filters = MetadataFilters(filters=simplified_filters, condition="and")
            logger.info(f"✅ 构建完成简化 MetadataFilters")
            
            # 执行简化检索
            simplified_retriever = index.as_retriever(
                similarity_top_k=1000,
                filters=simplified_metadata_filters
            )
            
            all_nodes = await simplified_retriever.aretrieve(search_text)
            logger.info(f"✅ 简化检索命中: {len(all_nodes)} 条数据")
    except Exception as e:
        logger.error(f"❌ 检索失败: {str(e)}")
        # 如果 LlamaIndex 检索失败，尝试使用原始的 PyMilvus API 方法
        logger.info("📝 降级使用 PyMilvus API")
        
        # 构建 PyMilvus 查询表达式，处理特殊字符
        expr_parts = [f"department == '{auth.department}'"]
        
        for k, v in column_filters.items():
            if v != "*":
                # 检查字段名是否包含特殊字符
                has_special_chars = any(c in k for c in ['\n', '\r', '\t', '（', '）'])
                if has_special_chars:
                    # 对于包含特殊字符的字段，使用反引号包裹
                    expr_parts.append(f"`{k}` == '{v}'")
                    logger.info(f"✅ PyMilvus 添加带反引号的过滤条件: `{k}` == '{v}'")
                elif 'syfa' not in k.lower():
                    # 普通字段直接添加
                    expr_parts.append(f"{k} == '{v}'")
                    logger.info(f"✅ PyMilvus 添加普通过滤条件: {k} == '{v}'")
        
        milvus_expr = " and ".join(expr_parts)
        logger.info(f"🔍 构建 PyMilvus 查询表达式: {milvus_expr}")
        
        # 直接使用 PyMilvus API 查询数据
        from pymilvus import connections, Collection
        from config import settings
        
        # 清理旧连接
        try:
            if connections.has_connection("default"):
                connections.disconnect("default")
        except Exception:
            pass
        
        # 重新连接 Milvus
        logger.info(f"🔍 连接 Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            timeout=30
        )
        logger.info(f"✅ Milvus 连接成功")
        
        # 获取集合
        collection = Collection("dept_market")
        logger.info(f"✅ 成功获取集合")
        
        # 执行查询
        logger.info(f"🔍 执行 PyMilvus 查询: {milvus_expr}")
        query_results = collection.query(
            expr=milvus_expr,
            output_fields=["*"],
            limit=1000
        )
        logger.info(f"✅ PyMilvus 查询命中: {len(query_results)} 条数据")
        
        # 如果有结果，转换为 LlamaIndex Node 格式
        if query_results:
            for result in query_results:
                # 从查询结果构建 TextNode
                from llama_index.core.schema import TextNode
                text_node = TextNode(
                    text=result.get("text", ""),
                    metadata=result,
                    id_=result.get("id", ""),
                    excluded_embed_metadata_keys=[],
                    excluded_llm_metadata_keys=[]
                )
                # 添加分数（直接查询没有相似度分数，设置为1.0）
                all_nodes.append(NodeWithScore(node=text_node, score=1.0))
        
        # 断开连接
        connections.disconnect("default")
        
    except Exception as e:
        logger.error(f"❌ 直接查询失败: {e}")
        # 降级：使用更简单的 LlamaIndex 检索器，不使用复杂的过滤条件
        # 1. 构建 Filter
        filters_list = [
            MetadataFilter(key="department", operator=FilterOperator.EQ, value=auth.department)
        ]
        
        # 只添加不包含特殊字符的过滤条件，提高成功率
        for k, v in main_filters:
            filters_list.append(MetadataFilter(key=k, operator=FilterOperator.EQ, value=v))
        
        metadata_filters = MetadataFilters(filters=filters_list, condition="and")
        
        # 2. 尝试使用不同的搜索词
        search_texts = ["数据", "实验", "样本", "小鼠", "心脏"]
        
        for search_text in search_texts:
            logger.info(f"📝 降级尝试: 使用搜索词 '{search_text}'")
            retriever = index.as_retriever(similarity_top_k=10000, filters=metadata_filters)
            nodes = await retriever.aretrieve(search_text)
            logger.info(f"📊 搜索词 '{search_text}' 命中: {len(nodes)} 条数据")
            
            if len(nodes) > len(all_nodes):
                all_nodes = nodes
            
            # 如果已经找到足够的结果，就停止尝试
            if len(all_nodes) > 0:
                break
        
        logger.info(f"📊 降级检索最终命中: {len(all_nodes)} 条数据")
    
    logger.info(f"📊 [Table] 最终检索命中: {len(all_nodes)} 条数据")

    # 4. 挑选 best_rows
    best_nodes = []
    
    # 直接返回所有数据，不做复杂处理
    if all_nodes:
        best_nodes = all_nodes[:llm_top_k]
        logger.info(f"📝 直接返回前 {len(best_nodes)} 条数据")
    else:
        best_nodes = []

    return {
        "mode": "table",
        "all_rows": _format_nodes(all_nodes, keep_full_json=True),
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
    # 设置一个很大的值，确保获取所有匹配结果用于统计分析
    retriever = index.as_retriever(similarity_top_k=10000, filters=filters)
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
        
        # 统计Rerank tokens消耗
        from utils.token_counter import token_counter
        # 提取文档文本
        docs = [node.text for node in nodes if hasattr(node, 'text')]
        token_counter.count_rerank_tokens(
            query=query,
            documents=docs,
            model_name="gte-rerank-v2"
        )
        
        return results
    except Exception as e:
        logger.warning(f"⚠️ Rerank 失败 (降级处理): {e}")
        return nodes[:top_n]

def _format_nodes(nodes, keep_full_json=True):
    results = []
    for node in nodes:
        safe_meta = node.metadata.copy() if node.metadata else {}
        
        # 提取 full_row_json
        row_data = {}
        if "full_row_json" in safe_meta:
            try:
                raw_json = safe_meta["full_row_json"]
                row_data = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
            except: pass
            
            if not keep_full_json:
                del safe_meta["full_row_json"]

        results.append({
            "node_id": node.node_id,
            "text": node.text,
            "score": float(node.score) if node.score else 0.0,
            "metadata": safe_meta,
            "row_data": row_data
        })
    return results