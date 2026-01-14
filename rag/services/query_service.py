import logging
import asyncio
import json
import os
from typing import List, Dict, Optional, Any

from llama_index.core import Settings
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.core.schema import NodeWithScore
from llama_index.core.llms import ChatMessage, MessageRole

from .milvus_manager import milvus_manager
from utils.auth import AuthContext

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

logger = logging.getLogger(__name__)

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
    semantic_top_k: int = 15
) -> Dict[str, Any]:
    """
    智能检索主入口
    """
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

    # 2. 路由策略：根据是否有 Filters 决定走哪条路
    search_result = {}
    
    # 🔍 调试日志：看看 Service 到底收到了什么
    logger.info(f"🛡️ [Service] 收到请求 -> Query: '{query_text}' | Filters: {column_filters}")

    if column_filters and len(column_filters) > 0:
        logger.info("🛤️ 命中策略: [结构化表格检索]")
        search_result = await _strategy_structured_table(
            index, query_text, auth, column_filters, llm_top_k
        )
    else:
        logger.info("🛤️ 命中策略: [通用语义检索]")
        search_result = await _strategy_general_semantic(
            index, query_text, auth, semantic_top_k
        )

    # 3. LLM 生成回答
    best_rows = search_result.get("best_rows", [])
    
    llm_answer = await _generate_summary(
        query_text=query_text if query_text else str(column_filters), # 如果query为空，用filters做上下文
        context_nodes=best_rows
    )

    return {
        "mode": search_result["mode"],
        "answer": llm_answer,
        "sources": best_rows,
        "all_rows": search_result["all_rows"]
    }

# # ==========================================
# # 2. LLM 生成模块
# # ==========================================
# async def _generate_summary(query_text: str, context_nodes: List[Dict]) -> str:
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

def _format_node_for_llm(node_data: dict) -> str:
    """
    将单条元数据字典格式化为 LLM 易读的字符串。
    过滤掉不必要的系统字段，节省 Token。
    """
    if not node_data:
        return ""
        
    lines = []
    # 定义不需要传给 LLM 的内部技术字段
    exclude_keys = [
        "source_id", "doc_id", "embedding", "window", 
        "original_text", "_node_content", "relationships"
    ]
    
    for k, v in node_data.items():
        # 1. 排除系统字段
        if k in exclude_keys:
            continue
        # 2. 排除空值
        if v is None:
            continue
        # 3. 排除空字符串
        if isinstance(v, str) and not v.strip():
            continue
            
        lines.append(f"{k}: {v}")
            
    return " | ".join(lines)


# ==========================================
# 2. LLM 生成模块 (优化版)
# ==========================================
async def _generate_summary(query_text: str, context_nodes: List[Dict]) -> str:
    if not context_nodes:
        return "抱歉，数据库中未找到符合条件的样本数据。"
    
    # 1. 精细化构建上下文 (Context Cleaning)
    context_str_list = []
    for idx, item in enumerate(context_nodes):
        meta = item.get("metadata", {})
        # 使用专门的格式化函数，提取关键信息，丢弃垃圾字段
        formatted_text = _format_node_for_llm(meta)
        context_str_list.append(f"【样本 {idx+1}】\n{formatted_text}")
    
    context_text = "\n\n".join(context_str_list)

    # 2. 专家级 Prompt (Prompt Engineering)
    # 告诉模型：怎么读数据、怎么组织语言
    system_prompt = (
        "你是一位资深的单细胞数据分析专家。\n"
        "你的任务是根据检索到的数据库记录，回答用户关于实验样本的问题。\n"
        "\n"
        "### 回答逻辑结构（必须遵守）：\n"
        "请将回答分为两个部分：\n"
        "\n"
        "#### 第一部分：核心结论与分析\n"
        "- **直接回答**：针对用户的问题进行总结性回答。不要只罗列数据，要进行分析。\n"
        "- **比如**：\n"
        "   - 如果用户问“两者有什么区别？”，请直接对比两者的活率、细胞类型分布差异。\n"
        "   - 如果用户问“有没有心脏数据？”，请回答“有，共X个样本，主要是解离和抽核两种方案”。\n"
        "- **总结趋势**：指出数据中的关键特征（如：解离样本的活率普遍高于抽核样本）。\n"
        "\n"
        "#### 第二部分：支持数据详情\n"
        "- 在结论之后，列出支撑上述结论的具体关键数据（即“证据”）。\n"
        "- 按样本或实验组进行分类展示，包含：平台、组织、处理方案、关键质控指标（活率、捕获数）及主要细胞类型。\n"
        "\n"
        "### 注意事项：\n"
        "1. **实事求是**：如果没有相关数据，直接说明“未找到相关数据”。\n"
        "2. **格式友好**：使用 Markdown 的列表、表格或加粗字体，确保排版清晰易读。\n"
        "3. **去伪存真**：忽略缺失值（如 '/' 或 'None'），只展示有意义的字段。\n"
    )
    user_prompt = (
        f"用户问题: {query_text}\n"
        f"\n"
        f"【数据库检索结果】:\n"
        f"{context_text}"
    )

    try:
        if not Settings.llm:
            return "LLM 未初始化"
            
        # 3. 调用 LLM (增加 max_tokens 防止断句)
        response = await Settings.llm.achat(
            messages=[
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, content=user_prompt)
            ],
            # 关键：给模型足够的输出空间，防止话说一半被截断
            max_tokens=2048, 
            temperature=0.3  # 降低随机性，保证事实准确
        )
        return str(response.message.content)
    except Exception as e:
        logger.error(f"LLM 生成失败: {e}")
        return "很抱歉，生成回答时遇到错误，请参考下方的原始数据列表。"

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
    # 1. 构建 Filter (保持不变)
    filters_list = [
        MetadataFilter(key="department", operator=FilterOperator.EQ, value=auth.department)
    ]
    for k, v in column_filters.items():
        filters_list.append(MetadataFilter(key=k, operator=FilterOperator.EQ, value=v))
    
    metadata_filters = MetadataFilters(filters=filters_list, condition="and")
    
    # 2. 准备检索
    # 第一次尝试：老老实实按用户说的词去搜
    search_text = query_text if query_text and query_text.strip() else ""
    
    # 设置较高的 top_k 以获取尽可能多的表格数据
    retriever = index.as_retriever(similarity_top_k=2000, filters=metadata_filters)
    
    # 3. 执行初次检索
    all_nodes = await retriever.aretrieve(search_text)
    
    logger.info(f"📊 [Table] 初次尝试 (Query='{search_text}') 命中: {len(all_nodes)} 条")

    # =====================================================
    # ✅ 【核心修复】增加降级重试逻辑
    # =====================================================
    # 如果没查到数据，但我们明明有强 Filter（比如指定了心脏、小鼠）
    # 这说明可能是 search_text (如"单细胞") 干扰了向量匹配。
    # 我们应该忽略文本，强制把符合 Filter 的数据全捞出来。
    if len(all_nodes) == 0 and column_filters:
        logger.warning(f"⚠️ [Table] 语义检索未命中，触发兜底策略：忽略关键词，仅按 Filter 检索...")
        
        # 传入空字符串或空格，告诉 Retriever 不要卡向量相似度，只要 Filter 匹配就行
        # (注意：具体取决于 VectorStore 实现，通常 " " 或 "" 有效，或者用特定通配符)
        all_nodes = await retriever.aretrieve(" ") 
        
        logger.info(f"📊 [Table] 兜底检索命中: {len(all_nodes)} 条")

    # 4. 挑选 best_rows (保持不变)
    best_nodes = []
    # 如果是触发了兜底逻辑（或者本来就是空搜），直接切片
    is_pure_filter = (len(search_text.strip()) == 0) or (len(all_nodes) > 0 and query_text == "单细胞") # 也可以在这里特判
    
    if not query_text or not query_text.strip():
         best_nodes = all_nodes[:llm_top_k]
    else:
        # 只有当确实查到了数据，且有具体查询词时，才做 Rerank
        # 如果是兜底查出来的，Rerank 可能也没太大必要，直接拿前 N 条即可，或者依然 Rerank 试试
        if all_nodes:
             best_nodes = await _perform_rerank(query_text, all_nodes[:30], llm_top_k)
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
    retriever = index.as_retriever(similarity_top_k=top_k*4, filters=filters)
    candidate_nodes = await retriever.aretrieve(query_text)
    
    logger.info(f"🔍 [Semantic] 初筛命中: {len(candidate_nodes)} 条 -> 准备 Rerank")

    final_nodes = await _perform_rerank(query_text, candidate_nodes, top_k)
    
    return {
        "mode": "semantic",
        "all_rows": [], # 语义模式通常不返回全量表，只返回由相关度排序的 best_rows
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

