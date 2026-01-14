# # 文件路径: /mnt/omicshub/rag/api/query.py
# from fastapi import APIRouter, Depends
# from pydantic import BaseModel, Field
# import logging

# # ✅ 1. 引入标准响应工具
# from utils.response import success, error

# # ✅ 2. 引入鉴权模块
# from utils.auth import get_auth_context, AuthContext

# # ✅ 3. 引入业务服务
# from services.query_parser import parse_user_query
# from services.query_service import unified_query_service 

# logger = logging.getLogger(__name__)

# # 定义路由
# # 注意：在 main.py 中挂载时建议使用 prefix="/query"，这里路径设为空字符串 ""
# router = APIRouter(tags=["智能检索"])

# class SmartSearchRequest(BaseModel):
#     text: str = Field(..., description="用户的自然语言输入", example="帮我找小鼠心脏的数据")
#     use_llm: bool = Field(True, description="是否需要LLM生成回答")

# @router.post("")
# async def smart_search(
#     request: SmartSearchRequest,
#     auth: AuthContext = Depends(get_auth_context)
# ):
#     """
#     RAG 智能检索接口
#     """
#     try:
#         # ================= Step 1: 意图理解 (Parser) =================
#         # 调用 services.query_parser
#         real_query, extracted_filters = await parse_user_query(request.text, auth)
        
#         logger.info(f"🧠 [API] 意图识别: Query='{real_query}' | Filters={extracted_filters}")

#         # ================= Step 2: 统一检索 (Service) =================
#         # 调用 services.query_service
#         # 参数对应 unified_query_service 的签名
#         result_set = await unified_query_service(
#             auth=auth,
#             query_text=real_query,
#             column_filters=extracted_filters,
#             llm_top_k=5,       # 传给 LLM 的上下文条数
#             semantic_top_k=10  # 语义检索初筛条数
#         )
        
#         # ================= Step 3: 数据清洗与组装 =================
#         # 根据 query_service.py 的返回结构获取数据
#         # keys: "mode", "answer", "sources", "all_rows"
#         final_answer = result_set.get("answer", "")
#         sources = result_set.get("sources", [])
#         all_rows = result_set.get("all_rows", [])
#         mode = result_set.get("mode", "unknown")

#         # 兜底逻辑：如果 all_rows 为空但 sources 有值（通常发生在纯语义模式），复用 sources
#         if not all_rows and sources:
#             all_rows = sources

#         # 处理 use_llm 开关：如果前端不需要回答，强行置空
#         if not request.use_llm:
#             final_answer = ""

#         # 构造 data 字典
#         response_data = {
#             "answer": final_answer,
#             "mode": mode,
#             "intent": {
#                 "original_text": request.text,
#                 "extracted_filters": extracted_filters,
#                 "search_term": real_query
#             },
#             "sources": sources,     # 对应 query_service 中的 best_rows (Source Card)
#             "all_rows": all_rows,   # 对应 query_service 中的 all_rows (Table)
#             "total_count": len(all_rows)
#         }

#         # ================= Step 4: 返回标准响应 =================
#         # 调用 utils.response.success(data=...)
#         return success(data=response_data)

#     except Exception as e:
#         logger.exception(f"❌ Search API Error: {e}")
#         # 调用 utils.response.error(message=..., code=...)
#         return error(message=f"检索服务异常: {str(e)}", code=500)


# 文件路径: /mnt/omicshub/rag/api/query.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
import logging

# ✅ 1. 引入标准响应工具
from utils.response import success, error

# ✅ 2. 引入鉴权模块
from utils.auth import get_auth_context, AuthContext

# ✅ 3. 引入业务服务
from services.query_parser import parse_user_query
from services.query_service import unified_query_service 

logger = logging.getLogger(__name__)

# 定义路由
# 注意：在 main.py 中挂载时建议使用 prefix="/query"，这里路径设为空字符串 ""
router = APIRouter(tags=["智能检索"])

class SmartSearchRequest(BaseModel):
    text: str = Field(..., description="用户的自然语言输入", example="帮我找小鼠心脏的数据")
    use_llm: bool = Field(True, description="是否需要LLM生成回答")

@router.post("")
async def smart_search(
    request: SmartSearchRequest,
    auth: AuthContext = Depends(get_auth_context)
):
    """
    RAG 智能检索接口
    """
    try:
        # ================= Step 1: 意图理解 (Parser) =================
        # 调用 services.query_parser
        # parse_user_query 返回 (搜索关键词, 过滤条件字典)
        real_query, extracted_filters = await parse_user_query(request.text, auth)
        
        logger.info(f"🧠 [API] 初步意图: 提取词='{real_query}' | Filters={extracted_filters}")

        # =================================================================
        # ✅ 【核心修改】空搜索词兜底策略
        # =================================================================
        # 如果意图识别把所有信息都转成了 Filter，导致 real_query 为空字符串，
        # 此时 Milvus 向量检索会报错 (Embedding 不能为 None/Empty)。
        # 解决方案：回退使用用户的“原始提问”作为向量检索的锚点。
        if not real_query or not real_query.strip():
            logger.info(f"ℹ️ [API] 提取的搜索词为空，回退使用原始提问进行向量检索: '{request.text}'")
            real_query = request.text

        logger.info(f"🚀 [API] 最终执行: Query='{real_query}' | Filters={extracted_filters}")

        # ================= Step 2: 统一检索 (Service) =================
        # 调用 services.query_service
        # 此时传入的 query_text 保证是有值的 (要么是提取的关键词，要么是原始提问)
        result_set = await unified_query_service(
            auth=auth,
            query_text=real_query,
            column_filters=extracted_filters,
            llm_top_k=5,       # 传给 LLM 的上下文条数
            semantic_top_k=10  # 语义检索初筛条数
        )
        
        # ================= Step 3: 数据清洗与组装 =================
        # 根据 query_service.py 的返回结构获取数据
        final_answer = result_set.get("answer", "")
        sources = result_set.get("sources", [])
        all_rows = result_set.get("all_rows", [])
        mode = result_set.get("mode", "unknown")

        # 兜底逻辑：如果 all_rows 为空但 sources 有值（通常发生在纯语义模式），复用 sources
        if not all_rows and sources:
            all_rows = sources

        # 处理 use_llm 开关：如果前端不需要回答，强行置空
        if not request.use_llm:
            final_answer = ""

        # 构造 data 字典
        response_data = {
            "answer": final_answer,
            "mode": mode,
            "intent": {
                "original_text": request.text,
                "extracted_filters": extracted_filters,
                "search_term": real_query # 这里返回最终实际使用的搜索词
            },
            "sources": sources,     # 对应 query_service 中的 best_rows (Source Card)
            "all_rows": all_rows,   # 对应 query_service 中的 all_rows (Table)
            "total_count": len(all_rows)
        }

        # ================= Step 4: 返回标准响应 =================
        return success(data=response_data)

    except Exception as e:
        logger.exception(f"❌ Search API Error: {e}")
        return error(message=f"检索服务异常: {str(e)}", code=500)