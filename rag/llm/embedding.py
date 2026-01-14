import os
import time
import logging
import asyncio
from typing import Any, Dict, List, Optional
from http import HTTPStatus

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
import dashscope

# 设置日志
logger = logging.getLogger(__name__)

class CustomDashScopeEmbedding(BaseEmbedding):
    """
    带日志和重试机制的自定义 DashScope Embedding
    """
    _model_name: str = PrivateAttr()
    _api_key: str = PrivateAttr()
    _dimension: int = PrivateAttr()
    _output_type: str = PrivateAttr()

    def __init__(
        self,
        model_name: str = "text-embedding-v4",
        api_key: str = None,
        base_url: str = None, # 虽不通过OpenAI协议，但保留参数兼容
        dimension: int = None,
        output_type: str = "dense",
        embed_batch_size: int = 10, # ✅ 显式默认值为 10
        **kwargs: Any,
    ) -> None:
        # 必须将 embed_batch_size 传给父类，否则 LlamaIndex 默认可能很大
        super().__init__(embed_batch_size=embed_batch_size, **kwargs)
        
        self._model_name = model_name
        self._api_key = api_key if api_key else os.getenv("DASHSCOPE_API_KEY")
        # dashscope SDK 读取 base_url 的方式可能不同，这里仅作兼容
        if base_url:
            dashscope.base_url = base_url
            
        self._dimension = dimension
        self._output_type = output_type

        if not self._api_key:
            raise ValueError("DashScope API Key is not provided.")
        dashscope.api_key = self._api_key

    @classmethod
    def class_name(cls) -> str:
        return "custom_dashscope_embedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding_batch([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding_batch([text])[0]

    def _get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """同步批量获取嵌入"""
        return self._call_dashscope_api(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        result = await self._aget_text_embedding_batch([query])
        return result[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        result = await self._aget_text_embedding_batch([text])
        return result[0]

    async def _aget_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """异步批量获取嵌入"""
        loop = asyncio.get_running_loop()
        # 使用 run_in_executor 包装同步的 API 调用
        return await loop.run_in_executor(None, self._call_dashscope_api, texts)

    def _call_dashscope_api(self, texts: List[str]) -> List[List[float]]:
        """
        统一的 API 调用逻辑，包含重试和日志
        """
        if not texts:
            return []

        # ✅ 日志：打印当前批次正在处理
        # logger.info(f"📡 DashScope Embedding 请求中... (本批次 {len(texts)} 条)")

        call_params = {
            "model": self._model_name,
            "input": texts,
            "output_type": self._output_type,
        }
        if self._dimension:
            call_params["dimension"] = self._dimension

        # ✅ 简单的重试逻辑 (最多重试 3 次)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = dashscope.TextEmbedding.call(**call_params)
                
                if resp.status_code == HTTPStatus.OK:
                    embeddings = []
                    for entry in resp.output['embeddings']:
                        embeddings.append(entry['embedding'])
                    return embeddings
                elif resp.status_code in [429, 500, 502, 503]:
                    # 如果是限流或服务器错误，等待后重试
                    wait_time = 2 * (attempt + 1)
                    logger.warning(f"⚠️ DashScope API 报错 {resp.status_code}，{wait_time}秒后重试 ({attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"DashScope Error: {resp.code} - {resp.message}")
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"❌ DashScope Embedding 最终失败: {e}")
                    raise e
                time.sleep(1)
                
        return []