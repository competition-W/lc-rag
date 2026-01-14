#!/mnt/chatchat/.venv/bin python3
# -*- coding: utf-8 -*-
from enum import Enum
#######DASHSCOPE
DASH_EMBEDDING_V3 = "text-embedding-v3" #########默认1024维!!经过测试是1536维

DASH_EMBEDDING_V4 = "text-embedding-v4"
###
DASH_MULTIMODAL="multimodal-embedding-one-image"##########说是1024维
class DashScopeMultiModalEmbeddingModels(str, Enum):
    """DashScope MultiModalEmbedding models."""
    MULTIMODAL_EMBEDDING_ONE_PEACE_V1 = "multimodal-embedding-v1"
    # MULTIMODAL_EMBEDDING_ONE_PEACE_V1 = "multimodal-embedding-one-peace-v1"
# =========================================================================
### 不同种类的模型可接受的参数不同
######文本生成模型
DASH_QWEN_TURBO = "qwen-turbo" ###简单任务，速度快、成本低
DASH_QWEN_PLUS = "qwen-plus" ###性能均衡，介于max与turbo之间
DASH_QWEN_MAX = "qwen-max" ###适合复杂任务，推理能力最强
######文本+图片==>文本，VL模型
DASH_QWEN_VL_MAX = "qwen-vl-max" ### 需要设置生成模型的输入图片参数
DASH_QWEN_VL_PLUS = "qwen-vl-plus"
##图片大小有限制哦


DASH_QWEN_OMNI = "qwen-omni-turbo" ###全模态模型，文本+图片+音频=>文本+音频
########doubao
## pip install -U 'volcengine-python-sdk[ark]'
DOUBAO_EMBEDDING_TEXT="doubao-embedding-text-240715" #########不支持修改，是2560维的嵌入向量
DOUBAO_EMBEDDING_VISION = "doubao-embedding-vision-241215"
## doubao deepseek-v3-250324
DOUBAO_DEEPSEEK_V3 = "deepseek-v3-250324"

import os
from typing import Any, Dict, List, Optional, Union
import asyncio
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
import dashscope
from http import HTTPStatus
import llm.key_url

from llama_index.core.schema import ImageType
import logging
logger = logging.getLogger(__name__)


def get_multimodal_embedding(
    model: str, input: list, api_key: Optional[str] = None, **kwargs: Any
) -> List[float]:
    """Call DashScope multimodal embedding.
       ref: https://help.aliyun.com/zh/dashscope/developer-reference/one-peace-multimodal-embedding-api-details.

    Args:
        model (str): The `DashScopeBatchTextEmbeddingModels`
        input (str): The input of the embedding, eg:
             [{'factor': 1, 'text': '你好'},
             {'factor': 2, 'audio': 'https://dashscope.oss-cn-beijing.aliyuncs.com/audios/cow.flac'},
             {'factor': 3, 'image': 'https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png'}]

    Raises:
        ImportError: Need install dashscope package.

    Returns:
        List[float]: Embedding result, if failed return empty list.
    """
    try:
        import dashscope
    except ImportError:
        raise ImportError("DashScope requires `pip install dashscope")
    response = dashscope.MultiModalEmbedding.call(
        model=model, input=input, api_key=api_key, kwargs=kwargs
    )
    if response.status_code == HTTPStatus.OK:
        return response.output["embedding"]
    else:
        logger.error("Calling MultiModalEmbedding failed, details: %s" % response)
        return []
    

class CustomDashScopeEmbedding(BaseEmbedding):
    """
    同样的，还有一个CustomLLM
    自定义 DashScope 嵌入模型，继承自 BaseEmbedding。
    封装 dashscope.TextEmbedding.call 的逻辑，使其适配 LlamaIndex 框架。
    用于用户指定输出向量维度，只适用于text-embedding-v3与text-embedding-v4模型。
    指定的值只能在2048（仅适用于text-embedding-v4）、1536（仅适用于text-embedding-v4）
    1024、768、512、256、128或64八个值之间选取，默认值为1024。
    最终实现了对向量数据库dim参数的支持，否则只能使用模型的默认输出
    另外，DashScopeParser这个商用方法
    """

    _model_name: str = PrivateAttr()
    _api_key: str = PrivateAttr()
    _base_url: str = PrivateAttr()
    _dimension: int = PrivateAttr()
    _output_type: str = PrivateAttr()

    def __init__(
        self,
        model_name: str = "text-embedding-v4",
        api_key: str = None,
        base_url: str = None,
        dimension: int = None,
        output_type: str = "dense",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model_name
        self._api_key = api_key if api_key else os.getenv("DASHSCOPE_API_KEY")
        self._base_url = base_url if base_url else os.getenv("DASHSCOPE_BASE_URL")
        self._dimension = dimension
        self._output_type = output_type

        if not self._api_key:
            raise ValueError(
                "DashScope API Key is not provided. "
                "Please set it via api_key argument or DASHSCOPE_API_KEY environment variable."
            )

        dashscope.api_key = self._api_key
        if self._base_url:
            dashscope.base_url = self._base_url

    @classmethod
    def class_name(cls) -> str:
        return "custom_dashscope_embedding"

    # --- 同步方法 ---
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding_batch([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding_batch([text])[0]

    def _get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        call_params = {
            "model": self._model_name,
            "input": texts,
            "output_type": self._output_type,
        }
        if self._dimension:
            call_params["dimension"] = self._dimension

        try:
            resp = dashscope.TextEmbedding.call(**call_params)

            if resp.status_code == HTTPStatus.OK:
                embeddings = []
                for entry in resp.output['embeddings']:
                    embeddings.append(entry['embedding'])
                return embeddings
            else:
                # If DashScope API returns an error status, raise an exception
                raise Exception(
                    f"DashScope API 调用失败。Status Code: {resp.status_code}, Message: {resp.message}"
                )
        except Exception as e:
            # Re-raise the exception to indicate a failure in getting embeddings
            raise RuntimeError(f"调用 DashScope 嵌入 API 时发生错误: {e}") from e

    # --- 异步方法 ---
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return (await self._aget_text_embedding_batch([query]))[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return (await self._aget_text_embedding_batch([text]))[0]

    async def _aget_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        call_params = {
            "model": self._model_name,
            "input": texts,
            "output_type": self._output_type,
        }
        if self._dimension:
            call_params["dimension"] = self._dimension

        loop = asyncio.get_event_loop()
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: dashscope.TextEmbedding.call(**call_params)
            )

            if resp.status_code == HTTPStatus.OK:
                embeddings = []
                for entry in resp.output['embeddings']:
                    embeddings.append(entry['embedding'])
                return embeddings
            else:
                # If DashScope API returns an error status, raise an exception
                raise Exception(
                    f"DashScope API 调用失败。Status Code: {resp.status_code}, Message: {resp.message}"
                )
        except Exception as e:
            # Re-raise the exception to indicate a failure in getting embeddings
            raise RuntimeError(f"调用 DashScope 嵌入 API 时发生错误: {e}") from e
        
