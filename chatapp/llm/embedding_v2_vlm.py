import os
import asyncio
from http import HTTPStatus
from typing import Any, List, Optional, Union
import llm.key_url

import dashscope
from pydantic import PrivateAttr

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.schema import ImageType
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
import numpy as np




class CustomDashScopeEmbedding(MultiModalEmbedding):
    """
    自定义 DashScope 嵌入模型，同时支持文本和图像嵌入。
    
    文本嵌入：
    - 支持 text-embedding-v3 与 text-embedding-v4 模型
    - 可指定输出向量维度（2048、1536、1024、768、512、256、128或64）
    - 默认维度为 1024
    
    图像嵌入：
    - 支持 multimodal-embedding-v1 模型
    - 接受图像文件路径或图像数据
    """

    _model_name: str = PrivateAttr()
    _api_key: str = PrivateAttr()
    _base_url: str = PrivateAttr()
    _dimension: int = PrivateAttr()
    _output_type: str = PrivateAttr()
    _image_model_name: str = PrivateAttr()

    def __init__(
        self,
        model_name: str = "text-embedding-v4",
        api_key: str = None,
        base_url: str = None,
        dimension: int = None,
        output_type: str = "dense",
        image_model_name: str = "multimodal-embedding-v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name = model_name
        self._api_key = api_key if api_key else os.getenv("DASHSCOPE_API_KEY")
        self._base_url = base_url if base_url else os.getenv("DASHSCOPE_BASE_URL")
        self._dimension = dimension
        self._output_type = output_type
        self._image_model_name = image_model_name

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
    
    # --- 图像嵌入方法（同步）---
    def _get_image_embedding(self, img_file_path: ImageType) -> List[float]:
        """获取单个图像的嵌入向量"""
        return self._get_image_embeddings([img_file_path])[0]
    
    def _get_image_embeddings(self, img_file_paths: List[ImageType]) -> List[List[float]]:
        """批量获取图像嵌入向量"""
        if not img_file_paths:
            return []
            
        try:
            # 准备图像数据
            image_inputs = []
            for img_path in img_file_paths:
                # 处理不同类型的图像输入（路径字符串、二进制数据、PIL图像等）
                if isinstance(img_path, str):
                    # 如果是文件路径，使用文件路径
                    image_inputs.append({"image": img_path})
                elif hasattr(img_path, "read") and callable(img_path.read):
                    # 如果是文件对象，读取二进制数据
                    image_data = img_path.read()
                    image_inputs.append({"image": image_data})
                else:
                    # 其他类型直接传递
                    image_inputs.append({"image": img_path})
            
            # 调用千问的多模态嵌入模型API
            resp = dashscope.MultiModalEmbedding.call(
                model=self._image_model_name,
                input=image_inputs,
            )
            
            if resp.status_code == HTTPStatus.OK:
                embeddings = []
                for entry in resp.output['embeddings']:
                    embeddings.append(entry['embedding'])
                return embeddings
            else:
                raise Exception(
                    f"DashScope API 调用失败。Status Code: {resp.status_code}, Message: {resp.message}"
                )
        except Exception as e:
            raise RuntimeError(f"调用 DashScope 图像嵌入 API 时发生错误: {e}") from e
    
    # --- 图像嵌入方法（异步）---
    async def _aget_image_embedding(self, img_file_path: ImageType) -> List[float]:
        """异步获取单个图像的嵌入向量"""
        return (await self._aget_image_embeddings([img_file_path]))[0]
    
    async def _aget_image_embeddings(self, img_file_paths: List[ImageType]) -> List[List[float]]:
        """异步批量获取图像嵌入向量"""
        if not img_file_paths:
            return []
        
        loop = asyncio.get_event_loop()
        try:
            # 准备图像数据
            image_inputs = []
            for img_path in img_file_paths:
                if isinstance(img_path, str):
                    # 如果是文件路径，使用文件路径
                    image_inputs.append({"image": img_path})
                elif hasattr(img_path, "read") and callable(img_path.read):
                    # 如果是文件对象，读取二进制数据
                    if hasattr(img_path, "seek") and callable(img_path.seek):
                        img_path.seek(0)
                    image_data = await loop.run_in_executor(None, img_path.read)
                    image_inputs.append({"image": image_data})
                else:
                    # 其他类型直接传递
                    image_inputs.append({"image": img_path})
            
            # 异步调用千问的多模态嵌入模型API
            resp = await loop.run_in_executor(
                None,
                lambda: dashscope.MultiModalEmbedding.call(
                    model=self._image_model_name,
                    input=image_inputs
                )
            )
            
            if resp.status_code == HTTPStatus.OK:
                embeddings = []
                for entry in resp.output['embeddings']:
                    embeddings.append(entry['embedding'])
                return embeddings
            else:
                raise Exception(
                    f"DashScope API 调用失败。Status Code: {resp.status_code}, Message: {resp.message}"
                )
        except Exception as e:
            raise RuntimeError(f"调用 DashScope 图像嵌入 API 时发生错误: {e}") from e

    # --- 文本嵌入方法（同步）---
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding_batch([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding_batch([text])[0]

    # def _get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
    #     if not texts:
    #         return []

    #     call_params = {
    #         "model": self._model_name,
    #         "input": texts,
    #         "output_type": self._output_type,
    #     }
    #     if self._dimension:
    #         call_params["dimension"] = self._dimension

    #     try:
    #         resp = dashscope.TextEmbedding.call(**call_params)

    #         if resp.status_code == HTTPStatus.OK:
    #             embeddings = []
    #             for entry in resp.output['embeddings']:
    #                 if 'embedding' in entry:
    #                     embeddings.append(entry['embedding'])
    #                 elif 'sparse_embedding' in entry:
    #                     embeddings.append(entry['sparse_embedding'])
    #                 else:
    #                     raise ValueError("Neither 'embedding' nor 'sparse_embedding' found in response.")
    #             return embeddings
    #         else:
    #             raise Exception(
    #                 f"DashScope API 调用失败。Status Code: {resp.status_code}, Message: {resp.message}"
    #             )
    #     except Exception as e:
    #         raise RuntimeError(f"调用 DashScope 嵌入 API 时发生错误: {e}") from e

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

    # --- 文本嵌入方法（异步）---
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
