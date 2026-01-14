from __future__ import annotations

DASH_QWEN_PLUS = "qwen-plus"
DASH_QWEN_MAX = ""

DOUBAO_DEEPSEEK_V3 = "doubao deepseek-v3-250324"

import llm.key_url

"""
DashScope ↔︎ LlamaIndex 自定义 LLM
依赖: pip install llama-index dashscope pydantic>=2
环境: 需设置 DASHSCOPE_API_KEY
"""

import os, asyncio
from http import HTTPStatus
from typing import Any, ClassVar, Dict, Generator, AsyncGenerator, Sequence

from pydantic import PrivateAttr, Field
from dashscope import Generation
from dashscope.aigc.generation import AioGeneration

from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse, CompletionResponseGen, CompletionResponseAsyncGen,
    ChatMessage, ChatResponse, ChatResponseGen, ChatResponseAsyncGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback

# ---------------------------------------------------------------------
class DashScopeLLM(CustomLLM):
    """DashScope (百炼) 模型的 LlamaIndex 适配器"""

    # ---------- 静态/类常量 ----------
    context_window: ClassVar[int] = 4096*2
    num_output: ClassVar[int]     = 512*2

    # ---------- Pydantic 模型字段 ----------
    model_name: str = Field(default="qwen-plus", description="DashScope 模型名称")
    api_key: str | None = Field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY"))
    temperature: float | None = None
    top_p: float | None = None

    # ---------- 其余参数走私有属性 ----------
    _generation_kwargs: Dict[str, Any] = PrivateAttr(default_factory=dict)

    # ---------- 自定义 __init__ ----------
    def __init__(self, **data: Any):
        # 从 kwargs 把额外 Generation.call 的参数拎出来
        extra = {k: data.pop(k) for k in list(data.keys())
                 if k not in self.__class__.model_fields}        # self.model_fields
        super().__init__(**data)                       # 让 Pydantic 处理字段
        self._generation_kwargs.update(extra)          # 保存额外 Generation 参数

    # ---------- 公共辅助 ----------
    def _build_params(self, **local_kwargs: Any) -> Dict[str, Any]:
        """组装 DashScope Generation/AioGeneration 共有参数"""
        params: Dict[str, Any] = {
            "api_key": self.api_key,
            "model":   self.model_name,
            "max_tokens": self.num_output,
            **self._generation_kwargs,
            **local_kwargs,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        return params

    # ---------- LLMMetadata ----------
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
            is_chat_model=True,
        )

    # -------------------- 同步 · Completion --------------------
    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
        resp = Generation.call(
            prompt=prompt,
            result_format="message",
            stream=False,
            **self._build_params(**kwargs),
        )
        text = (
            resp.output.choices[0].message.content
            if resp.status_code == HTTPStatus.OK
            else ""
        )
        return CompletionResponse(text=text, raw=resp)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs
    ) -> CompletionResponseGen:
        resp_iter = Generation.call(
            prompt=prompt,
            result_format="message",
            stream=True,
            incremental_output=True,
            **self._build_params(**kwargs),
        )

        def gen() -> Generator[CompletionResponse, None, None]:
            text = ""
            for chunk in resp_iter:
                delta = (
                    chunk.output.choices[0].message.content
                    if chunk.status_code == HTTPStatus.OK
                    else ""
                )
                text += delta
                yield CompletionResponse(text=text, delta=delta, raw=chunk)

        return gen()

    # -------------------- 异步 · Completion --------------------
    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs
    ) -> CompletionResponse:
        resp = await AioGeneration.call(
            prompt=prompt,
            result_format="message",
            stream=False,
            **self._build_params(**kwargs),
        )
        text = (
            resp.output.choices[0].message.content
            if resp.status_code == HTTPStatus.OK
            else ""
        )
        return CompletionResponse(text=text, raw=resp)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs
    ) -> CompletionResponseAsyncGen:
        resp_iter = await AioGeneration.call(
            prompt=prompt,
            result_format="message",
            stream=True,
            incremental_output=True,
            **self._build_params(**kwargs),
        )

        async def agen() -> AsyncGenerator[CompletionResponse, None]:
            text = ""
            async for chunk in resp_iter:
                delta = (
                    chunk.output.choices[0].message.content
                    if chunk.status_code == HTTPStatus.OK
                    else ""
                )
                text += delta
                yield CompletionResponse(text=text, delta=delta, raw=chunk)

        return agen()

    # -------------------- 同步 · Chat --------------------
    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponse:
        resp = Generation.call(
            messages=[m.model_dump() for m in messages],  # 避免 .dict() 弃用
            result_format="message",
            stream=False,
            **self._build_params(**kwargs),
        )
        answer = (
            resp.output.choices[0].message.content
            if resp.status_code == HTTPStatus.OK
            else ""
        )
        return ChatResponse(message=ChatMessage(role="assistant", content=answer), raw=resp)

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponseGen:
        resp_iter = Generation.call(
            messages=[m.model_dump() for m in messages],
            result_format="message",
            stream=True,
            incremental_output=True,
            **self._build_params(**kwargs),
        )

        def gen() -> Generator[ChatResponse, None, None]:
            content = ""
            for chunk in resp_iter:
                delta = (
                    chunk.output.choices[0].message.content
                    if chunk.status_code == HTTPStatus.OK
                    else ""
                )
                content += delta
                yield ChatResponse(
                    message=ChatMessage(role="assistant", content=content),
                    delta=delta,
                    raw=chunk,
                )

        return gen()

    # ---------- class_name 供序列化 ----------
    @classmethod
    def class_name(cls) -> str:  # noqa: D401
        return "dashscope_custom_llm"

# ---------------------------------------------------------------------
# 若需简单自测
if __name__ == "__main__":
    llm = DashScopeLLM(model_name="qwen-turbo", temperature=1.8) ## 默认0.7; 0.8 top_p
    print("请为Fate 今晚留下来写一首诗")
    print(llm.complete("请为'Fate 今晚留下来'写首一个字的诗。人物:saber,远坂凛，间桐樱，archer，卫宫士郎，伊莉雅。情节：圣杯战争。结果：正义的伙伴。").text)
