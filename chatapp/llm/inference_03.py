import llm.key_url  # 若内部有引用可保留，没有可删除

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
    context_window: ClassVar[int] = 4096 * 2
    num_output:   ClassVar[int]   = 512  * 4

    # ---------- Pydantic 字段 ----------
    model_name:  str       = Field(default="qwen-plus", description="DashScope 模型名称")
    api_key:     str | None = Field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY"))
    temperature: float | None = None
    top_p:       float | None = None

    # ---------- 私有属性 ----------
    _generation_kwargs: Dict[str, Any] = PrivateAttr(default_factory=dict)

    # ---------- 自定义 __init__ ----------
    def __init__(self, **data: Any):
        # 把传入但不在 model_fields 中的参数收集起来
        extra = {k: data.pop(k) for k in list(data.keys())
                 if k not in self.__class__.model_fields}
        super().__init__(**data)              # 让 Pydantic 处理字段
        self._generation_kwargs.update(extra) # 存储额外 Generation 参数

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

    # ---------- Prompt ➜ Messages ----------
    def _to_messages(self, prompt: str, *, assume_formatted: bool = False) -> list[dict]:
        """把 LlamaIndex 传来的长 prompt 转成 DashScope 聊天 messages"""
        if assume_formatted:
            # 调用方表示已是最终格式，这里只包一层 user
            return [{"role": "user", "content": prompt}]
        # 简单启发式：若包含 "Query:"，拆 system/user
        if "Query:" in prompt:
            sys_part, usr_part = prompt.split("Query:", 1)
            return [
                {"role": "system", "content": sys_part.strip()},
                {"role": "user",   "content": "Query:" + usr_part.strip()},
            ]
        # 默认全部作为 user
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def _extract_delta(chunk) -> str:
        """安全提取流式 chunk 的增量内容"""
        if (chunk is None or
            chunk.status_code != HTTPStatus.OK or
            not getattr(chunk, "output", None) or
            not getattr(chunk.output, "choices", None) or
            not chunk.output.choices or
            not getattr(chunk.output.choices[0], "message", None)):
            return ""
        return chunk.output.choices[0].message.content

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
        messages = self._to_messages(prompt, assume_formatted=formatted)
        resp = Generation.call(
            messages=messages,
            result_format="message",
            stream=False,
            **self._build_params(**kwargs),
        )
        text = (
            resp.output.choices[0].message.content
            if resp.status_code == HTTPStatus.OK else ""
        )
        return CompletionResponse(text=text, raw=resp)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs
    ) -> CompletionResponseGen:
        messages = self._to_messages(prompt, assume_formatted=formatted)
        resp_iter = Generation.call(
            messages=messages,
            result_format="message",
            stream=True,
            incremental_output=True,
            **self._build_params(**kwargs),
        )

        def gen() -> Generator[CompletionResponse, None, None]:
            text = ""
            for chunk in resp_iter:
                delta = self._extract_delta(chunk)
                text += delta
                yield CompletionResponse(text=text, delta=delta, raw=chunk)

        return gen()

    # -------------------- 异步 · Completion --------------------
    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs
    ) -> CompletionResponse:
        messages = self._to_messages(prompt, assume_formatted=formatted)
        resp = await AioGeneration.call(
            messages=messages,
            result_format="message",
            stream=False,
            **self._build_params(**kwargs),
        )
        text = (
            resp.output.choices[0].message.content
            if resp.status_code == HTTPStatus.OK else ""
        )
        return CompletionResponse(text=text, raw=resp)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs
    ) -> CompletionResponseAsyncGen:
        messages = self._to_messages(prompt, assume_formatted=formatted)
        resp_iter = await AioGeneration.call(
            messages=messages,
            result_format="message",
            stream=True,
            incremental_output=True,
            **self._build_params(**kwargs),
        )

        async def agen() -> AsyncGenerator[CompletionResponse, None]:
            text = ""
            async for chunk in resp_iter:
                delta = self._extract_delta(chunk)
                text += delta
                yield CompletionResponse(text=text, delta=delta, raw=chunk)

        return agen()

    # -------------------- 同步 · Chat --------------------
    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponse:
        resp = Generation.call(
            messages=[m.model_dump() for m in messages],
            result_format="message",
            stream=False,
            **self._build_params(**kwargs),
        )
        answer = (
            resp.output.choices[0].message.content
            if resp.status_code == HTTPStatus.OK else ""
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
                delta = self._extract_delta(chunk)
                content += delta
                yield ChatResponse(
                    message=ChatMessage(role="assistant", content=content),
                    delta=delta,
                    raw=chunk,
                )

        return gen()

    # -------------------- 异步 · Chat --------------------
    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs
    ) -> ChatResponseAsyncGen:
        resp_iter = await AioGeneration.call(
            messages=[m.model_dump() for m in messages],
            result_format="message",
            stream=True,
            incremental_output=True,
            **self._build_params(**kwargs),
        )

        async def agen() -> AsyncGenerator[ChatResponse, None]:
            content = ""
            async for chunk in resp_iter:
                delta = self._extract_delta(chunk)
                content += delta
                yield ChatResponse(
                    message=ChatMessage(role="assistant", content=content),
                    delta=delta,
                    raw=chunk,
                )

        return agen()

    # ---------- class_name 供序列化 ----------
    @classmethod
    def class_name(cls) -> str:  # noqa: D401
        return "dashscope_custom_llm"
