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
    num_output: ClassVar[int]     = 512*4

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

    # 添加这个辅助方法来处理格式化问题
    def _format_prompt_to_messages(self, prompt: str) -> list:
        """将文本prompt转换为DashScope期望的消息格式"""
        return [{"role": "user", "content": prompt}]
    
    # 修改complete方法
    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
        # 处理formatted参数
        if not formatted:
            messages = self._format_prompt_to_messages(prompt)
            resp = Generation.call(
                messages=messages,
                result_format="message",
                stream=False,
                **self._build_params(**kwargs),
            )
        else:
            # 如果已格式化，假设prompt是原始文本
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
        # 处理formatted参数
        if not formatted:
            messages = self._format_prompt_to_messages(prompt)
            resp_iter = Generation.call(
                messages=messages,
                result_format="message",
                stream=True,
                incremental_output=True,
                **self._build_params(**kwargs),
            )
        else:
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
                # 添加更健壮的错误处理
                if (chunk is None or 
                    chunk.status_code != HTTPStatus.OK or
                    getattr(chunk, 'output', None) is None or
                    getattr(chunk.output, 'choices', None) is None or
                    len(chunk.output.choices) == 0 or
                    getattr(chunk.output.choices[0], 'message', None) is None):
                    # 处理错误情况
                    delta = ""
                else:
                    delta = chunk.output.choices[0].message.content
                    
                text += delta
                yield CompletionResponse(text=text, delta=delta, raw=chunk)

        return gen()
    # # 同样修改stream_complete方法
    # @llm_completion_callback()
    # def stream_complete(
    #     self, prompt: str, formatted: bool = False, **kwargs
    # ) -> CompletionResponseGen:
    #     # 处理formatted参数
    #     if not formatted:
    #         messages = self._format_prompt_to_messages(prompt)
    #         resp_iter = Generation.call(
    #             messages=messages,
    #             result_format="message",
    #             stream=True,
    #             incremental_output=True,
    #             **self._build_params(**kwargs),
    #         )
    #     else:
    #         resp_iter = Generation.call(
    #             prompt=prompt,
    #             result_format="message",
    #             stream=True,
    #             incremental_output=True,
    #             **self._build_params(**kwargs),
    #         )
    #     def gen() -> Generator[CompletionResponse, None, None]:
    #         text = ""
    #         for chunk in resp_iter:
    #             delta = (
    #                 chunk.output.choices[0].message.content
    #                 if chunk.status_code == HTTPStatus.OK
    #                 else ""
    #             )
    #             text += delta
    #             yield CompletionResponse(text=text, delta=delta, raw=chunk)

    #     return gen()
    
    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs
    ) -> CompletionResponse:
        if not formatted:
            messages = self._format_prompt_to_messages(prompt)
            resp = await AioGeneration.call(
                messages=messages,
                result_format="message",
                stream=False,
                **self._build_params(**kwargs),
            )
        else:
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
    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponseAsyncGen:
        """异步流式补全方法"""
        print(f"发送提示到DashScope API (异步流式): {prompt}")
        
        resp_iter = await AioGeneration.call(
            prompt=prompt,
            stream=True,
            incremental_output=True,
            **self._build_params(**kwargs),
        )

        async def agen():
            text = ""
            
            async for chunk in resp_iter:
                # 增强错误处理
                if (chunk is None or 
                    chunk.status_code != HTTPStatus.OK or
                    getattr(chunk, 'output', None) is None or
                    getattr(chunk.output, 'text', None) is None):
                    delta = ""
                else:
                    delta = chunk.output.text
                    
                text += delta
                yield CompletionResponse(text=text, delta=delta, raw=chunk)

        return agen()

    # # 修改astream_complete方法
    # @llm_completion_callback()
    # async def astream_complete(
    #     self, prompt: str, formatted: bool = False, **kwargs
    # ) -> CompletionResponseAsyncGen:
    #     if not formatted:
    #         messages = self._format_prompt_to_messages(prompt)
    #         resp_iter = await AioGeneration.call(
    #             messages=messages,
    #             result_format="message",
    #             stream=True,
    #             incremental_output=True,
    #             **self._build_params(**kwargs),
    #         )
    #     else:
    #         resp_iter = await AioGeneration.call(
    #             prompt=prompt,
    #             result_format="message",
    #             stream=True,
    #             incremental_output=True,
    #             **self._build_params(**kwargs),
    #         )
    #     async def agen() -> AsyncGenerator[CompletionResponse, None]:
    #         text = ""
    #         async for chunk in resp_iter:
    #             delta = (
    #                 chunk.output.choices[0].message.content
    #                 if chunk.status_code == HTTPStatus.OK
    #                 else ""
    #             )
    #             text += delta
    #             yield CompletionResponse(text=text, delta=delta, raw=chunk)

    #     return agen()
    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], formatted: bool = False, **kwargs) -> ChatResponse:
        """同步聊天方法"""
        # 正确转换消息格式
        dashscope_messages = []
        for m in messages:
            dashscope_messages.append({
                "role": m.role.value if hasattr(m.role, 'value') else m.role,
                "content": m.content
            })
        
        print(f"发送消息到DashScope API: {dashscope_messages}")
        
        resp = Generation.call(
            messages=dashscope_messages,
            result_format="message",
            stream=False,
            **self._build_params(**kwargs),
        )
        
        # 其余代码不变...
        # 打印响应状态和详情
        print(f"API响应状态: {resp.status_code}")
        print(f"API响应详情: {resp}")
        
        # 增强错误处理
        if resp.status_code != HTTPStatus.OK:
            print(f"错误: {getattr(resp, 'message', '未知错误')}")
            return ChatResponse(message=ChatMessage(role="assistant", content="API调用失败"), raw=resp)
        
        # 检查响应结构
        if not hasattr(resp, 'output') or not hasattr(resp.output, 'choices') or len(resp.output.choices) == 0:
            print("响应结构不完整")
            return ChatResponse(message=ChatMessage(role="assistant", content="响应结构不完整"), raw=resp)
        
        # 获取内容
        answer = resp.output.choices[0].message.content
        print(f"获取到的内容: {answer}")
        
        return ChatResponse(message=ChatMessage(role="assistant", content=answer), raw=resp)

    # @llm_chat_callback()
    # def chat(self, messages: Sequence[ChatMessage], formatted: bool = False, **kwargs) -> ChatResponse:
    #     """同步聊天方法"""
    #     # 打印调试信息
    #     print(f"发送消息到DashScope API: {[m.model_dump() for m in messages]}")
        
    #     resp = Generation.call(
    #         messages=[m.model_dump() for m in messages],
    #         result_format="message",
    #         stream=False,
    #         **self._build_params(**kwargs),
    #     )
        
    #     # 打印响应状态和详情
    #     print(f"API响应状态: {resp.status_code}")
    #     print(f"API响应详情: {resp}")
        
    #     # 增强错误处理
    #     if resp.status_code != HTTPStatus.OK:
    #         print(f"错误: {getattr(resp, 'message', '未知错误')}")
    #         return ChatResponse(message=ChatMessage(role="assistant", content="API调用失败"), raw=resp)
        
    #     # 检查响应结构
    #     if not hasattr(resp, 'output') or not hasattr(resp.output, 'choices') or len(resp.output.choices) == 0:
    #         print("响应结构不完整")
    #         return ChatResponse(message=ChatMessage(role="assistant", content="响应结构不完整"), raw=resp)
        
    #     # 获取内容
    #     answer = resp.output.choices[0].message.content
    #     print(f"获取到的内容: {answer}")
        
    #     return ChatResponse(message=ChatMessage(role="assistant", content=answer), raw=resp)

    # # 修改chat方法
    # @llm_chat_callback()
    # def chat(self, messages: Sequence[ChatMessage], formatted: bool = False, **kwargs) -> ChatResponse:
    #     # chat方法始终使用messages格式，但可能需要处理formatted参数
    #     # formatted在chat中通常影响较小，但保持参数一致性很重要
    #     resp = Generation.call(
    #         messages=[m.model_dump() for m in messages],
    #         result_format="message",
    #         stream=False,
    #         **self._build_params(**kwargs),
    #     )
    #     answer = (
    #         resp.output.choices[0].message.content
    #         if resp.status_code == HTTPStatus.OK
    #         else ""
    #     )
    #     return ChatResponse(message=ChatMessage(role="assistant", content=answer), raw=resp)

    # # 修改stream_chat方法
    # @llm_chat_callback()
    # def stream_chat(self, messages: Sequence[ChatMessage], formatted: bool = False, **kwargs) -> ChatResponseGen:
    #     # 保持参数一致性
    #     resp_iter = Generation.call(
    #         messages=[m.model_dump() for m in messages],
    #         result_format="message",
    #         stream=True,
    #         incremental_output=True,
    #         **self._build_params(**kwargs),
    #     )
    
    #     # 保留现有的生成器实现
    #     def gen() -> Generator[ChatResponse, None, None]:
    #         content = ""
    #         for chunk in resp_iter:
    #             delta = (
    #                 chunk.output.choices[0].message.content
    #                 if chunk.status_code == HTTPStatus.OK
    #                 else ""
    #             )
    #             content += delta
    #             yield ChatResponse(
    #                 message=ChatMessage(role="assistant", content=content),
    #                 delta=delta,
    #                 raw=chunk,
    #             )

    #     return gen()

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], formatted: bool = False, **kwargs) -> ChatResponseGen:
        """流式聊天方法"""
        # 正确转换消息格式
        dashscope_messages = []
        for m in messages:
            dashscope_messages.append({
                "role": m.role.value if hasattr(m.role, 'value') else m.role,
                "content": m.content
            })
        
        print(f"发送消息到DashScope API (流式): {dashscope_messages}")
        
        resp_iter = Generation.call(
            messages=dashscope_messages,
            result_format="message",
            stream=True,
            incremental_output=True,
            **self._build_params(**kwargs),
        )

        def gen() -> Generator[ChatResponse, None, None]:
            content = ""
            chunk_count = 0
            
            for chunk in resp_iter:
                chunk_count += 1
                
                # 增强错误处理
                if (chunk is None or 
                    chunk.status_code != HTTPStatus.OK or
                    getattr(chunk, 'output', None) is None or
                    getattr(chunk.output, 'choices', None) is None or
                    len(chunk.output.choices) == 0 or
                    getattr(chunk.output.choices[0], 'message', None) is None):
                    delta = ""
                else:
                    delta = chunk.output.choices[0].message.content
                    
                content += delta
                yield ChatResponse(
                    message=ChatMessage(role="assistant", content=content),
                    delta=delta,
                    raw=chunk,
                )

        return gen()

    # @llm_chat_callback()
    # def stream_chat(self, messages: Sequence[ChatMessage], formatted: bool = False, **kwargs) -> ChatResponseGen:
    #     resp_iter = Generation.call(
    #         messages=[m.model_dump() for m in messages],
    #         result_format="message",
    #         stream=True,
    #         incremental_output=True,
    #         **self._build_params(**kwargs),
    #     )

    #     def gen() -> Generator[ChatResponse, None, None]:
    #         content = ""
    #         for chunk in resp_iter:
    #             # 添加健壮的错误处理
    #             if (chunk is None or 
    #                 chunk.status_code != HTTPStatus.OK or
    #                 getattr(chunk, 'output', None) is None or
    #                 getattr(chunk.output, 'choices', None) is None or
    #                 len(chunk.output.choices) == 0 or
    #                 getattr(chunk.output.choices[0], 'message', None) is None):
    #                 delta = ""
    #             else:
    #                 delta = chunk.output.choices[0].message.content
                    
    #             content += delta
    #             yield ChatResponse(
    #                 message=ChatMessage(role="assistant", content=content),
    #                 delta=delta,
    #                 raw=chunk,
    #             )

    #     return gen()

    # @llm_completion_callback()
    # async def astream_complete(
    #     self, prompt: str, formatted: bool = False, **kwargs
    # ) -> CompletionResponseAsyncGen:
    #     if not formatted:
    #         messages = self._format_prompt_to_messages(prompt)
    #         resp_iter = await AioGeneration.call(
    #             messages=messages,
    #             result_format="message",
    #             stream=True,
    #             incremental_output=True,
    #             **self._build_params(**kwargs),
    #         )
    #     else:
    #         resp_iter = await AioGeneration.call(
    #             prompt=prompt,
    #             result_format="message",
    #             stream=True,
    #             incremental_output=True,
    #             **self._build_params(**kwargs),
    #         )

    #     async def agen() -> AsyncGenerator[CompletionResponse, None]:
    #         text = ""
    #         async for chunk in resp_iter:
    #             # 添加健壮的错误处理
    #             if (chunk is None or 
    #                 chunk.status_code != HTTPStatus.OK or
    #                 getattr(chunk, 'output', None) is None or
    #                 getattr(chunk.output, 'choices', None) is None or
    #                 len(chunk.output.choices) == 0 or
    #                 getattr(chunk.output.choices[0], 'message', None) is None):
    #                 delta = ""
    #             else:
    #                 delta = chunk.output.choices[0].message.content
                    
    #             text += delta
    #             yield CompletionResponse(text=text, delta=delta, raw=chunk)

    #     return agen()

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], formatted: bool = False, **kwargs) -> ChatResponse:
        """异步聊天方法"""
        # 正确转换消息格式
        dashscope_messages = []
        for m in messages:
            dashscope_messages.append({
                "role": m.role.value if hasattr(m.role, 'value') else m.role,
                "content": m.content
            })
        
        print(f"发送消息到DashScope API (异步): {dashscope_messages}")
        
        resp = await AioGeneration.call(
            messages=dashscope_messages,
            result_format="message",
            stream=False,
            **self._build_params(**kwargs),
        )
        
        # 打印响应状态和详情
        print(f"API响应状态: {resp.status_code}")
        
        # 增强错误处理
        if resp.status_code != HTTPStatus.OK:
            print(f"错误: {getattr(resp, 'message', '未知错误')}")
            return ChatResponse(message=ChatMessage(role="assistant", content="API调用失败"), raw=resp)
        
        # 获取内容
        answer = resp.output.choices[0].message.content
        
        return ChatResponse(message=ChatMessage(role="assistant", content=answer), raw=resp)


    @llm_chat_callback()
    async def astream_chat(self, messages: Sequence[ChatMessage], formatted: bool = False, **kwargs) -> ChatResponseAsyncGen:
        """异步流式聊天方法"""
        # 正确转换消息格式
        dashscope_messages = []
        for m in messages:
            dashscope_messages.append({
                "role": m.role.value if hasattr(m.role, 'value') else m.role,
                "content": m.content
            })
        
        print(f"发送消息到DashScope API (异步流式): {dashscope_messages}")
        
        resp_iter = await AioGeneration.call(
            messages=dashscope_messages,
            result_format="message",
            stream=True,
            incremental_output=True,
            **self._build_params(**kwargs),
        )

        async def agen():
            content = ""
            
            async for chunk in resp_iter:
                # 增强错误处理
                if (chunk is None or 
                    chunk.status_code != HTTPStatus.OK or
                    getattr(chunk, 'output', None) is None or
                    getattr(chunk.output, 'choices', None) is None or
                    len(chunk.output.choices) == 0 or
                    getattr(chunk.output.choices[0], 'message', None) is None):
                    delta = ""
                else:
                    delta = chunk.output.choices[0].message.content
                    
                content += delta
                yield ChatResponse(
                    message=ChatMessage(role="assistant", content=content),
                    delta=delta,
                    raw=chunk,
                )

        return agen()

    # @llm_chat_callback()
    # async def astream_chat(self, messages: Sequence[ChatMessage], formatted: bool = False, **kwargs) -> ChatResponseAsyncGen:
    #     """异步流式聊天方法"""
    #     resp_iter = await AioGeneration.call(
    #         messages=[m.model_dump() for m in messages],
    #         result_format="message",
    #         stream=True,
    #         incremental_output=True,
    #         **self._build_params(**kwargs),
    #     )

    #     async def agen() -> AsyncGenerator[ChatResponse, None]:
    #         content = ""
    #         async for chunk in resp_iter:
    #             # 添加健壮的错误处理
    #             if (chunk is None or 
    #                 chunk.status_code != HTTPStatus.OK or
    #                 getattr(chunk, 'output', None) is None or
    #                 getattr(chunk.output, 'choices', None) is None or
    #                 len(chunk.output.choices) == 0 or
    #                 getattr(chunk.output.choices[0], 'message', None) is None):
    #                 delta = ""
    #             else:
    #                 delta = chunk.output.choices[0].message.content
                    
    #             content += delta
    #             yield ChatResponse(
    #                 message=ChatMessage(role="assistant", content=content),
    #                 delta=delta,
    #                 raw=chunk,
    #             )

    #     return agen()
