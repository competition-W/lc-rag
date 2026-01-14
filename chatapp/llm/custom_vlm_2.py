import os
import base64
from typing import Optional, Dict, List, Union, Generator, AsyncGenerator, Sequence
from dashscope import MultiModalConversation
from pathlib import Path
import asyncio
import tempfile
from llama_index.core.schema import ImageDocument
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock, MessageRole
import llm.key_url

class DashScopeVLM:
    """自定义DashScope视觉语言模型接口"""
    
    def __init__(
        self,
        model_name: str = "qwen-vl-max-latest",
        api_key: Optional[str] = None,
        incremental_output: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = 1234,
        max_tokens: Optional[int] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY') ###应该是能自己找到key_url中的api key的
        self.incremental_output = incremental_output
        self.top_k = top_k
        self.top_p = top_p
        self.seed = seed
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("API key must be provided or set as DASHSCOPE_API_KEY environment variable")
    
    def _prepare_parameters(self, stream: bool = False) -> Dict:
        """准备API调用参数"""
        params = {}
        if self.top_k is not None:
            params["top_k"] = self.top_k
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.seed is not None:
            params["seed"] = self.seed
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        
        if stream:
            params["incremental_output"] = self.incremental_output
            params["stream"] = True
            
        return params
    
    def _prepare_image_path(self, image_path: str) -> str:
        """处理图像路径，确保使用file://前缀"""
        if image_path.startswith("http://") or image_path.startswith("https://"):
            return image_path
        
        # 处理本地路径
        path = Path(image_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        return f"file://{path}"
    
    def _prepare_messages(self, prompt: str, image_paths: List[str], system_prompt: Optional[str] = None) -> List[Dict]:
        """准备消息格式"""
        messages = []
        
        # 添加系统提示
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"text": system_prompt}]
            })
        
        # 准备用户消息
        user_content = []
        for img_path in image_paths:
            user_content.append({"image": self._prepare_image_path(img_path)})
        
        user_content.append({"text": prompt})
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    def _extract_image_paths_from_documents(self, image_documents: List[ImageDocument]) -> List[str]:
        """从ImageDocument对象中提取图像路径"""
        image_paths = []
        for doc in image_documents:
            if doc.image_resource and doc.image_resource.path:
                image_paths.append(str(doc.image_resource.path))
            elif doc.image_resource and doc.image_resource.url:
                image_paths.append(doc.image_resource.url)
        return image_paths
    
    def _process_input_images(self, images: Union[str, List[str], List[ImageDocument], ImageDocument]) -> List[str]:
        """处理各种类型的图像输入，统一转换为路径列表"""
        if isinstance(images, ImageDocument):
            return self._extract_image_paths_from_documents([images])
        elif isinstance(images, list) and all(isinstance(img, ImageDocument) for img in images):
            return self._extract_image_paths_from_documents(images)
        elif isinstance(images, str):
            return [images]
        elif isinstance(images, list) and all(isinstance(img, str) for img in images):
            return images
        else:
            raise TypeError("images must be string path(s) or ImageDocument(s)")
    
    def _process_chat_messages(self, messages: List[ChatMessage]) -> List[Dict]:
        """处理ChatMessage列表转换为DashScope API格式"""
        dashscope_messages = []
        
        for message in messages:
            content = []
            
            # 处理消息块
            if hasattr(message, 'blocks') and message.blocks:
                for block in message.blocks:
                    if isinstance(block, TextBlock):
                        content.append({"text": block.text})
                    elif isinstance(block, ImageBlock):
                        # 处理图像块
                        if block.path:
                            content.append({"image": self._prepare_image_path(block.path)})
                        elif block.url:
                            content.append({"image": block.url})
                        elif block.image:
                            # 如果是base64编码的图像
                            try:
                                # 尝试解码base64字符串，可能是路径
                                decoded = base64.b64decode(block.image).decode('utf-8')
                                # 如果解码成功且像是路径
                                if os.path.exists(decoded):
                                    content.append({"image": self._prepare_image_path(decoded)})
                                else:
                                    # 可能是内存中的图像数据，需要保存到临时文件
                                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                                    temp_file.write(base64.b64decode(block.image))
                                    temp_file.close()
                                    content.append({"image": self._prepare_image_path(temp_file.name)})
                            except Exception:
                                # 如果解码失败，假设它可能已经是一个路径
                                content.append({"image": self._prepare_image_path(str(block.image))})
            # 处理直接包含内容的消息
            elif hasattr(message, 'content') and message.content:
                content.append({"text": message.content})
            
            role = message.role.value if hasattr(message.role, 'value') else message.role
            
            dashscope_messages.append({
                "role": role,
                "content": content
            })
        
        return dashscope_messages
    
    def complete(self, prompt: str, images: Union[str, List[str], List[ImageDocument], ImageDocument], system_prompt: Optional[str] = None, **kwargs) -> str:
        """同步调用模型获取回复，支持多种图像输入类型"""
        image_paths = self._process_input_images(images)
        messages = self._prepare_messages(prompt, image_paths, system_prompt)
        params = self._prepare_parameters(stream=False)
        params.update(kwargs)
        
        try:
            response = MultiModalConversation.call(
                api_key=self.api_key,
                model=self.model_name,
                messages=messages,
                **params
            )
            
            if response.status_code == 200 and response.get("output") and response["output"].get("choices"):
                return response["output"]["choices"][0]["message"]["content"][0]["text"]
            else:
                error_msg = response.get("message", "Unknown error")
                raise RuntimeError(f"DashScope API error: {error_msg}")
        except Exception as e:
            raise RuntimeError(f"Error calling DashScope API: {str(e)}")
        
        return ""
    
    def stream_complete(self, prompt: str, images: Union[str, List[str], List[ImageDocument], ImageDocument], system_prompt: Optional[str] = None, **kwargs) -> Generator[str, None, None]:
        """流式调用模型获取回复，支持多种图像输入类型"""
        image_paths = self._process_input_images(images)
        messages = self._prepare_messages(prompt, image_paths, system_prompt)
        params = self._prepare_parameters(stream=True)
        params.update(kwargs)
        
        try:
            responses = MultiModalConversation.call(
                api_key=self.api_key,
                model=self.model_name,
                messages=messages,
                **params
            )
            
            for response in responses:
                if response.status_code == 200:
                    if response.get("output") and response["output"].get("choices"):
                        content = response["output"]["choices"][0]["message"]["content"]
                        if content and len(content) > 0:
                            yield content[0]["text"]
                else:
                    error_msg = response.get("message", "Unknown error")
                    raise RuntimeError(f"DashScope API error: {error_msg}")
        except Exception as e:
            raise RuntimeError(f"Error streaming from DashScope API: {str(e)}")
    
    def chat(self, messages: List[ChatMessage], **kwargs) -> str:
        """同步聊天方法，支持ChatMessage列表输入"""
        dashscope_messages = self._process_chat_messages(messages)
        params = self._prepare_parameters(stream=False)
        params.update(kwargs)
        
        try:
            response = MultiModalConversation.call(
                api_key=self.api_key,
                model=self.model_name,
                messages=dashscope_messages,
                **params
            )
            
            if response.status_code == 200 and response.get("output") and response["output"].get("choices"):
                return response["output"]["choices"][0]["message"]["content"][0]["text"]
            else:
                error_msg = response.get("message", "Unknown error")
                raise RuntimeError(f"DashScope API error: {error_msg}")
        except Exception as e:
            raise RuntimeError(f"Error calling DashScope API: {str(e)}")
        
        return ""
    
    def stream_chat(self, messages: List[ChatMessage], **kwargs) -> Generator[str, None, None]:
        """流式聊天方法，支持ChatMessage列表输入"""
        dashscope_messages = self._process_chat_messages(messages)
        params = self._prepare_parameters(stream=True)
        params.update(kwargs)
        
        try:
            responses = MultiModalConversation.call(
                api_key=self.api_key,
                model=self.model_name,
                messages=dashscope_messages,
                **params
            )
            
            for response in responses:
                if response.status_code == 200:
                    if response.get("output") and response["output"].get("choices"):
                        content = response["output"]["choices"][0]["message"]["content"]
                        if content and len(content) > 0:
                            yield content[0]["text"]
                else:
                    error_msg = response.get("message", "Unknown error")
                    raise RuntimeError(f"DashScope API error: {error_msg}")
        except Exception as e:
            raise RuntimeError(f"Error streaming from DashScope API: {str(e)}")
    
    async def acomplete(self, prompt: str, images: Union[str, List[str], List[ImageDocument], ImageDocument], system_prompt: Optional[str] = None, **kwargs) -> str:
        """异步调用模型获取回复"""
        # 使用线程池执行同步操作以避免阻塞事件循环
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.complete(prompt, images, system_prompt, **kwargs)
        )
    
    async def astream_complete(self, prompt: str, images: Union[str, List[str], List[ImageDocument], ImageDocument], system_prompt: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        """异步流式调用模型获取回复"""
        # 使用线程池获取同步生成器
        loop = asyncio.get_event_loop()
        generator = await loop.run_in_executor(
            None,
            lambda: self.stream_complete(prompt, images, system_prompt, **kwargs)
        )
        
        # 转换为异步生成器
        for chunk in generator:
            yield chunk
            # 让出控制权给事件循环
            await asyncio.sleep(0)
    
    async def achat(self, messages: List[ChatMessage], **kwargs) -> str:
        """异步聊天方法"""
        # 使用线程池执行同步操作以避免阻塞事件循环
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.chat(messages, **kwargs)
        )
    
    async def astream_chat(self, messages: List[ChatMessage], **kwargs) -> AsyncGenerator[str, None]:
        """异步流式聊天方法"""
        # 使用线程池获取同步生成器
        loop = asyncio.get_event_loop()
        generator = await loop.run_in_executor(
            None,
            lambda: self.stream_chat(messages, **kwargs)
        )
        
        # 转换为异步生成器
        for chunk in generator:
            yield chunk
            # 让出控制权给事件循环
            await asyncio.sleep(0)
