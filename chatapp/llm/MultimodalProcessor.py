# import io
# import base64
# import tempfile
# import os
# from PIL import Image
# from io import BytesIO
# from typing import Dict, List, Optional, Any
# from pathlib import Path
# from llm.custom_vlm_2 import DashScopeVLM

# class MultimodalProcessor:
#     def __init__(self, dashscope_vlm: Optional[DashScopeVLM] = None):
#         """
#         初始化多模态处理器
        
#         Args:
#             dashscope_vlm: DashScope视觉语言模型实例
#         """
#         self.recognizer = sr.Recognizer()
#         self.dashscope_vlm = dashscope_vlm
        
#         # 如果没有提供VLM实例，则创建默认实例
#         if self.dashscope_vlm is None:
#             try:
#                 self.dashscope_vlm = DashScopeVLM()
#             except Exception as e:
#                 logger.warning(f"无法初始化DashScope VLM: {e}")
#                 self.dashscope_vlm = None
    
#     def _save_base64_image_to_temp(self, image_content: str, format_hint: str = "PNG") -> str:
#         """
#         将base64编码的图像保存到临时文件
        
#         Args:
#             image_content: base64编码的图像数据
#             format_hint: 图像格式提示
            
#         Returns:
#             临时文件路径
#         """
#         try:
#             # 解码base64图像
#             image_data = base64.b64decode(image_content)
            
#             # 创建临时文件
#             suffix = f".{format_hint.lower()}" if format_hint else ".png"
#             temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
#             temp_file.write(image_data)
#             temp_file.close()
            
#             return temp_file.name
            
#         except Exception as e:
#             raise ValueError(f"无法保存图像到临时文件: {e}")
    
#     async def process_image(self, image_content: str, metadata: Dict = None) -> str:
#         """
#         处理图像，返回图像描述文本
        
#         Args:
#             image_content: base64编码的图像数据
#             metadata: 图像元数据，可能包含filename、content_type等
            
#         Returns:
#             图像描述文本
#         """
#         try:
#             # 如果没有VLM模型，返回基本信息
#             if self.dashscope_vlm is None:
#                 image_data = base64.b64decode(image_content)
#                 image = Image.open(BytesIO(image_data))
#                 return f"检测到一张{image.size[0]}x{image.size[1]}像素的图片，格式为{image.format}。（注：图像理解功能未启用）"
            
#             # 保存图像到临时文件
#             temp_image_path = self._save_base64_image_to_temp(image_content)
            
#             try:
#                 # 构建图像分析提示
#                 prompt = self._build_image_analysis_prompt(metadata)
                
#                 # 使用DashScope VLM进行图像理解
#                 description = await self.dashscope_vlm.acomplete(
#                     prompt=prompt,
#                     images=temp_image_path,
#                     system_prompt="你是一个专业的图像分析助手，请详细描述图像内容，包括物体、场景、文字、统计图等信息。"
#                 )
                
#                 return description
                
#             finally:
#                 # 清理临时文件
#                 try:
#                     os.unlink(temp_image_path)
#                 except:
#                     pass
                    
#         except Exception as e:
#             logger.error(f"图像处理失败: {e}")
#             return f"图像处理失败: {str(e)}"
    
#     # def _build_image_analysis_prompt(self, metadata: Dict = None) -> str:
#     #     """构建图像分析提示词"""
#     #     base_prompt = "请详细分析这张图片，描述其中包含的内容，包括：\n1. 主要物体和场景\n2. 图片中的文字内容（如果有）\n3. 图片的整体风格和特点"
        
#     #     if metadata:
#     #         filename = metadata.get('filename', '')
#     #         if filename:
#     #             base_prompt += f"\n\n图片文件名：{filename}"
        
#     #     return base_prompt
#     def _build_image_analysis_prompt(self, metadata: Dict = None) -> str:
#         return """请专注于分析统计图表内容，包括：
#         1. 图表类型（柱状图、折线图、饼图等）
#         2. 坐标轴含义（单位、标签）
#         3. 数据趋势、关键数值和异常点
#         4. 图中的文字说明（标题、图例）
#         * 忽略与数据分析无关的背景信息"""
#     async def process_multimodal_messages(self, messages: List[MessageContent]) -> str:
#         image_desc, text_query = "", ""
#         for msg in messages:
#             if msg.type == MessageType.IMAGE:
#                 image_desc = await self.process_image(msg.content, msg.metadata)
#             elif msg.type == MessageType.TEXT:
#                 text_query = msg.content
        
#         if not text_query:  # 只有图片输入时
#             return f"图片分析结果：{image_desc}"
#         elif "数据" in text_query or "趋势" in text_query:  # 数据相关查询
#             return f"根据图表分析：{image_desc}\n\n用户问题：{text_query}"
#         else:  # 普通文本查询
#             return text_query  # 或简单拼接
    # async def process_audio(self, audio_content: str, metadata: Dict = None) -> str:
    #     """
    #     处理音频，返回语音转文本结果
        
    #     Args:
    #         audio_content: base64编码的音频数据
    #         metadata: 音频元数据
            
    #     Returns:
    #         语音转文本结果
    #     """
    #     try:
    #         # 解码base64音频
    #         audio_data = base64.b64decode(audio_content)
            
    #         # 保存到临时文件
    #         temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    #         temp_audio.write(audio_data)
    #         temp_audio.close()
            
    #         try:
    #             # 使用语音识别
    #             with sr.AudioFile(temp_audio.name) as source:
    #                 audio = self.recognizer.record(source)
    #                 # 尝试使用百度语音识别（支持中文）
    #                 try:
    #                     text = self.recognizer.recognize_sphinx(audio)
    #                 except:
    #                     # 如果百度识别失败，使用Google识别
    #                     text = self.recognizer.recognize_google(audio, language='zh-CN')
                    
    #                 return text
                    
    #         finally:
    #             # 清理临时文件
    #             try:
    #                 os.unlink(temp_audio.name)
    #             except:
    #                 pass
                    
    #     except Exception as e:
    #         logger.error(f"音频处理失败: {e}")
    #         return f"音频处理失败: {str(e)}"
    
    # async def process_file(self, file_content: str, metadata: Dict = None) -> str:
    #     """
    #     处理文件，返回文件内容摘要
        
    #     Args:
    #         file_content: base64编码的文件数据
    #         metadata: 文件元数据
            
    #     Returns:
    #         文件内容摘要
    #     """
    #     try:
    #         file_data = base64.b64decode(file_content)
    #         file_name = metadata.get('filename', 'unknown') if metadata else 'unknown'
    #         file_size = len(file_data)
            
    #         # 根据文件类型处理
    #         if file_name.lower().endswith('.txt'):
    #             try:
    #                 content = file_data.decode('utf-8')
    #                 # 如果内容太长，进行截断
    #                 if len(content) > 1000:
    #                     content = content[:1000] + "...(内容已截断)"
    #                 return f"文本文件内容：\n{content}"
    #             except UnicodeDecodeError:
    #                 return f"文本文件解码失败，文件大小：{file_size} bytes"
            
    #         elif file_name.lower().endswith(('.pdf', '.doc', '.docx')):
    #             return f"收到文档文件：{file_name}，大小：{file_size} bytes。（注：文档解析功能待实现）"
            
    #         elif file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
    #             # 如果是图像文件，转换为图像处理
    #             image_content = base64.b64encode(file_data).decode('utf-8')
    #             return await self.process_image(image_content, metadata)
            
    #         else:
    #             return f"收到文件：{file_name}，大小：{file_size} bytes，类型：{metadata.get('content_type', '未知')}"
                
    #     except Exception as e:
    #         logger.error(f"文件处理失败: {e}")
    #         return f"文件处理失败: {str(e)}"
    
    # async def process_multimodal_messages(self, messages: List[MessageContent]) -> str:
    #     """
    #     处理多模态消息，返回统一的文本查询
        
    #     Args:
    #         messages: 多模态消息列表
            
    #     Returns:
    #         处理后的统一文本查询
    #     """
    #     processed_parts = []
        
    #     for i, message in enumerate(messages):
    #         try:
    #             if message.type == MessageType.TEXT:
    #                 processed_parts.append(message.content)
                
    #             elif message.type == MessageType.IMAGE:
    #                 logger.info(f"处理第{i+1}张图像...")
    #                 image_desc = await self.process_image(message.content, message.metadata)
    #                 processed_parts.append(f"[图像{i+1}分析结果]: {image_desc}")
                
    #             # elif message.type == MessageType.AUDIO:
    #             #     logger.info(f"处理第{i+1}个音频...")
    #             #     audio_text = await self.process_audio(message.content, message.metadata)
    #             #     processed_parts.append(f"[语音{i+1}转文本]: {audio_text}")
                
    #             # elif message.type == MessageType.FILE:
    #             #     logger.info(f"处理第{i+1}个文件...")
    #             #     file_desc = await self.process_file(message.content, message.metadata)
    #             #     processed_parts.append(f"[文件{i+1}内容]: {file_desc}")
                
    #         except Exception as e:
    #             logger.error(f"处理消息{i+1}失败: {e}")
    #             processed_parts.append(f"[消息{i+1}处理失败]: {str(e)}")
        
    #     return "\n\n".join(processed_parts)
