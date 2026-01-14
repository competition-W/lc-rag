import os, base64, tempfile
from io import BytesIO
from PIL import Image
from typing import Optional, Dict, Union
from llm.custom_vlm_2 import DashScopeVLM

class MultimodalProcessor:
    def __init__(self, dashscope_vlm: Optional[DashScopeVLM] = None):
        self.dashscope_vlm = dashscope_vlm or DashScopeVLM() ##似乎没有设置模型名称啊这种,调用的时候传入了VLM

    def _is_base64(self, s: str) -> bool:
        try:
            # 判断是否为Base64数据（忽略开头的data URI）
            b = s.split(",")[-1]
            base64.b64decode(b, validate=True)
            return True
        except Exception:
            return False

    def _save_base64_to_file(self, b64: str, fmt="png") -> str:
        data = base64.b64decode(b64.split(",")[-1])
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{fmt}")
        tmp.write(data); tmp.close()
        return tmp.name

    def _prepare_input(self, image_input: str) -> str:
        if self._is_base64(image_input):
            return self._save_base64_to_file(image_input)
        if image_input.startswith(("http://","https://")):
            return image_input
        # 处理 file:// 前缀
        if image_input.startswith("file://"):
            path = image_input[len("file://"):]
            if os.path.exists(path):
                return path  # 直接返回绝对路径
        if os.path.exists(image_input):
            return image_input
        raise ValueError("Unsupported image input")

    async def process_image(self, image_input: str, metadata: Dict = None) -> str:
        prompt = metadata.get("prompt","请分析图像内容")
        path = self._prepare_input(image_input)
        # 如果没有 VLM，可做基础解析
        if not self.dashscope_vlm:
            data = base64.b64decode(image_input.split(",")[-1])
            img = Image.open(BytesIO(data))
            return f"图像基本信息：{img.size[0]}×{img.size[1]} 像素，格式 {img.format}"
        try:
            return await self.dashscope_vlm.acomplete(
                prompt=prompt,
                images=path,
                system_prompt="你是一个专业的图像分析助手..."
            )
        finally:
            if self._is_base64(image_input):
                try: os.unlink(path)
                except: pass
