import os
from llama_index.llms.openai import OpenAI
from config import settings # 导入您的配置

def get_dashscope_qwen(model_name: str = None) -> OpenAI:
    """
    使用 OpenAI 兼容协议连接通义千问
    """
    # 1. 优先使用传入的模型名，否则用配置文件里的
    target_model = model_name or settings.LLM_MODEL or "qwen-plus"
    
    # 2. 获取 API Key (优先从 settings 取，取不到则读环境变量)
    api_key = settings.DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY")
    
    # 3. 获取 Base URL (阿里云的 OpenAI 兼容地址)
    # 注意：您的 settings 里写了两次 OPENAI_BASE_URL，取最后生效的那个
    base_url = settings.OPENAI_BASE_URL or "https://dashscope.aliyuncs.com/compatible-mode/v1"

    if not api_key:
        print("❌ [Error] 无法获取 DASHSCOPE_API_KEY，请检查 .env 文件")
        return None

    print(f"🔌 连接 LLM: {target_model} via {base_url}")

    # 4. 初始化 OpenAI 类 (但在此时连接的是阿里云)
    llm = OpenAI(
        model=target_model,
        api_key=api_key,
        api_base=base_url,
        temperature=0.1,  # 这里的温度适合 Parser
        max_tokens=2048,
        reuse_client=False
    )
    
    return llm