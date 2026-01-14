from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
import llm.key_url
import os
######这个接口支持的模型比较少,这样调用没反应
#####先用turbo顶一下
def get_dashscope_qwen_turbo(model_name : str ="qwen-plus"):
    dashscope_llm = DashScope(model_name=model_name,
                          base_url ="https://dashscope.aliyuncs.com/compatible-mode/v1",
                          max_tokens=1024,
                          temperature=0.8,
                          api_key=os.getenv("DASHSCOPE_API_KEY"))
    return dashscope_llm

# from llm.inference import DashScopeLLM
from llm.custom_vlm_2 import DashScopeVLM

# def get_dashscope_text_llm():

#     return 0

def get_dashscope_vlm(model_name: str="qwen-omni-turbo"):
    # 创建实例
    vlm = DashScopeVLM(
        model_name=model_name,
        api_key=os.getenv('DASHSCOPE_API_KEY'),  # 或直接提供API密钥
        max_tokens=1000
    )
    return vlm