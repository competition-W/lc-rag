import os
from llama_index.embeddings.dashscope import DashScopeEmbedding
##呼叫api_key and url
import llm.key_url
# 设置模型
from llm.embedding import CustomDashScopeEmbedding
###只能dense嵌入
EMBED_MODEL = CustomDashScopeEmbedding(
    model_name="text-embedding-v4",         # 文本嵌入模型
    image_model_name="multimodal-embedding-v1",  # 图像嵌入模型
    dimension=1024,                          # 可选：指定文本嵌入向量维度
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 从环境变量获取API密钥
    base_url=os.getenv("DASHSCOPE_BASE_URL")  # 从环境变量获取基础URL
)

from llm.embedding import CustomDashScopeEmbedding###使用自定义的EMBEDDING类，与llama-index和langchain的集成不同，可以设置
### 也只能dense嵌入，没写那个sparse_embedding的处理方法
def get_embed_model(model_name: str= "text-embedding-v4", dim: int=1024, output_type:str = "dense"):
    '''
    获取自定义嵌入模型实例
    '''
    # ✅ 重点：embed_batch_size=10
    # 阿里云限制单次请求最多 25 条文本。
    # 设置为 10 是比较稳妥的，既有速度又不会超限。
    embedding = CustomDashScopeEmbedding(
            model_name=model_name, 
            dimension=dim,
            output_type=output_type,
            embed_batch_size=100  # 👈 必须加上这个！
        )
    return embedding

''' 
    这里不支持修改嵌入向量维度，Qwen3实现一个接口类，可能可以支持定义嵌入向量维度
    豆包不支持修改
    from llama_index.core.embeddings import BaseEmbedding
    这里需要实现
    判断哪个模型供应商，选择它下面的模型
    dashscope字符串列表和文件数最多都是25
'''