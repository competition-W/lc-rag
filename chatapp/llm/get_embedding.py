import os
from llama_index.embeddings.dashscope import DashScopeEmbedding
##呼叫api_key and url
import llm.key_url
# 设置模型
from llm.embedding_v2_vlm import CustomDashScopeEmbedding
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
def get_embed_model(model_name: str= "text-embedding-v4",dim: int=1024, output_type:str = "dense"):
    '''output_type string 可选
    用户指定输出离散向量表示只适用于text_embedding_v3与text_embedding_v4模型，
    取值在dense、sparse、dense&sparse之间，默认取dense，只输出连续向量。'''
    embedding = CustomDashScopeEmbedding(
            model_name= model_name, ##支持的维度64->2048
            dimension=dim, # 例如，如果您的模型支持且需要指定维度
            output_type=output_type
        )
    ## output_type= [dense、sparse、dense&sparse]
    # https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api?spm=a2c4g.11186623.help-menu-2400256.d_2_6_0.7cb548234p7QgN
    return embedding

''' 
    这里不支持修改嵌入向量维度，Qwen3实现一个接口类，可能可以支持定义嵌入向量维度
    豆包不支持修改
    from llama_index.core.embeddings import BaseEmbedding
    这里需要实现
    判断哪个模型供应商，选择它下面的模型
    dashscope字符串列表和文件数最多都是25
'''
### 符合这个规范就能使用from llama_index.core.embeddings import BaseEmbedding
### 还是说能嵌入就能用？？目前两种都能用 ##########但是两种调用方法的内置方法都不一样
from llama_index.core.embeddings import BaseEmbedding

from langchain_community.embeddings.dashscope import DashScopeEmbeddings
def get_text_embeddings():
    '''目前使用的llama-index的方法，会让系统报错，为什么呢，默认为1536？
    这个langchain_community默认嵌入维度为1024'''
    embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)
    return embeddings

##############################################################多模态嵌入文本、图片，输出1024维度
# import dashscope
# import json
# from http import HTTPStatus

# text = "通用多模态表征模型示例"
# input = [{'text': text}]
# # 调用模型接口
# resp = dashscope.MultiModalEmbedding.call(
#     model="multimodal-embedding-v1",
#     input=input
# )

# if resp.status_code == HTTPStatus.OK:
#     print(json.dumps(resp.output, ensure_ascii=False, indent=4))

# import dashscope
# import base64
# import json
# from http import HTTPStatus
# # 读取图片并转换为Base64,实际使用中请将xxx.png替换为您的图片文件名或路径
# image_path = "xxx.png"
# with open(image_path, "rb") as image_file:
#     # 读取文件并转换为Base64
#     base64_image = base64.b64encode(image_file.read()).decode('utf-8')
# # 设置图像格式
# image_format = "png"  # 根据实际情况修改，比如jpg、bmp 等
# image_data = f"data:image/{image_format};base64,{base64_image}"
# # 输入数据
# inputs = [{'image': image_data}]

# # 调用模型接口
# resp = dashscope.MultiModalEmbedding.call(
#     model="multimodal-embedding-v1",
#     input=inputs
# )
# if resp.status_code == HTTPStatus.OK:
#     print(json.dumps(resp.output, ensure_ascii=False, indent=4))
# def get_image_embeddings():
#     image_embed_model = DashScopeEmbedding(
#         model_name="multimodal-embedding-one-image",  # 通义千问的多模态嵌入模型
#         api_key=os.getenv("DASHSCOPE_API_KEY"),
#         embed_batch_size=2
#     )
#     return image_embed_model

# import dashscope
# import json
# from http import HTTPStatus
# # 实际使用中请将url地址替换为您的图片url地址
# image = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png"
# input = [{'image': image}]
# # 调用模型接口
# resp = dashscope.MultiModalEmbedding.call(
#     model="multimodal-embedding-v1",
#     input=input
# )

# if resp.status_code == HTTPStatus.OK:
#     print(json.dumps(resp.output, ensure_ascii=False, indent=4))

