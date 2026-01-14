GLOBAL_DIMENSION = 1024
'''milvus特性，建立GPU索引，查询速度快'''
class Config:
    # 高性能和高召回率的场景，如大规模相似度搜索
    DEFAULT_index_config = {
    "index_type": "GPU_CAGRA",
    "params": {
        "intermediate_graph_degree": 64,
        "graph_degree": 32,
        "build_algo": "IVF_PQ",
        "cache_dataset_on_device": "true"
        },
    "metric_type": "COSINE"
    }
    # ### 内存充足、快速搜索
    # index_config = {
    # "index_type": "GPU_IVF_FLAT",
    # "params": {
    #     "nlist": 1024
    # },
    # "metric_type": "L2" #欧式距离
    # }
    # index_config = {
    # "index_type": "GPU_IVF_PQ",
    # "params": {
    #     "nlist": 1024,
    #     "m": 8,
    #     "nbits": 8
    # },
    # "metric_type": "L2"
    # }
    # index_config = {
    # "index_type": "GPU_BRUTE_FORCE",
    # "params": {},
    # "metric_type": "L2"
    # }

from llama_index.core.schema import TextNode
from typing import List
from llama_index.core import Settings

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore


def buildup_vector_store(
        nodes: List[TextNode],
        EMBED_MODEL,
        persist_dir: str,
        URI: str = "http://localhost:19530", ##主程序会传入http://milvus-standalone:19530
        collection_name: str = "lcsw0617",
        overwrite: bool = True,
        dim: int = GLOBAL_DIMENSION):
    '''
    使用切好块的nodes建立索引
    collection_name用于区分不同语料库的向量数据
    
    # host="milvus-standalone",  # 或您的服务器IP
    # port="19530"
    参数:
        URI: Milvus服务器地址，默认本地容器的19530端口
        nodes: 文本节点列表
        persist_dir: 持久化目录路径
        collection_name: 集合名称，用于区分不同语料库
        overwrite: 是否覆盖现有集合，初始化语料时默认为True
        dim: 结果向量维度
    '''
    Settings.embed_model = EMBED_MODEL
    

    # 初始化Milvus向量存储
    vector_store = MilvusVectorStore(
        uri=URI, 
        dim=dim, 
        overwrite=overwrite,
        collection_name=collection_name,
        index_config=Config.DEFAULT_index_config
        # 可选添加更多Milvus配置
        # user="username",  # 如果设置了认证
        # password="password",
        # token="api_token",
        # secure=True  # 如果使用TLS
    )
    
    # 创建文档存储
    docstore = SimpleDocumentStore()
    for node in nodes:
        docstore.add_documents([node])
    
    # 创建存储上下文
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, 
        docstore=docstore
    )
    
    # 创建索引
    index = VectorStoreIndex(
        nodes=nodes, 
        storage_context=storage_context, 
        embed_model=EMBED_MODEL
    )
    
    # 持久化存储
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"索引已建立，集合名称: {collection_name}")
    return True