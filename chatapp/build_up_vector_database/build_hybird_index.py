## 0703,除了GPU索引外，使用sparse和dense两种嵌入向量
## 0715 暂时没用到吧
'''使用了hybird嵌入，问答的时候就用使用hybird query'''
from llama_index.core.schema import TextNode
from typing import List
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
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

def build_up_vector_store_default_hybird(
        nodes: List[TextNode],
        EMBED_MODEL,
        persist_dir: str, ##参考主文件build_up_vector_store的step_dir
        URI: str = "http://localhost:19530",
        collection_name: str = "lcsw0617",
        overwrite: bool = True,
        dim: int = 1024):
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
        dim=dim,    #1536
        overwrite=overwrite,
        collection_name=collection_name,
        index_config=Config.DEFAULT_index_config,
        enable_dense=True, # enable the default full-text search using BM25
        ##这种情况只会使用
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


from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction
def build_up_vector_store_default_hybird(
        nodes: List[TextNode],
        EMBED_MODEL,
        persist_dir: str, ##参考主文件build_up_vector_store的step_dir
        URI: str = "http://localhost:19530",
        collection_name: str = "lcsw0617",
        overwrite: bool = True,
        dim: int = 1024):
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
        dim=dim,    #1536
        overwrite=overwrite,
        collection_name=collection_name,
        index_config=Config.DEFAULT_index_config,
        enable_dense=True, # enable the default full-text search using BM25
        sparse_embedding_function=BGEM3SparseEmbeddingFunction(),
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