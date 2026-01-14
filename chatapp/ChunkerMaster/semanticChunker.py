# #!/mnt/chatchat/.venv/bin python3
# # -*- coding: utf-8 -*-
from llama_index.core import Settings
from typing import List
'''语义分块必须指定嵌入模型来计算文本的嵌入向量，
    两次调用嵌入模型，
    一次是按照Sentence计算，确定断点的位置，
    第二次是合并后的块，重新计算一次嵌入向量
    '''
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode, Document

from .show_the_nodes import show_the_nodes


def get_nodes_with_SemanticChunker(documents: List[Document], EMBED_MODEL,
                                   breakpoint_percentile_threshold: int = 95,
                                   max_chars: int = 4096
                                   ) -> List[TextNode]:
    ## max_chars: 超长节点的最大字符数，超过这个长度的节点会被二次分割
    
    # 设置嵌入模型
    Settings.embed_model = EMBED_MODEL
    
    # 使用语义分割器
    splitter = SemanticSplitterNodeParser(
        buffer_size=1, 
        breakpoint_percentile_threshold=breakpoint_percentile_threshold, 
        embed_model=EMBED_MODEL
    )
    
    print("*"*50)
    print(documents[0].text[:500])
    print(f"文档数量: {len(documents)}")
    
    # 获取语义分割的节点
    nodes = splitter.get_nodes_from_documents(documents)
    
    # 检查是否有超长节点
    long_nodes = [node for node in nodes if len(node.text) > max_chars]
    
    # 如果有超长节点，进行额外处理
    if long_nodes:
        print(f"发现 {len(long_nodes)} 个超过 {max_chars} 字符的节点，进行二次分割")
        
        # 创建一个句子分割器用于处理超长节点，语义分割设置得特色一点
        sentence_splitter = SentenceSplitter(chunk_size=max_chars//2, chunk_overlap=200)
        
        # 替换超长节点
        final_nodes = []
        for node in nodes:

            if len(node.text) > max_chars:
                # 对超长节点进行再分割
                # 对超长节点进行分割
                print(f"分割超长节点，长度: {len(node.text)}")
                from llama_index.core import Document
                doc = Document(text=node.text, metadata=node.metadata, id_=node.id_)
                sub_nodes = sentence_splitter.get_nodes_from_documents([doc])

                final_nodes.extend(sub_nodes)
            else:
                final_nodes.append(node)
        
        nodes = final_nodes
    
    show_the_nodes(nodes)
    return nodes

# import os
# from langchain_community.embeddings.dashscope import DashScopeEmbeddings
# from langchain_experimental.text_splitter import SemanticChunker
# import sys
# # 将项目根目录加入 sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) ####实现调用平级文件夹下的文件

# ##呼叫api_key and url
# # import llm.key_url
# ##呼叫embedding models
# import llm.embedding

# # def get_embeddings(model_name):
# #     embeddings = DashScopeEmbeddings(
# #     model="text-embedding-v3",
# #     dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
# # )
# #     return embeddings

# from llm.get_embedding import get_embeddings

# embed_text_model = get_embeddings(llm.embedding.DASH_EMBEDDING_V3) #embed_model是不是在main.py中指定?

# # text_splitter = SemanticChunker(embed_text_model)

# text_splitter = SemanticChunker(
#     embed_text_model, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=90
# )#########这些参数也必须在调用时指定!


