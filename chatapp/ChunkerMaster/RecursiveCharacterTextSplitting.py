# from .hyper_paramter import *
class hyper_param:
    CHUNK_SIZE = 768
    CHUNK_OVERLAP = 100

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core.schema import TextNode, Document

from .show_the_nodes import show_the_nodes

def get_nodes_with_Recursive(
        documents: List[Document],
        chunk_size: int = hyper_param.CHUNK_SIZE,
        chunk_overlap: int = hyper_param.CHUNK_OVERLAP) -> List[TextNode]:
    '''目前只使用了一种Parser，待补充
    Args: 需要输入切块类型，chunk_size, chunk_overlap
    反正CHUNK_SIZE和CHUNK_OVERLAP、GLOBAL_DIMENSION都会从/chatchat下的主函数传进来
    '''
    parser = LangchainNodeParser(RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=chunk_overlap))
    nodes = parser.get_nodes_from_documents(documents)
    show_the_nodes(nodes)

    return nodes

from llama_index.core.node_parser import MarkdownNodeParser
def get_nodes_with_MarkdownPaser(documents: List[Document]):
    parser = MarkdownNodeParser()

    nodes = parser.get_nodes_from_documents(documents)

    show_the_nodes(nodes)
    return nodes

from llama_index.core.node_parser import SentenceSplitter
def get_nodes_with_SentencePaser(documents: List[Document],
                                 chunk_size: int = hyper_param.CHUNK_SIZE,
        chunk_overlap: int = hyper_param.CHUNK_OVERLAP) -> List[TextNode]:
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    show_the_nodes(nodes)
    return nodes