# #!/mnt/chatchat/.venv/bin python3
# # -*- coding: utf-8 -*-
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceSplitter,
)
from llama_index.core.schema import TextNode, Document
from typing import List
from .show_the_nodes import show_the_nodes
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes

def get_nodes_from_hierarchical(documents: List[Document]) -> List[TextNode]:
    ## 索引载入，需要专门载入左子节点，那我们返回左子节点如何？
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[1536, 512, 128],
        chunk_overlap = 100
        )
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    show_the_nodes(leaf_nodes)

    return leaf_nodes

# def get_nodes_with_SentenceWindow(documents: List[Document],
#                                   window_size: int =3) -> List[TextNode]:
#     # create the sentence window node parser w/ default settings
#     # 必须发起不同的检索器，进入node_postprocessors = [MetadataReplacementPostProcessor(target_metadata_key="window")]
#     node_parser = SentenceWindowNodeParser.from_defaults(
#         window_size=window_size,
#         window_metadata_key="window",
#         original_text_metadata_key="original_text",
#     )
#     nodes = node_parser.get_nodes_from_documents(documents)

#     show_the_nodes(nodes)
#     return nodes