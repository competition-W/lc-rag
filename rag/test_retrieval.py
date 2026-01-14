import asyncio
import sys
import os

# 添加路径
sys.path.insert(0, '/mnt/omicshub/rag')

from config import settings
from llm.async_embedding import get_async_embed_model
from pymilvus import connections, Collection

# 1. 设置查询问题 (根据你的数据内容调整)
QUERY_TEXT = "猪背最长肌"  # 这是一个根据日志推测的关键词

async def test_search():
    print(f"🔍 正在测试检索，问题: [{QUERY_TEXT}]")
    
    # 2. 生成 Query 向量
    print("   1. 生成向量中...")
    client = get_async_embed_model()
    # 注意：embed_texts 返回的是列表的列表，我们取第一个
    embeddings = await client.embed_texts([QUERY_TEXT])
    query_vector = embeddings[0]
    print(f"   ✅ 向量生成完毕 (维度: {len(query_vector)})")
    
    # 3. 连接 Milvus 直接查询
    print("   2. 连接 Milvus...")
    connections.connect(
        alias="default", 
        host=settings.MILVUS_HOST, 
        port=settings.MILVUS_PORT
    )
    
    collection_name = "dept_market"  # 日志中显示的集合名
    print(f"   3. 在集合 [{collection_name}] 中搜索...")
    
    col = Collection(collection_name)
    col.load()
    
    search_params = {
        "metric_type": "COSINE", 
        "params": {"nprobe": 10}
    }
    
    results = col.search(
        data=[query_vector], 
        anns_field="embedding", 
        param=search_params, 
        limit=3,  # 取前3个结果
        output_fields=["text", "source"] # 假设字段名为 text 和 source，如果报错请根据你的Schema调整
    )
    
    print("\n" + "="*50)
    print(f"🎉 检索结果 (Top 3):")
    print("="*50)
    
    for hits in results:
        for hit in hits:
            print(f"📄 [相似度: {hit.score:.4f}] ID: {hit.id}")
            # 尝试打印内容，如果字段名不对可能会获取不到，但不影响核心流程
            try:
                content = hit.entity.get('text')
                source = hit.entity.get('source')
                print(f"   来源: {source}")
                print(f"   内容: {content[:100]}..." if content else "   内容: (空)")
            except:
                pass
            print("-" * 30)

if __name__ == "__main__":
    asyncio.run(test_search())
