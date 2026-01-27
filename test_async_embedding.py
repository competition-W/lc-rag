#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试async_embedding.py的API Key处理逻辑
"""
import asyncio
from rag.llm.async_embedding import get_async_embed_model

async def test_async_embedding():
    """测试异步嵌入模型的初始化"""
    print("🔍 测试async_embedding.py的API Key处理...")
    
    # 初始化嵌入模型客户端
    embed_client = get_async_embed_model()
    print(f"✅ 成功初始化嵌入模型客户端")
    print(f"   模型名称: {embed_client.model_name}")
    print(f"   API URL: {embed_client.url}")
    print(f"   API Key状态: {'已配置' if embed_client.api_key else '未配置'}")
    
    # 测试简单的嵌入生成
    try:
        texts = ["测试文本1", "测试文本2"]
        embeddings = await embed_client.embed_texts(texts)
        print(f"✅ 成功生成嵌入，返回 {len(embeddings)} 个向量")
        print(f"   第一个向量维度: {len(embeddings[0])}")
    except Exception as e:
        print(f"⚠️  嵌入生成失败 (可能是API Key问题): {e}")

if __name__ == "__main__":
    asyncio.run(test_async_embedding())