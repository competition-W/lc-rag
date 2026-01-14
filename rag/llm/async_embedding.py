# 文件: /mnt/omicshub/rag/llm/async_embedding.py
import aiohttp
import asyncio
import json
from typing import List
from config import settings
from utils.logger import logger

class AsyncEmbeddingClient:
    """
    【最终稳定版】同步 API + 客户端并发
    特点：
    1. 不依赖 MinIO 或本地文件上传，纯 HTTP 请求。
    2. 使用 text-embedding-v3 实时接口，稳定可靠。
    3. 利用 asyncio 实现并发，处理 4000 条数据仅需约 15-20 秒。
    """
    
    def __init__(self, model_name: str = "text-embedding-v4"):
        # 无论传入什么（比如传入了 async-v2），强制切换到实时模型 v4
        self.model_name = "text-embedding-v4"
        self.api_key = settings.DASHSCOPE_API_KEY
        self.url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        
        # 性能调优配置
        self.batch_size = 10       # 阿里云单次请求最大支持 10 条
        self.max_concurrent = 2   # 并发协程数（相当于同时开 10 个线程请求）
        
        logger.info(f"🛡️ [StableMode] 初始化 Embedding 客户端")
        logger.info(f"   强制使用模型: {self.model_name}")
        logger.info(f"   并发策略: 批次大小={self.batch_size}, 并发数={self.max_concurrent}")

    async def _fetch_batch(self, session: aiohttp.ClientSession, texts: List[str], batch_id: int):
        """发送单个 HTTP 请求"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "input": {"texts": texts},
            "parameters": {"text_type": "document"}
        }

        try:
            # 设置 60 秒超时，防止网络波动
            async with session.post(self.url, json=payload, headers=headers, timeout=60) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"❌ [Batch {batch_id}] API 请求失败: {error_text}")
                    raise Exception(f"API Error {response.status}: {error_text}")
                
                result = await response.json()
                if "output" not in result or "embeddings" not in result["output"]:
                    raise ValueError(f"响应格式异常: {result}")
                
                embeddings = [item["embedding"] for item in result["output"]["embeddings"]]
                logger.info(f"   ✅ [Batch {batch_id}] 完成 ({len(embeddings)} 条)")
                return embeddings
        except Exception as e:
            logger.error(f"❌ [Batch {batch_id}] 网络/解析错误: {e}")
            raise

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """主入口：并发处理所有文本"""
        total_count = len(texts)
        logger.info(f"🚀 [Start] 开始处理 {total_count} 条文本 (纯内存并发模式)")
        
        # 1. 切分批次
        batches = [texts[i:i + self.batch_size] for i in range(0, total_count, self.batch_size)]
        logger.info(f"   📦 数据被切分为 {len(batches)} 个批次")

        all_embeddings = []
        
        # 2. 创建并发连接池
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            # 使用 Semaphore 限制同时进行的请求数量，防止触发阿里云限流
            sem = asyncio.Semaphore(self.max_concurrent)

            async def _worker(batch, idx):
                async with sem:
                    return await self._fetch_batch(session, batch, idx)

            # 3. 创建所有任务
            tasks = [
                _worker(batch, i+1) 
                for i, batch in enumerate(batches)
            ]
            
            # 4. 并发执行并等待所有结果
            results = await asyncio.gather(*tasks)
            
            # 5. 合并结果
            for res in results:
                all_embeddings.extend(res)

        # 6. 最终校验
        if len(all_embeddings) != total_count:
            raise ValueError(f"数量校验失败: 输入 {total_count} vs 输出 {len(all_embeddings)}")
            
        logger.info(f"🎉 [Success] 全部完成！生成 {len(all_embeddings)} 个向量")
        return all_embeddings

def get_async_embed_model(model_name: str = "text-embedding-v3"):
    # 工厂函数，兼容旧调用
    return AsyncEmbeddingClient(model_name=model_name)