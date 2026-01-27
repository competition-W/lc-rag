# 文件: /mnt/omicshub/rag/tasks/document_tasks.py

import asyncio
from celery import Task
from llama_index.core.schema import TextNode
from llama_index.core import Settings
from rag.llm.get_embedding import get_embed_model 
from rag.tasks.celery_app import celery_app
from rag.services.minio_client import minio_client
from rag.services.processors.factory import DocumentProcessorFactory
from rag.services.milvus_manager import milvus_manager
from rag.utils.logger import logger

class DocumentTask(Task):
    """Celery任务基类"""
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed: {exc}")
        super().on_failure(exc, task_id, args, kwargs, einfo)

@celery_app.task(bind=True, base=DocumentTask, name="tasks.process_document")
def process_document_task(self, object_name: str, filename: str, department: str, user_id: str = "system"):
    """
    文档处理任务 - 包含异步环境包装
    """
    logger.info(f"🚀 开始处理文档: {filename} (部门: {department}, 用户: {user_id})")

    # ✅ 定义一个内部异步函数，包含所有逻辑
    async def _async_workflow():
        try:
            # 1. 从 MinIO 下载 (同步操作，但在 async 函数中是允许的，虽然会阻塞 Loop)
            file_content = minio_client.download_file(object_name)
            
            # 2. 获取解析器
            processor = DocumentProcessorFactory.get_processor(filename, department=department)
            if not processor:
                raise ValueError(f"不支持的文件格式: {filename}")
            
            # 3. 执行解析 (异步等待)
            # 注意：这里直接 await，不需要再包 asyncio.run
            process_result = await processor.process(file_content, user_id=user_id, filename=filename)
            
            # 4. 获取结果列表
            if hasattr(process_result, 'nodes'):
                raw_chunks = process_result.nodes
            elif hasattr(process_result, 'chunks'):
                raw_chunks = process_result.chunks
            elif hasattr(process_result, 'documents'):
                raw_chunks = process_result.documents
            else:
                logger.error(f"❌ ProcessResult 属性未知，包含: {dir(process_result)}")
                raise AttributeError("ProcessResult object has no attribute 'nodes' or 'chunks'")

            if not raw_chunks:
                logger.warning("⚠️ 文档解析结果为空")
                return {"status": "skipped"}
                
            logger.info(f"📄 解析完成，生成 {len(raw_chunks)} 个片段")

             # 5. 转换为 LlamaIndex TextNode
            nodes = []
            logger.info("🔄 开始转换 TextNode 对象...")
            for i, chunk in enumerate(raw_chunks):
                # --- ✅ 新增：每 500 个打印一次日志 ---
                if i % 500 == 0:
                    logger.info(f"   正在转换第 {i}/{len(raw_chunks)} 个片段...")
                # -----------------------------------
                
                metadata = chunk.extra_info.copy() if chunk.extra_info else {}
                metadata["department"] = department
                metadata["filename"] = filename
                metadata["chunk_id"] = i
                metadata["owner"] = user_id 
                
                node = TextNode(
                    text=chunk.text,
                    metadata=metadata,
                    excluded_embed_metadata_keys=["chunk_id", "filename", "department", "owner"]
                )
                nodes.append(node)
            
            logger.info(f"✅ TextNode 转换完毕，准备调用 MilvusManager (共 {len(nodes)} 个节点)")

            # 6. 存入 Milvus
            # --- ✅ 关键：这里最容易卡住 ---
            logger.info("⏳ 正在调用 milvus_manager.create_index_from_nodes (这可能需要较长时间进行 Embedding)...")
            
            logger.info("🔧 配置 LlamaIndex 全局 Embedding 模型...")
            
            # 初始化模型 (会自动使用我们在 get_embedding.py 里设置的 batch_size=10)
            custom_model = get_embed_model(model_name="text-embedding-v4", dim=1024)
            
            # 强制覆盖全局设置，确保不走 OpenAI 默认逻辑
            Settings.embed_model = custom_model
            
            # 测试连接
            logger.info("🧪 测试 Embedding 连接...")
            await custom_model.aget_text_embedding("test") 
            logger.info("🧪 连接成功！")

            milvus_manager.create_index_from_nodes(
                department=department,
                nodes=nodes,
                show_progress=True # 👈 建议暂时改为 True，看看 Worker 终端有没有进度条（虽然 Celery 中可能显示乱码，但能看出动没动）
            )
            
            logger.info(f"🎉 处理完成! 成功写入 {len(nodes)} 个节点到 {department}")
            return {
                "status": "success", 
                "filename": filename, 
                "nodes_count": len(nodes)
            }

        except Exception as e:
            logger.exception(f"❌ 内部处理流程失败: {filename}")
            raise e

    # ✅ 使用 asyncio.run 运行整个流程
    # 这会创建一个 Event Loop，直到 _async_workflow 执行完毕
    try:
        return asyncio.run(_async_workflow())
    except Exception as e:
        # 捕获 asyncio.run 抛出的异常
        logger.exception(f"❌ 任务执行异常: {filename}")
        raise e