# 文件: /mnt/omicshub/rag/tasks/document_tasks_async.py

import asyncio
from celery import Task
from llama_index.core.schema import TextNode
from llama_index.core import Settings
from llm.async_embedding import get_async_embed_model
from tasks.celery_app import celery_app
from services.minio_client import minio_client
from services.processors.factory import DocumentProcessorFactory
from services.milvus_manager import milvus_manager
from utils.logger import logger


class DocumentTask(Task):
    """Celery 任务基类"""
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"❌ Task {task_id} failed: {exc}")
        super().on_failure(exc, task_id, args, kwargs, einfo)


@celery_app.task(
    bind=True, 
    base=DocumentTask, 
    name="tasks.process_document_async"  # 👈 修改任务名
)
def process_document_task_async(  # 👈 修改函数名
    self, 
    object_name: str, 
    filename: str, 
    department: str, 
    user_id: str = "system"
):
    """文档处理任务 - 使用异步 Embedding（优化版）"""
    
    logger.info(f"🚀 [AsyncTask] 开始处理文档: {filename}")
    logger.info(f"   部门: {department}, 用户: {user_id}")

    async def _async_workflow():
        try:
            # 1. 从 MinIO 下载
            logger.info("📥 [Step 1/8] 下载文件...")
            file_content = minio_client.download_file(object_name)
            logger.info(f"   文件大小: {len(file_content)} bytes")
            
            # 2. 获取解析器
            logger.info("🔍 [Step 2/8] 获取解析器...")
            processor = DocumentProcessorFactory.get_processor(
                filename, 
                department=department
            )
            if not processor:
                raise ValueError(f"不支持的文件格式: {filename}")
            
            # 3. 执行解析
            logger.info("📄 [Step 3/8] 解析文档...")
            process_result = await processor.process(
                file_content, 
                user_id=user_id, 
                filename=filename
            )
            
            # 4. 获取结果列表
            logger.info("🔄 [Step 4/8] 提取文档片段...")
            if hasattr(process_result, 'nodes'):
                raw_chunks = process_result.nodes
            elif hasattr(process_result, 'chunks'):
                raw_chunks = process_result.chunks
            elif hasattr(process_result, 'documents'):
                raw_chunks = process_result.documents
            else:
                logger.error(f"❌ ProcessResult 属性未知: {dir(process_result)}")
                raise AttributeError("ProcessResult 缺少 nodes/chunks/documents 属性")

            if not raw_chunks:
                logger.warning("⚠️ 文档解析结果为空")
                return {"status": "skipped", "message": "文档为空"}
                
            logger.info(f"✅ 解析完成，生成 {len(raw_chunks)} 个片段")
           
            # 5. 转换为 TextNode
            logger.info("🔄 [Step 5/8] 转换为 TextNode...")
            nodes = []
            for i, chunk in enumerate(raw_chunks):
                # 每 500 个打印一次进度
                if i > 0 and i % 500 == 0:
                    logger.info(f"   进度: {i}/{len(raw_chunks)}")
                
                metadata = chunk.extra_info.copy() if chunk.extra_info else {}
                metadata.update({
                    "department": department,
                    "filename": filename,
                    "chunk_id": i,
                    "owner": user_id
                })
                
                node = TextNode(
                    text=chunk.text,
                    metadata=metadata,
                    excluded_embed_metadata_keys=[
                        "chunk_id", "filename", "department", "owner"
                    ]
                )
                nodes.append(node)
            
            logger.info(f"✅ 转换完成 {len(nodes)} 个节点")
            
            # 6. 使用异步接口批量生成 Embeddings
            logger.info("🚀 [Step 6/8] 异步生成 Embeddings...")
            logger.info(f"   模型: text-embedding-async-v2")
            logger.info(f"   数量: {len(nodes)} 个文本")
            
            async_embed_client = get_async_embed_model(
                model_name="text-embedding-async-v4"
            )
            
            texts = [node.text for node in nodes]
            
            # 一次性提交所有文本（关键优化！）
            embeddings = await async_embed_client.embed_texts(texts)
            
            logger.info(f"✅ Embeddings 生成完成，共 {len(embeddings)} 个向量")
            
            # 7. 将 embeddings 附加到 nodes
            logger.info("🔗 [Step 7/8] 附加向量到节点...")
            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding
            
            # 8. 直接写入 Milvus
            logger.info("💾 [Step 8/8] 写入 Milvus...")
            milvus_manager.insert_nodes_with_embeddings(
                department=department,
                nodes=nodes,
                show_progress=True
            )
            
            logger.info(f"🎉 处理完成! 成功写入 {len(nodes)} 个节点到 {department}")
            return {
                "status": "success", 
                "filename": filename, 
                "nodes_count": len(nodes),
                "department": department
            }

        except Exception as e:
            logger.exception(f"❌ [AsyncTask] 处理失败: {filename}")
            raise e

    # 运行异步工作流
    try:
        return asyncio.run(_async_workflow())
    except Exception as e:
        logger.exception(f"❌ [AsyncTask] 任务执行异常")
        raise e
