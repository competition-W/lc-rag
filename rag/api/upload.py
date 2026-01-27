# 文件: /mnt/omicshub/rag/api/upload.py

from fastapi import APIRouter, UploadFile, File, Form, Depends
import uuid
import os

from utils.response import success, error
from utils.logger import logger
from utils.auth import get_auth_context, AuthContext, verify_upload_permission
from services.minio_client import minio_client
from tasks.document_tasks_async import process_document_task_async
from config import settings

router = APIRouter(tags=["文档管理"])

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    # 允许前端通过Form指定部门，默认为用户当前部门
    target_dept: str = Form(None, description="目标部门（可选）"),
    auth: AuthContext = Depends(get_auth_context)
):
    """
    上传文档接口
    需要权限验证
    """
    # 1. 确定目标部门
    department = target_dept or auth.department
    
    # 2. 权限验证
    has_perm = await verify_upload_permission(auth, department)
    if not has_perm:
        return error(f"无权上传文件到部门: {department}", code=403)

    # 3. 文件类型验证 (基于 config.py)
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.FILE_TYPE_MAPPING:
        return error(f"不支持的文件类型: {ext}. 支持: {list(settings.FILE_TYPE_MAPPING.keys())}")
    
    doc_type = settings.FILE_TYPE_MAPPING[ext]
    if doc_type not in settings.ENABLED_DOCUMENT_TYPES:
        return error(f"该文件类型尚未启用: {doc_type}")

    try:
        # 4. 生成存储路径: department/yyyyMMdd/uuid.ext
        # 为了简单，这里使用 department/uuid.ext
        file_id = str(uuid.uuid4())
        object_name = f"{department}/{file_id}{ext}"
        
        # 5. 读取并上传 MinIO
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE:
            return error(f"文件大小超过限制: {settings.MAX_FILE_SIZE/1024/1024}MB")

        minio_path = minio_client.upload_file(object_name, content, file.content_type)
        
        # 6. 触发 Celery 任务
        task = process_document_task_async.delay(
            object_name=object_name,
            filename=file.filename,
            department=department
        )
        
        logger.info(f"Task submitted: {task.id} | File: {file.filename} | Dept: {department}")

        return success({
            "task_id": task.id,
            "file_id": file_id,
            "minio_path": minio_path,
            "status": "processing"
        }, message="上传成功，已加入处理队列")

    except Exception as e:
        logger.exception("Upload process failed")
        return error(f"上传处理失败: {str(e)}", code=500)