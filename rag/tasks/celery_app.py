# 文件: /mnt/omicshub/rag/tasks/celery_app.py

from celery import Celery
from config import settings
import logging
import os

# 确保日志目录存在
if not os.path.exists(settings.LOG_DIR):
    os.makedirs(settings.LOG_DIR, exist_ok=True)

celery_app = Celery(
    "rag_tasks",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "tasks.document_tasks",
        "tasks.document_tasks_async"
    ]  # 注册任务模块
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    # 任务路由（可选）
    # task_routes={
    #     'tasks.document_tasks.*': {'queue': 'rag_docs'},
    # },
    # 日志配置
    worker_log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    worker_task_log_format='%(asctime)s - %(name)s - %(levelname)s - [%(task_name)s(%(task_id)s)] - %(message)s',
    worker_log_color=True,
    worker_redirect_stdouts=False,
    worker_stdouts_level=settings.LOG_LEVEL,
    worker_stderr_level='ERROR',
)

# 配置日志器
logger = logging.getLogger('celery')
logger.setLevel(getattr(logging, settings.LOG_LEVEL))

# 确保控制台日志输出
if not logger.handlers:
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 添加文件处理器
    file_handler = logging.FileHandler(os.path.join(settings.LOG_DIR, 'celery.log'))
    file_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

if __name__ == "__main__":
    celery_app.start()