# 文件: /mnt/omicshub/rag/tasks/celery_app.py

from celery import Celery
from config import settings

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
)

if __name__ == "__main__":
    celery_app.start()