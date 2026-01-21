# 这是知识库工程的 readme 文档
目前用来初始化主分支

```python
# 激活虚拟环境：
venv\Scripts\Activate.ps1

```
启动celery异步任务队列
celery -A tasks.celery_app worker --loglevel=info -P solo -n worker1@%hostname%