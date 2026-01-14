# 文件: /mnt/omicshub/rag/utils/logger.py

import sys
from pathlib import Path
from loguru import logger
from config import settings

# 移除默认handler
logger.remove()

# 定义日志格式
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

# 1. 控制台输出
logger.add(
    sys.stdout,
    format=LOG_FORMAT,
    level=settings.LOG_LEVEL,
    colorize=True
)

# 2. 确保日志目录存在
log_dir = Path(settings.LOG_DIR)
log_dir.mkdir(exist_ok=True)

# 3. 文件输出 (App日志)
logger.add(
    log_dir / "app.log",
    rotation="100 MB",
    retention="30 days",
    compression="zip",
    encoding="utf-8",
    level="INFO",
    enqueue=True
)

# 4. 文件输出 (错误日志)
logger.add(
    log_dir / "error.log",
    rotation="50 MB",
    retention="60 days",
    encoding="utf-8",
    level="ERROR",
    enqueue=True
)

# 导出 logger
__all__ = ["logger"]