FROM python:3.11-slim

# 更新软件包并安装必要的系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev curl \
    && rm -rf /var/lib/apt/lists/*

# 复制工具文件并配置
COPY ./tool /tool
WORKDIR /

# 移动并授权 uv 和 uvx
RUN mv /tool/uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv && \
    chmod +x /usr/local/bin/uv && \
    mv /tool/uv-x86_64-unknown-linux-gnu/uvx /usr/local/bin/uvx && \
    chmod +x /usr/local/bin/uvx && \
    rm -rf /tool/uv-x86_64-unknown-linux-gnu

# 复制项目文件
COPY ./rag /rag
COPY ./data /data
COPY ./.env /.env

# 设置工作目录
WORKDIR /rag

# 初始化虚拟环境并安装依赖
RUN rm -rf .venv && uv venv .venv && uv sync

# 使用虚拟环境作为默认环境
ENV PATH="/rag/.venv/bin:$PATH"

# 暴露应用端口
EXPOSE 8848

# 启动应用
CMD ["uv", "run", "main.py"]