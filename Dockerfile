FROM python:3.11-slim

# 更新软件包并安装必要的依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev curl \
    && rm -rf /var/lib/apt/lists/*


# 复制工具文件并配置
COPY ./tool /app
WORKDIR /app

# 移动并授权 uv 和 uvx
RUN mv ./uv-x86_64-unknown-linux-gnu/uv /usr/local/bin/uv && \
    chmod +x /usr/local/bin/uv && \
    mv ./uv-x86_64-unknown-linux-gnu/uvx /usr/local/bin/uvx && \
    chmod +x /usr/local/bin/uvx && \
    rm -rf /app/uv-x86_64-unknown-linux-gnu

# 复制聊天应用文件
COPY ./chatapp /chatapp
WORKDIR /chatapp

# 初始化虚拟环境并安装依赖
RUN rm -rf .venv && uv venv .venv && uv sync

# 使用虚拟环境作为默认环境
ENV PATH="/chatapp/.venv/bin:$PATH"

# 暴露应用端口
EXPOSE 8848

# 启动应用
CMD ["uv", "run", "/chatapp/rag_fusion.py"]