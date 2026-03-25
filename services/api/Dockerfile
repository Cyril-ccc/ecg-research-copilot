FROM python:3.11-slim

WORKDIR /app

# 复制源码
COPY . /app

# 安装 uv 并安装依赖
RUN pip install --no-cache-dir uv \
    && uv sync --frozen || uv sync

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
