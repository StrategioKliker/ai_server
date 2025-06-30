FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install Python 3.10 and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Get uv
COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /uvx /bin/

# Set working dir
WORKDIR /app

# Build llama-cpp-python with CUDA support
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=61"
ENV FORCE_CMAKE=1
ENV LLAMA_CPP_LOG_LEVEL=error

ENV UV_HTTP_TIMEOUT=600
ENV UV_RESOLVE_PREFER_BINARY=1
COPY pyproject.toml uv.lock ./
RUN uv sync --locked

COPY minicpm/mmproj-model-f16.gguf /app/minicpm/mmproj-model-f16.gguf
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
