FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install Python 3.10 and system tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3.10-distutils \
    build-essential cmake git curl wget && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy precompiled requirements
COPY requirements.txt .

# Install deps (from compiled file + custom index)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
      llama-cpp-python==0.3.9 \
      uvicorn \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Copy code and model
COPY minicpm/mmproj-model-f16.gguf /app/minicpm/mmproj-model-f16.gguf
COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
