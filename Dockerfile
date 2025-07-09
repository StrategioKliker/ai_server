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

# Install base deps
RUN pip install --no-cache-dir -r requirements.txt || true
RUN pip install --no-cache-dir uvicorn

# --- Build llama-cpp-python with CUDA ---
RUN git clone https://github.com/abetlen/llama-cpp-python.git && \
    cd llama-cpp-python && \
    pip install . --no-cache-dir \
      --config-settings="--build-option=--force-cuda" \
      --config-settings="--build-option=--extra-cmake-args=-DLLAMA_CUBLAS=on"

# Confirm if using gpu
RUN python -c "from llama_cpp import Llama; print('🔥 CUDA Build:', 'n_gpu_layers' in Llama.__init__.__code__.co_varnames)"

# Copy code and model
COPY minicpm/mmproj-model-f16.gguf /app/minicpm/mmproj-model-f16.gguf
COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
