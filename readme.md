# AI Server Overview

This repository contains a small FastAPI application for running computer vision tasks using locally hosted large language models. Images are processed asynchronously via an RQ worker and the results are returned in JSON. The stack is containerised and includes monitoring with Prometheus and Grafana. Two model interfaces are provided:

- **Ollama** (`vision/ollama.py`) for talking to an external Ollama instance
- **llama-cpp** (`vision/cpp.py`) to run the MiniCPM model locally via GPU

Jobs are queued in Redis and metrics are exported for observability.

## Installation

## NVIDIA GPU Setup Guide for Docker Runtime (Ubuntu)

This guide assumes Ubuntu 20.04+ but can be adapted to similar distros.

---

### 1. Verify your GPU and driver status

Open terminal and run:

    nvidia-smi

- If it shows your GPU info and driver version, NVIDIA drivers are installed correctly.
- If `command not found` or errors, install drivers below.

---

### 2. Install NVIDIA Drivers (if missing)

Add the graphics drivers PPA:

    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt update

Check recommended driver version:

    ubuntu-drivers devices

Install recommended driver (example):

    sudo apt install nvidia-driver-525

Reboot after install:

    sudo reboot

Verify with `nvidia-smi` again.

---

### 3. Install Docker (if not installed)

If Docker is not yet installed:

    sudo apt update
    sudo apt install -y ca-certificates curl gnupg lsb-release

    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io

Verify Docker:

    sudo docker run hello-world

---

### 4. Install NVIDIA Container Toolkit for GPU support in Docker

Add the NVIDIA Docker repo:

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt update

Install NVIDIA Docker runtime:

    sudo apt install -y nvidia-docker2

Restart Docker daemon:

    sudo systemctl restart docker

---

### 5. Test GPU support in Docker

Run a test container with GPU access:

    sudo docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi

You should see your GPU info inside the container output.

---

### 6. Troubleshooting

- If `nvidia-smi` inside container fails, double-check driver installation on host.
- Ensure user is in `docker` group or use `sudo` for docker commands.
- Validate `nvidia-container-runtime` is installed:

    docker info | grep -i runtime

Should list `nvidia`.

---

This should get your CUDA GPU properly exposed to Docker containers so your llama-cpp based models can run efficiently.

---


### Requirements

- Python 3.12+
- Docker and Docker Compose
- Redis
- (Optional) NVIDIA GPU for the CUDA based image model

### First-time setup

```bash
# clone the repository and enter it
git clone https://github.com/your-org/ai_server.git
cd ai_server

# create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install uv (optional but recommended)
pip install uv

# compile a fresh requirements.txt from pyproject.toml (uses uv)
./compile.sh

# install all Python dependencies
uv pip install -r pyproject.toml
# or use plain pip
pip install -r requirements.txt

# download the MiniCPM projection model and verify the GGUF path
python vision/cpp.py
```
The `requirements.txt` file is generated from `pyproject.toml` using
`./compile.sh`. Running that script ensures the lock file and requirements are in
sync.

Running `python vision/cpp.py` for the first time downloads the `mmproj-model-f16.gguf`
projector into the `minicpm/` folder so the model can run locally.

To run the full stack (app, worker, Redis and monitoring) you can use Docker Compose:

```bash
docker-compose up --build
```

## Usage

1. Start the API (either locally with `uvicorn server:app` or via Docker Compose).
2. Submit a vision task using the `/inference/new_vision_task` endpoint:

```bash
curl -X POST http://localhost:8000/inference/new_vision_task \
     -H 'Content-Type: application/json' \
     -d '{"images": ["images/printer.jpg"], "request_id": "1", "prompt": "Describe"}'
```

The request is queued and processed by the worker. Results are returned in JSON. Metrics are available at `http://localhost:8000/metrics`.

Run the simple test script after installing dependencies:

```bash
python tests.py
```

## Configuration

The following environment variables can be used to configure the service:

| Variable      | Default                       | Description                              |
|---------------|-------------------------------|------------------------------------------|
| `REDIS_URL`   | `redis://localhost:6379/0`    | Redis connection string                  |
| `OLLAMA_URL`  | `http://localhost:11434/api/generate` | Ollama API endpoint                |
| `LLAMA_CPP_LOG_LEVEL` | `error`                | Log level when running llama-cpp models  |

When using Docker Compose these values are set automatically.

## Folder Structure

```
.
├── Dockerfile            # Production image running the FastAPI app
├── Dockerfile.ollama     # Image that preloads the MiniCPM model for Ollama
├── docker-compose.yml    # Development stack with monitoring
├── grafana/              # Grafana dashboards and data source config
├── images/               # Example images used in tests
├── jsonz/                # JSON extraction and validation utilities
├── nginx/                # Example reverse proxy configuration
├── protoypes/            # Stand‑alone scripts demonstrating model usage
├── compile.sh            # Generate requirements.txt via uv
├── vision/               # Model interface implementations
│   ├── cpp.py            # llama-cpp based MiniCPM runner
│   └── ollama.py         # Ollama API helper
├── server.py             # FastAPI application
├── worker.py             # RQ worker processing queued jobs
└── tests.py              # Basic vision inference tests
```

## Technologies Used

- **Python** – FastAPI, Pydantic, RQ, Redis, Requests
- **Prometheus** and **Grafana** for metrics and dashboards
- **Docker** / **Docker Compose** for local deployment
- **llama-cpp-python** and **Ollama** for running multimodal LLMs

---
This project provides a minimal setup for experimenting with local vision LLMs and can serve as a template for more advanced computer vision backends.