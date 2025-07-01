# AI Server Overview

This project exposes a FastAPI service that performs visual inference using
local large language models served through Ollama. Jobs are queued in Redis and
processed asynchronously by an RQ worker. Monitoring is provided via Prometheus
and Grafana.

## Directory Layout

```
.
├── Dockerfile              # Runtime image for the FastAPI app
├── Dockerfile.ollama       # Image to preload the vision model
├── docker-compose.yml      # Development stack (app, worker, redis, ollama, monitoring)
├── grafana/                # Grafana provisioning files
├── images/                 # Sample images for tests/examples
├── jsonz/                  # Helpers for JSON extraction and validation
├── nginx/                  # Example reverse proxy config
├── protoypes/              # Experimental scripts for model usage
├── server.py               # FastAPI application with `/inference/new_vision_task`
├── vision.py               # `ImageInference` class communicating with Ollama
└── worker.py               # RQ worker consuming queued jobs
```

## Application Components

### FastAPI Server
- Defined in `server.py`.
- Exposes a single POST endpoint `/inference/new_vision_task`.
- Incoming requests are enqueued for background processing.
- Custom Prometheus metrics track queue size, job duration and failures.

### Worker
- `worker.py` connects to Redis and continuously processes queued jobs.
- Each job calls `run_vision_inference` which uses `ImageInference` to contact the Ollama model.

### Vision Model Interface
- Implemented in `vision.py`.
- Encodes images to Base64 and sends them to the configured Ollama endpoint.
- Currently supports the `minicpm-v` model by default.

### JSON Utilities
- The `jsonz/` package provides `extract_json_from_str` and `is_valid_json`.
- Useful for parsing and validating structured data returned by the model.

### Monitoring Stack
- `prometheus.yml` configures Prometheus scrapes for the app, the worker
  container, cadvisor, node exporter, and GPU metrics.
- `grafana/` contains datasources and a sample dashboard that visualises
  the custom metrics.

## Running Locally

1. Install dependencies (either with [uv](https://github.com/astral-sh/uv) or pip):
   ```bash
   uv pip install -r pyproject.toml
   ```
   or
   ```bash
   pip install -r requirements.txt
   ```
2. Start Redis and Ollama (via Docker Compose for convenience):
   ```bash
   docker-compose up --build
   ```
   The FastAPI API is then accessible on `http://localhost:8000` and metrics at
   `http://localhost:8000/metrics`.

3. Submit a task:
   ```bash
   curl -X POST http://localhost:8000/inference/new_vision_task \
        -H 'Content-Type: application/json' \
        -d '{"images": ["images/printer.jpg"], "request_id": "1", "prompt": "Describe"}'
   ```

## Testing

A small script `tests.py` demonstrates calls to the `ImageInference` class. After
dependencies are installed it can be executed with:

```bash
python tests.py
```

(Some tests may fail because they rely on model outputs.)



docker build -f Dockerfile.llama -t llama-cpp:cuda .
