#!/usr/bin/env bash
# Bring up the F16 model server on host so Docker containers can hit it:
uvicorn model_server:app --host 0.0.0.0 --port 8001