import os
from inference import run_vision_inference
from prometheus_client import Gauge
from rq import Queue
from time import sleep 
from redis import Redis
from fastapi import FastAPI
from pydantic import BaseModel
from jsonz.validator import is_valid_json
from typing import List, Optional, Any, Dict, Union 
from jsonz.extractor import extract_json_from_str
from prometheus_fastapi_instrumentator  import Instrumentator


# Start FastAPI app 
app = FastAPI()

# Connect prometheus to track FastAPI app and establish a /metrics endpoint
Instrumentator().instrument(app).expose(app)
queue_size_gauge = Gauge("queue_size_gauge", "Current size of the inference job queue")


# Establish redis connection and access the queue 
redis_conn = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), socket_timeout=None, retry_on_timeout=True)
queue = Queue(connection=redis_conn)

class VisionTaskRequest(BaseModel):
    images: List[str]
    request_id: str 
    prompt: str 
    metadata: Optional[Dict[str, Any]] = None 
    expected_json_schema: Optional[Dict[str, str]] = None 

@app.post("/inference/new_vision_task")
def new_vision_task(task: VisionTaskRequest):
    print(f"Submitting new task: {task.request_id}", flush=True)
    job = queue.enqueue(run_vision_inference, task.prompt, task.images, task.request_id, task.expected_json_schema, job_timeout=300)

    queue_size_gauge.set(queue.count)  # update gauge here when job is queued

    return {"status": "queued", "request_id": task.request_id, "job_id": job.id}


