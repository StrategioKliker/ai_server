import os
from fastapi import FastAPI
from typing import List, Optional, Any, Dict 
from prometheus_fastapi_instrumentator  import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
from pydantic import BaseModel
from vision import ImageInference
from rq import Queue
from redis import Redis

# Start FastAPI app 
app = FastAPI()

# Connect prometheus to track FastAPI app and establish a /metrics endpoint
Instrumentator().instrument(app).expose(app)

# Custom Prometheus metrics 
visual_inference_jobs_created_count = Counter("visual_inference_jobs_created_count", "Total number of inference jobs created")
visual_inference_duration_in_seconds = Histogram("visual_inference_duration_in_seconds", "Duration of vision inference jobs in seconds")
queue_size_gauge = Gauge("queue_size_gauge", "Current size of the inference job queue")
visual_inference_failure_count = Counter("visual_inference_failure_count", "Total number of inference jobs failed")

# Establish redis connection and access the queue 
redis_conn = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), socket_timeout=None, retry_on_timeout=True)
queue = Queue(connection=redis_conn)

class VisionTaskRequest(BaseModel):
    images: List[str]
    request_id: str 
    prompt: str 
    metadata: Optional[Dict[str, Any]] = None 

@app.post("/inference/new_vision_task")
def new_vision_task(task: VisionTaskRequest):
    print(f"Submitting new task: {task.request_id}")
    job = queue.enqueue(run_vision_inference, task.prompt, task.images, task.request_id)

    visual_inference_jobs_created_count.inc()
    queue_size_gauge.set(queue.count)
    return {"status": "queued", "request_id": task.request_id, "job_id": job.id},

def run_vision_inference(prompt: str, images: List[str], request_id: str):
    queue_size_gauge.set(queue.count)
    print("Running vision inference for request id: ", request_id)
    img_inf = ImageInference('minicpm')
    
    try: 
        with visual_inference_duration_in_seconds.time():
            response = img_inf.prompt(prompt, images)
        print("Completed with response: ", response)
    except Exception as e: 
        visual_inference_failure_count.inc()
        print("Inference failed: ", e)
        raise



