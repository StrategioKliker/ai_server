import os
from rq import Queue
from time import sleep 
from redis import Redis
from fastapi import FastAPI
from pydantic import BaseModel
from vision.cpp import ImageInference
from jsonz.validator import is_valid_json
from typing import List, Optional, Any, Dict, Union 
from jsonz.extractor import extract_json_from_str
from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator  import Instrumentator

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
    expected_json_schema: Optional[Dict[str, str]] = None 

@app.post("/inference/new_vision_task")
def new_vision_task(task: VisionTaskRequest):
    print(f"Submitting new task: {task.request_id}", flush=True)
    job = queue.enqueue(run_vision_inference, task.prompt, task.images, task.request_id, task.expected_json_schema, job_timeout=300)

    visual_inference_jobs_created_count.inc()
    queue_size_gauge.set(queue.count)
    return {"status": "queued", "request_id": task.request_id, "job_id": job.id},

def run_vision_inference(prompt: str, images: List[str], request_id: str, expected_json_schema: Union[List[str], None]):
    queue_size_gauge.set(queue.count)
    print("Running vision inference for request id: ", request_id, flush=True)
    img_inf = ImageInference()

    inference_attemps = 3    
    try: 
        json_result = None 
        while inference_attemps > 0:
            response = None 
            with visual_inference_duration_in_seconds.time():
                response = img_inf.prompt(prompt, images)
                print("Completed with response: ", response, flush=True)

            if not response: 
                inference_attemps -= 1
                print("Inference failed to get response, retrying with attempts remaining: ", {inference_attemps}, flush=True)
                sleep(5)
                continue
            
            extracted_json = extract_json_from_str(response)
            if not extracted_json:
                inference_attemps -= 1
                print("Failed to extract json from string, retrying with attempts remaining: ", {inference_attemps}, flush=True)
                sleep(5)
                continue

            if expected_json_schema and not is_valid_json(expected_json_schema, extracted_json):
                inference_attemps -= 1
                print("JSON validation failed, retrying with attempts remaining: ", {inference_attemps}, flush=True)
                sleep(5)
                continue

            json_result = extracted_json
            break
        
        if json_result is None: 
            visual_inference_failure_count.inc()
            print("Inference failed for request: ", request_id, flush=True)
            return {"success": False, "reason": f"Inference failed for request: {request_id}"}
        
        print("Extracted: ", extracted_json, flush=True)
        return {"success": True, "result": json_result}
        
    except Exception as e: 
        visual_inference_failure_count.inc()
        print("Inference failed: ", e, flush=True)
        return {"success": False, "reason": "Inference failed"}




