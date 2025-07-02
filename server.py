import os

import requests
from inference import run_vision_inference
from rq import Queue
from time import sleep 
from redis import Redis
from fastapi import FastAPI
from pydantic import BaseModel
from jsonz.validator import is_valid_json
from typing import List, Optional, Any, Dict, Union 
from jsonz.extractor import extract_json_from_str
from prometheus_fastapi_instrumentator  import Instrumentator
from prometheus_client import Counter, Histogram, Gauge


# Start FastAPI app 
app = FastAPI()

# Connect prometheus to track FastAPI app and establish a /metrics endpoint
Instrumentator().instrument(app).expose(app)
queue_size_gauge = Gauge("queue_size_gauge", "Current size of the inference job queue")
visual_inference_duration_in_seconds = Histogram("visual_inference_duration_in_seconds", "Duration of vision inference jobs in seconds")
visual_inference_failure_count = Counter("visual_inference_failure_count", "Total number of inference jobs failed")

# Establish redis connection and access the queue 
redis_conn = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), socket_timeout=None, retry_on_timeout=True)
queue = Queue(connection=redis_conn)

class VisionTaskRequest(BaseModel):
    images: List[str]
    task_id: str 
    system_prompt: Union[str, None]
    prompt: str 
    metadata: Optional[Dict[str, Any]] = None 
    expected_json_schema: Optional[Dict[str, str]] = None 

@app.post("/inference/new_vision_task")
def new_vision_task(task: VisionTaskRequest):
    print(f"Submitting new task: {task.task_id}", flush=True)
    job = queue.enqueue(run_vision_inference, task.prompt, task.system_prompt, task.images, task.task_id, task.expected_json_schema, job_timeout=600)

    queue_size_gauge.set(queue.count)  # update gauge here when job is queued

    return {"status": "queued", "task_id": task.task_id, "job_id": job.id}


MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8001/infer")

def run_vision_inference(prompt, system_prompt, images, task_id, expected_json_schema):
    print("Running vision inference for request id:", task_id, flush=True)

    inference_attempts = 3
    json_result = None
    try:
        while inference_attempts > 0:
            response = None
            with visual_inference_duration_in_seconds.time():
                try:
                    res = requests.post(
                        MODEL_SERVER_URL,
                        json={"prompt": prompt, "images": images, "system_prompt": system_prompt},
                        timeout=600,
                    )
                    res.raise_for_status()
                    response = res.json().get("result")
                except Exception as e:
                    print("Model server request failed:", e, flush=True)
                    response = None
                print("Completed with response:", response, flush=True)


            if not response:
                inference_attempts -= 1
                print(f"Inference failed, attempts left: {inference_attempts}", flush=True)
                sleep(5)
                continue

            extracted_json = extract_json_from_str(response)
            if not extracted_json:
                inference_attempts -= 1
                print(f"Failed to extract JSON, attempts left: {inference_attempts}", flush=True)
                sleep(5)
                continue

            if expected_json_schema and not is_valid_json(expected_json_schema, extracted_json):
                inference_attempts -= 1
                print(f"JSON validation failed, attempts left: {inference_attempts}", flush=True)
                sleep(5)
                continue

            json_result = extracted_json
            break

        if json_result is None:
            visual_inference_failure_count.inc()
            print(f"Inference failed for request: {task_id}", flush=True)
            send_prompt_task_result(task_id, None, f"Inference failed for request: {task_id}")
            return {"success": False, "reason": f"Inference failed for request: {task_id}"}

        print("Extracted:", json_result, flush=True)
        send_prompt_task_result(task_id, json_result)
        return {"success": True, "result": json_result}

    except Exception as e:
        visual_inference_failure_count.inc()
        print("Inference failed:", e, flush=True)
        send_prompt_task_result(task_id, json_result, f"Inference failed for request: {task_id}")
        return {"success": False, "reason": "Inference failed"}
    

def send_prompt_task_result(task_id, result, error = None):
    print("Task id: ", task_id)
    print("Sending back: ", result)
    print("Error: ", error)
