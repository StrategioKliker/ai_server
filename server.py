import os
import env 
import dns
import json
import socket 
import requests
import traceback
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
    token: str 
    task_id: str 
    system_prompt: Union[str, None]
    prompt: str 
    expected_json_schema: Optional[Dict[str, str]] = None 

@app.post("/inference/new_vision_task")
def new_vision_task(task: VisionTaskRequest):
    if task.token != env.SERVER_TOKEN: 
        print("Invalid access token")
        return {"status": "denied"}

    print(f"Submitting new task: {task.task_id}", flush=True)
    job = queue.enqueue(run_vision_inference, task.prompt, task.system_prompt, task.images, task.task_id, task.expected_json_schema, job_timeout=600)

    queue_size_gauge.set(queue.count)  # update gauge here when job is queued

    return {"status": "queued", "task_id": task.task_id, "job_id": job.id}

@app.get("/ping")
def ping():
    return {"ping": "pong"}


_model_server_ip = None 
def _get_model_server_url() -> str: 
    env_url = os.getenv("MODEL_SERVER_URL")
    if env_url:
        return env_url

    global _model_server_ip
    if _model_server_ip is None:
        _model_server_ip = dns.resolve('model-server')

    return f"http://localhost:8001/infer"


def run_vision_inference(prompt, system_prompt, images, task_id, expected_json_schema):
    print("Running vision inference for request id:", task_id, flush=True)
    # Ping local IP instead of spamming docker DNS 
    model_server_url = _get_model_server_url()

    inference_attempts = 3
    json_result = None
    try:
        while inference_attempts > 0:
            response = None
            with visual_inference_duration_in_seconds.time():
                try:
                    session = requests.Session()

                    res = session.post(
                        model_server_url,
                        json={"prompt": prompt, "images": images, "system_prompt": system_prompt, "expected_json_schema": expected_json_schema},
                        timeout=600,
                    )
                    res.raise_for_status()
                    response = res.json().get("result")
                except Exception as e:
                    print("Model server request failed:", e, flush=True)
                    print("Stack: ", traceback.format_exc(), flush=True)
                    response = None
                print("Completed with response:", response, flush=True)


            if not response:
                inference_attempts -= 1
                print(f"Inference failed, attempts left: {inference_attempts}", flush=True)
                sleep(5)
                continue

            json_result = response
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
    # For now 
    if error: 
        return 
    
    print("Task id: ", task_id, flush=True)
    print("Sending back: ", result, flush=True)
    print("Error: ", error, flush=True)

    result_url = env.MANAGER_API + env.SEND_PROMPT_RESULT_ROUTE
    # Symfony backend onyl accepts formdata so we must send it like this
    payload = {
        "token": env.PROMPT_TOKEN,
        "result_json": json.dumps({
            "task_id": task_id,
            "prompt_result": result
        })
    }

    res = requests.post(result_url, data=payload)

    print("Server saving result responded: ", res.content, flush=True )


