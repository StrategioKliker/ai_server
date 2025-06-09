from fastapi import FastAPI
from typing import List, Optional, Any, Dict 
from pydantic import BaseModel
from vision import ImageInference
from rq import Queue
from redis import Redis

app = FastAPI()
redis_conn = Redis()
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

    return {"status": "queued", "request_id": task.request_id, "job_id": job.id},

def run_vision_inference(prompt: str, images: List[str], request_id: str):
    print("Running vision inference for request id: ", request_id)
    img_inf = ImageInference('minicpm')
    response = img_inf.prompt(prompt, images)
    print("Completed with response: ", response)



