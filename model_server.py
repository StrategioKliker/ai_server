from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from vision.cpp import ImageInference

app = FastAPI()

# Load the vision model once on startup
model = ImageInference()

class InferenceRequest(BaseModel):
    prompt: str
    system_prompt: str 
    images: List[str]

class InferenceResponse(BaseModel):
    result: str | None

@app.post("/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest):
    result = model.prompt(req.prompt, req.images)
    return {"result": result}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)