from fastapi import FastAPI
from pydantic import BaseModel
from vision.cpp import ImageInference
from typing import List, Optional, Dict, Any

app = FastAPI()

# Load the vision model once on startup
model = ImageInference()

class InferenceRequest(BaseModel):
    prompt: str
    system_prompt: str 
    images: List[str]
    expected_json_schema: Optional[Dict[str, Any]] = None 

class InferenceResponse(BaseModel):
    result: dict | None

@app.post("/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest):
    result = model.prompt(req.prompt, req.system_prompt, req.images, req.expected_json_schema)
    return {"result": result}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)