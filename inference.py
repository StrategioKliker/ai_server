# inference.py
import os
import requests
from jsonz.validator import is_valid_json
from jsonz.extractor import extract_json_from_str
from prometheus_client import Counter, Histogram
from time import sleep 

# Metrics
visual_inference_duration_in_seconds = Histogram("visual_inference_duration_in_seconds", "Duration of vision inference jobs in seconds")
visual_inference_failure_count = Counter("visual_inference_failure_count", "Total number of inference jobs failed")

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8001/infer")

def run_vision_inference(prompt, images, request_id, expected_json_schema):
    print("Running vision inference for request id:", request_id, flush=True)

    inference_attempts = 3
    json_result = None
    try:
        while inference_attempts > 0:
            response = None
            with visual_inference_duration_in_seconds.time():
                try:
                    res = requests.post(
                        MODEL_SERVER_URL,
                        json={"prompt": prompt, "images": images},
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
            print(f"Inference failed for request: {request_id}", flush=True)
            return {"success": False, "reason": f"Inference failed for request: {request_id}"}

        print("Extracted:", json_result, flush=True)
        return {"success": True, "result": json_result}

    except Exception as e:
        visual_inference_failure_count.inc()
        print("Inference failed:", e, flush=True)
        return {"success": False, "reason": "Inference failed"}
