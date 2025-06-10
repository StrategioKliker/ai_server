import os
import json
import base64
import requests
from datetime import datetime
from typing import List, Union 


class ImageInference: 
    def __init__(self, model):
        # NOTE -> Top performers are qwen and minicpm 
        available_models = {
            # "granite": "granite3.2-vision",
            # "qwen": "qwen2.5vl",
            "minicpm": "minicpm-v",
            # "llava": "llava",
            # "llava-phi": "llava-phi3",
            # "llava-llama3": "llava-llama3",
        }
        
        if model not in available_models:
            raise Exception("Provided model not in available model list")

        self._model = available_models[model]
        self._ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

        self.elapsed_minutes = 0
        
    def __encode_image(self, path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
        
    def __encode_images(self, images):
        encoded_images = []
        for img_path in images:
            try:
                encoded_images.append(self.__encode_image(img_path))
            except FileNotFoundError:
                print(f"Error: Couldn’t find image at '{img_path}'. Double-check the path.")
                exit(1)

        return encoded_images
    
    def __process_images(self, images): 
        pass
            
    def prompt(self, prompt: str, images: List[str]) -> Union[str, None]:
        if not (isinstance(prompt, str)):
            print("Prompt is not of expected type string")
            return 
        
        if not (isinstance(images, list)) or len(list) == 0:
            print("Images is not of expected type list")
            return 

        prompt = "Respond in English. " + prompt

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "images": self.__encode_images(images)
        }
        # If this errors out, check that Ollama is actually running on port 11434.
        start_time = datetime.now()
        response = requests.post(self._ollama_url, json=payload)
        self.elapsed_minutes = round((datetime.now() - start_time).total_seconds() / 60, 4)

        # print(response.text
        # Don’t blindly trust the API—inspect the response.
        if response.status_code != 200:
            raise Exception(f"Ollama API returned {response.status_code}: {response.text}")
        
        data = json.loads(response.text)
        # Ollama’s generate endpoint always returns something under "response"
        return data.get("response", "").strip()
