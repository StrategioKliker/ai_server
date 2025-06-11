import os
import requests
from datetime import datetime
from typing import List, Union
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler

print("Loading MiniCPM model with multimodal support…")

#  Ensure projector exists (correct GGUF file)
projector_dir = "minicpm"
projector_path = os.path.join(projector_dir, "mmproj-model-f16.gguf")
os.makedirs(projector_dir, exist_ok=True)

if not os.path.isfile(projector_path):
    print("Downloading correct projector model...")
    download_url = (
        "https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/"
        "main/mmproj-model-f16.gguf"
    )
    resp = requests.get(download_url, stream=True)
    chunk = resp.raw.read(20, decode_content=False)
    if resp.status_code != 200 or b"<html" in chunk:
        raise RuntimeError("Failed to fetch valid binary projector (.gguf).")
    with open(projector_path, "wb") as f:
        f.write(chunk)  # write the initial part
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("✅ Projector downloaded:", projector_path)

#  Setup chat handler and model
chat_handler = MiniCPMv26ChatHandler(clip_model_path=projector_path)


# For a list of available quantized models see 
# -> https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf?library=llama-cpp-python 

# Tested 
# -> ggml-model-Q2_K.gguf -> 2-bit , okay response, not too heavy on machine
# -> ggml-model-Q4_K.gguf -> 4-bit -> okay respons, heavier on machine 
llm = Llama.from_pretrained(
    repo_id="openbmb/MiniCPM-V-2_6-gguf",
    filename="ggml-model-Q2_K.gguf",
    chat_handler=chat_handler,
    n_ctx=2048, 
    n_gpu_layers=-1,
    main_gpu=0,     
    gpu_mlock=True  
)

class ImageInference:
    def __init__(self):
        self.elapsed_minutes = 0

    def prompt(self, prompt: str, images: List[str]) -> Union[str, None]:
        if not prompt or not isinstance(prompt, str) or not images:
            print("Invalid prompt or images")
            return None

        # 👇 Build structured multimodal content
        content = [{"type": "text", "text": "Respond in English. " + prompt}]

        for img in images:
            if os.path.isfile(img):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"file://{os.path.abspath(img)}"}
                })
            elif img.startswith(("http://", "https://")):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img}
                })
            else:
                print("Skipping unsupported image:", img)

        # 👇 Run the model
        start = datetime.now()
        try:
            res = llm.create_chat_completion(
                messages=[
                    {"role": "user", "content": content}
                ], 
                response_format={
                    "type": "json_object",
                }, 
            )
            self.elapsed_minutes = (datetime.now() - start).total_seconds() / 60
            return res["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print("LLM inference error:", e)
            return None

# 🧪 Local test
if __name__ == "__main__":
    
    prompt = """
        You are a visual analysis assistant. Analyze the provided image carefully and extract the following information in strict JSON format. Do not include any explanation or extra text. Only return a well-formatted JSON object.

        TASKS:
        1. Determine if there is a bottle of ink in the image.
        2. Count how many bottles are present.
        3. Identify the color(s) of the bottles.
        4. Detect the printer brand, if any printer is visible.
        5. Extract any text that is visible in the image.

        JSON OUTPUT FORMAT:
        {
            "ink_bottle_present": true,
            "bottle_count": 3,
            "bottle_colors": ["cyan", "magenta", "yellow"],
            "printer_brand": "Canon",
            "extracted_text": "PIXMA G2020"
        }
    """

    out = ImageInference().prompt(prompt, ["images/printer.jpg"])
    print("Result:", out)

