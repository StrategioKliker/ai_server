import base64
import os
import sys
import imghdr
import hashlib
import requests
import filetype 
from datetime import datetime
from llama_cpp import Llama
from typing import List, Union
from urllib.parse import urlparse, unquote
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler


# Lamma cpp has too many logs, it prints too much and floods the 
class suppress_stdout(object):
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup    = sys.stdout.fileno()
        self.old_stderr_fileno_undup    = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup ( sys.stdout.fileno() )
        self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2 ( self.outnull_file.fileno(), self.old_stdout_fileno_undup )
        os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )

        sys.stdout = self.outnull_file        
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):        
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2 ( self.old_stdout_fileno, self.old_stdout_fileno_undup )
        os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )

        os.close ( self.old_stdout_fileno )
        os.close ( self.old_stderr_fileno )

        self.outnull_file.close()
        self.errnull_file.close()


class ImageInference:
    def __init__(self):
        self.elapsed_minutes = 0

        self.image_dir = 'images'
        os.makedirs(self.image_dir, exist_ok=True)

        print("Loading MiniCPM model with multimodal support…", flush=True)
        #  Ensure projector exists (correct GGUF file)
        projector_dir = "minicpm"
        self._projector_path = os.path.join(projector_dir, "mmproj-model-f16.gguf")
        os.makedirs(projector_dir, exist_ok=True)

        with suppress_stdout():
            #  Setup chat handler and model
            chat_handler = MiniCPMv26ChatHandler(clip_model_path=self._projector_path)

            # For a list of available quantized models see 
            # -> https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf?library=llama-cpp-python 

            # Tested 
            # -> ggml-model-Q2_K.gguf -> 2-bit , okay response, not too heavy on machine
            # -> ggml-model-Q4_K.gguf -> 4-bit -> okay respons, heavier on machine 
            self.llm = Llama.from_pretrained(
                repo_id="openbmb/MiniCPM-V-2_6-gguf",
                filename="ggml-model-Q4_K.gguf",
                chat_handler=chat_handler,
                n_ctx=2048, 
                n_gpu_layers=32,
                main_gpu=0,     
                gpu_mlock=True  
            )


    def __get_image_filename(self, img_url: str, img_content: bytes) -> Union[str, None]:
        img_path = urlparse(img_url).path
        img_url_base = unquote(img_path.rsplit('/', 1)[-1])
        img_url_base = img_url_base or ''
        img_url_base = img_url_base.rsplit('.', 1)[0]
        if not img_url_base or len(img_url_base) < 3: 
            img_url_base = f"img_{hashlib.sha256(img_content).hexdigest()[:12]}"


        common_suffix = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".svg", ".ico", ".heic", ".avif", ".jfif", ".apng" )
        for suffix in common_suffix: 
            if img_url.lower().endswith(suffix): 
                suffix = suffix.lstrip('.')
                return f"{img_url_base}.{suffix}"

        img_kind = imghdr.what(None, img_content)
        if img_kind:
            return f"{img_url_base}.{img_kind}"
        
        img_info = filetype.guess(img_content)
        if img_info: 
            return f"{img_url_base}.{img_info.extension}"
        
        return None 

    def __save_image(self, img_url: str) -> Union[str, None]:
        res = requests.get(img_url, timeout=60)
        if res is None or res.status_code != 200: 
            print("Failed to get image on url: ", img_url, " with response: ", res.status_code)
            return None 
        
        image_filename = self.__get_image_filename(img_url, res.content)   
        if not image_filename: 
            print("Failed to find image filename")
            return None 
        
        local_path = os.path.join(self.image_dir, image_filename)
        try: 
            with open(local_path, 'wb') as f: 
                f.write(res.content)
            return f"file://{os.path.abspath(local_path)}"
        except: 
            return None 
        

    def __get_image_base64_data_from_url(self, image_url: str) -> str:
        res = requests.get(image_url, timeout=10)
        if res.status_code != 200:
            raise Exception(f"Failed to fetch image: {res.status_code}")
        
        mime = res.headers.get("Content-Type", "image/png")
        base64_data = base64.b64encode(res.content).decode('utf-8')
        return f"data:{mime};base64,{base64_data}"


    def __process_image_content(self, images, content):
        image_found = False 
        for img in images:
            img_data = None 

            if os.path.isfile(img):
                img_data = f"file://{os.path.abspath(img)}"    
            elif img.startswith(("http://", "https://")):
                img_data = self.__get_image_base64_data_from_url(img)
            
            if img_data is None: 
                print("Skipping unsupported image:", img, flush=True)
                continue
            
            print("Reading image from path: ", img_data)
            content.append({
                "type": "image_url",
                "image_url": {"url": img_data}
            })
            image_found = True 
            
    
        if not image_found: 
            return None         


    def prompt(self, prompt: str, system_prompt: Union[str, None], images: List[str]) -> Union[str, None]:
        print("Running prompt", flush=True)
        if not prompt or not isinstance(prompt, str) or not images:
            print("Invalid prompt or images", flush=True)
            return None

        # Build structured multimodal content
        content = [{"type": "text", "text": "Respond in English. " + prompt}]

        found_images = self.__process_image_content(images=images, content=content)
        if not found_images: 
            print("No valid images")
            return None         

        # Run the model
        start = datetime.now()
        messages=[
            {"role": "user", "content": content},
        ] 

        if system_prompt: 
            messages.insert(0, {
                'role': 'system', 'content': system_prompt
            })

        try:
            res = self.llm.create_chat_completion(
                messages = messages,
                response_format={
                    "type": "json_object",
                }, 
            )
            self.elapsed_minutes = (datetime.now() - start).total_seconds() / 60
            print("Result in cpp: ", res["choices"][0]["message"]["content"])
            return res["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print("LLM inference error:", e, flush=True)
            return None
