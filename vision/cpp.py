
import io
import os
import sys
import base64
import imghdr
import hashlib
import requests
import filetype 

from PIL import Image 
from time import sleep
from llama_cpp import Llama
from datetime import datetime
from typing import List, Union
from jsonz.validator import is_valid_json
from urllib.parse import urlparse, unquote
from jsonz.extractor import extract_json_from_str
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
        minicpm_dir = "minicpm"
        self._projector_path = os.path.join(minicpm_dir, "mmproj-model-f16.gguf")
        os.makedirs(minicpm_dir, exist_ok=True)

        with suppress_stdout():
            #  Setup chat handler and model
            chat_handler = MiniCPMv26ChatHandler(clip_model_path=self._projector_path)

            # For a list of available quantized models see 
            # -> https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf?library=llama-cpp-python 

            # Tested 
            # -> ggml-model-Q2_K.gguf -> 2-bit , okay response, not too heavy on machine
            # -> ggml-model-Q4_K.gguf -> 4-bit -> okay respons, heavier on machine 
            model_path = os.path.join(minicpm_dir, 'ggml-model-f16.gguf')

            self.llm = Llama(
                # repo_id="openbmb/MiniCPM-V-2_6-gguf",
                # F16 is full blown model
                # filename="ggml-model-f16.gguf",
                #filename="ggml-model-Q6_K.gguf",

                # Switched to local model download 
                model_path=model_path,

                chat_handler=chat_handler,
                n_threads=os.cpu_count(),
                n_ctx=4096, 
                n_gpu_layers=8,
                main_gpu=0,     
                gpu_mlock=False,

                # ↓–– default generation params ↓––
                temperature=0.0,        # pure greedy
                top_p=1.0,              # don’t cut off the distribution
                top_k=1,                # force highest‐prob token
                repeat_penalty=1.9,     # no repeat boosting
                typical_p=1.0,          # no typical sampling
                mirostat_mode=0,        # off (greedy)
                #–– end defaults
            )

        print("Model loaded", flush=True)
        print("GPU used:", self.llm.model_params.n_gpu_layers > 0, flush=True)


    def __get_image_base64_data_from_url(self, image_url: str) -> str:
        res = requests.get(image_url, timeout=10)
        if res.status_code != 200:
            raise Exception(f"Failed to fetch image: {res.status_code}")
        

        # Only supports png images through base64
        img = Image.open(io.BytesIO(res.content)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        base64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:png;base64,{base64_data}"
    
    def __process_image_content(self, images, content):
        image_found = False 
        for img in images:
            img_data = None 

            if os.path.isfile(img):
                img_data = f"file://{os.path.abspath(img)}"    
            elif img.startswith(("http://", "https://")):
                res = requests.get(img, timeout=10)
                res.raise_for_status()

                name_hash = hashlib.md5(img.encode()).hexdigest()
                save_path = os.path.join(self.image_dir, f"{name_hash}.png")                
                with Image.open(io.BytesIO(res.content)).convert("RGB") as im:
                    im.save(save_path, format="PNG")

                img_data = f"file://{os.path.abspath(save_path)}"    
            
            if img_data is None: 
                print("Skipping unsupported image:", img, flush=True)
                continue
            
            # print("Reading image from path: ", img_data)
            content.append({
                "type": "image_url",
                "image_url": {"url": img_data}
            })
            image_found = True 
            
    
        if not image_found: 
            return None   

        return content      


    def prompt(self, prompt: str, system_prompt: Union[str, None], images: List[str], expected_json_schema: dict) -> Union[str, None]:
        print("Running prompt", flush=True)
        if not prompt or not isinstance(prompt, str) or not images:
            print("Invalid prompt or images", flush=True)
            return None

        # Build structured multimodal content
        content = [{"type": "text", "text": "Respond in English. " + prompt}]

        content = self.__process_image_content(images=images, content=content)
        if not content: 
            print("No valid images")
            return None         

        # Run the model
        messages=[
            {"role": "user", "content": content},
        ] 

        if system_prompt: 
            messages.insert(0, {
                'role': 'system', 'content': system_prompt
            })

        return self.__get_inference_result(messages, expected_json_schema)

    def __get_inference_result(self, messages, expected_json_schema, repeat_target=3):
        repeat_count = 0

        repeated_results = []

        repeat_count_target = repeat_target * 3
        while repeat_count < repeat_count_target:
            try:
                start = datetime.now()
                res = self.llm.create_chat_completion(
                    messages = messages,
                    response_format={
                        "type": "json_object",
                    }, 
                )
                self.elapsed_minutes = (datetime.now() - start).total_seconds() / 60
                print("Result in cpp: ", res["choices"][0]["message"]["content"])
                result = res["choices"][0]["message"]["content"].strip()

                extracted_json = extract_json_from_str(result)
                if not extracted_json:
                    repeat_count += 1
                    print(f"Failed to extract JSON, repeat step: {repeat_count}", flush=True)
                    sleep(2)
                    continue

                if expected_json_schema and not is_valid_json(expected_json_schema, extracted_json):
                    repeat_count += 1
                    print(f"JSON validation failed, repeat step: {repeat_count}", flush=True)
                    sleep(2)
                    continue

                repeated_results.append(extracted_json)
                repeat_count += 1      

                if len(repeated_results) >= repeat_target: 
                    break 

            except Exception as e:
                print("LLM inference error:", e, flush=True)
                repeat_count += 1 
                sleep(2)
                continue

        if len(repeated_results) == 0:
            return None 
        
        return self.__reconcile_result(repeated_results)


    def __reconcile_result(self, repeated_results: List[dict]):
        result_value_counter = {}

        print("---------- Reconciling results: ---------")
        for result in repeated_results: 
            print("Result: ", result)
            for key, value in result.items(): 
                if key not in result_value_counter: 
                    result_value_counter[key] = {}
                    
                if value not in result_value_counter[key]:
                    result_value_counter[key][value] = 1
                else:
                    result_value_counter[key][value] += 1
                
        final_result = {}
        for key, value_count in result_value_counter.items(): 
            winner_value = None 
            winner_count = 0
            for value, count in value_count.items(): 
                if winner_value is None or (value != winner_value and count > winner_count): 
                    winner_value = value 
                    winner_count = count 
            
            final_result[key] = winner_value

        if len(final_result) == 0:
            return None 
        
        print("~=[FINAL RESULT:]=~ ")
        print(final_result)

        print("---------- Reconciling finished ---------")
        return final_result


