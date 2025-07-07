import os
import sys
import requests
from datetime import datetime
from typing import List, Union
from llama_cpp import Llama
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

    def prompt(self, prompt: str, system_prompt: Union[str, None], images: List[str]) -> Union[str, None]:
        print("Running prompt", flush=True)
        if not prompt or not isinstance(prompt, str) or not images:
            print("Invalid prompt or images", flush=True)
            return None

        # Build structured multimodal content
        content = [{"type": "text", "text": "Respond in English. " + prompt}]

        image_found = False 
        for img in images:
            if os.path.isfile(img):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"file://{os.path.abspath(img)}"}
                })
                image_found = True 
            elif img.startswith(("http://", "https://")):
                res = requests.get(img)
                if res is None or res.status_code != 200:
                    print("Failed to get image from url: ", img, " with status code: ", res.status_code)
                    continue
                
                content.append({
                    "type": "image_url",
                    "image_url": {"bytes": res.content}
                })
                image_found = True 
            else:
                print("Skipping unsupported image:", img, flush=True)

        if not image_found: 
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
                messages,
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
