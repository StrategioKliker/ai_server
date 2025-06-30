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


    def init_projector_if_not_exists(self):
        if not os.path.isfile(self._projector_path):
            print("Downloading correct projector model...", flush=True)
            download_url = (
                "https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/"
                "main/mmproj-model-f16.gguf"
            )
            resp = requests.get(download_url, stream=True)
            if resp.status_code != 200:
                raise RuntimeError("Failed to fetch projector GGUF file.")

            with open(self._projector_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Projector downloaded:", self._projector_path, flush=True)


    def prompt(self, prompt: str, images: List[str]) -> Union[str, None]:
        print("Running prompt", flush=True)
        if not prompt or not isinstance(prompt, str) or not images:
            print("Invalid prompt or images", flush=True)
            return None

        # Build structured multimodal content
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
                print("Skipping unsupported image:", img, flush=True)

        # Run the model
        start = datetime.now()
        try:
            res = self.llm.create_chat_completion(
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
            print("LLM inference error:", e, flush=True)
            return None


if __name__ == "__main__":
    ImageInference().init_projector_if_not_exists()