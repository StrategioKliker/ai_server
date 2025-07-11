import io
import os
import sys
import base64
import hashlib
import requests

from PIL import Image
from time import sleep
from llama_cpp import Llama
from datetime import datetime
from typing import List, Union, Dict
from jsonz.validator import is_valid_json
from jsonz.extractor import extract_json_from_str
from llama_cpp.llama_chat_format import register_chat_format, Llava15ChatHandler

@register_chat_format("minicpm-o-2_6")
class MiniCPMo26ChatHandler(Llava15ChatHandler):
    DEFAULT_SYSTEM_MESSAGE = None

    CHAT_FORMAT = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "<|user|>\n"
        "{% if message['content'] is iterable %}"
        "{% for part in message['content'] %}"
            "{% if part['type'] == 'text' %}"
                "{{ part['text'] }}\n"
            "{% elif part['type'] == 'image_url' %}"
                "{{ part['image_url']['url'] }}\n"
            "{% endif %}"
        "{% endfor %}"
        "{% else %}"
        "{{ message['content'] }}\n"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "<|assistant|>\n{{ message['content'] }}\n"
        "{% endif %}"
        "{% endfor %}"
        "<|assistant|>\n"
    )

class suppress_stdout(object):
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


class ImageInference:
    def __init__(self):
        self.elapsed_minutes = 0

        self.image_dir = 'images'
        os.makedirs(self.image_dir, exist_ok=True)

        print("Loading MiniCPM-o-2_6 model with multimodal support…", flush=True)

        # Paths
        model_path = os.path.join("minicpm-o", "Model-7.6B-Q6_K.gguf")
        projector_path = os.path.join("minicpm-o", "mmproj-model-f16.gguf")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Missing model at {model_path}")
        if not os.path.isfile(projector_path):
            raise FileNotFoundError(f"Missing projector at {projector_path}")

        with suppress_stdout():
            chat_handler = MiniCPMo26ChatHandler(clip_model_path=projector_path)

            self.llm = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                chat_format="minicpm-o-2_6",
                n_threads=os.cpu_count(),
                n_ctx=4068,
                n_gpu_layers=8,
                main_gpu=0,
                gpu_mlock=True,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                repeat_penalty=1.2,
                typical_p=1.0,
                mirostat_mode=0,
            )

        print("Model loaded ✔️", flush=True)
        print("GPU used:", self.llm.model_params.n_gpu_layers > 0, flush=True)

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

            content.append({
                "type": "image_url",
                "image_url": {"url": img_data}
            })
            image_found = True

        return content if image_found else None

    def prompt(self, prompt: str, system_prompt: Union[str, None], images: List[str], expected_json_schema: dict) -> Union[str, None]:
        print("Running prompt", flush=True)
        if not prompt or not isinstance(prompt, str) or not images:
            print("Invalid prompt or images", flush=True)
            return None

        content = [{"type": "text", "text": prompt.strip()}]
        content = self.__process_image_content(images=images, content=content)
        if not content:
            print("No valid images found")
            return None

        messages = [{"role": "user", "content": content}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        return self.__get_inference_result(messages, expected_json_schema)

    def __get_inference_result(self, messages, expected_json_schema, repeat_target=5):
        repeat_count = 0
        repeated_results = []
        repeat_count_target = repeat_target * 3

        while repeat_count < repeat_count_target:
            try:
                start = datetime.now()
                res = self.llm.create_chat_completion(
                    messages=messages,
                    # response_format={"type": "json_object"},
                )
                self.elapsed_minutes = (datetime.now() - start).total_seconds() / 60
                result = res["choices"][0]["message"]["content"].strip()
                print("Result:", result)

                extracted_json = extract_json_from_str(result)
                print("Extracted: ", extracted_json)
                print("Type: ", type(extracted_json))
                if not isinstance(extracted_json, dict):
                    repeat_count += 1
                    print(f"⚠️ Failed to extract JSON — retrying ({repeat_count})")
                    sleep(2)
                    continue

                if len(extracted_json) > 0 and expected_json_schema and not is_valid_json(expected_json_schema, extracted_json):
                    repeat_count += 1
                    print(f"⚠️ JSON validation failed — retrying ({repeat_count})")
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

        if not repeated_results:
            return None

        return self.__reconcile_result(repeated_results)

    def __reconcile_result(self, repeated_results: List[dict]):
        result_value_counter = {}

        print("---------- Reconciling results ----------")
        for result in repeated_results:
            print("Result:", result)
            for key, value in result.items():
                result_value_counter.setdefault(key, {})
                result_value_counter[key][value] = result_value_counter[key].get(value, 0) + 1

        final_result = {
            key: max(value_count.items(), key=lambda x: x[1])[0]
            for key, value_count in result_value_counter.items()
        }

        print("✅ Final result:")
        print(final_result)
        print("----------------------------------------")
        return final_result
