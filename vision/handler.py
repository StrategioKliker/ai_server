from llama_cpp.llama_chat_format import register_chat_format, ChatFormatterResponseItem
from typing import List, Dict
import os

@register_chat_format("minicpm-o-2_6")
class MiniCPMo26ChatHandler:
    def __init__(self, clip_model_path: str = None):
        if clip_model_path is None or not os.path.isfile(clip_model_path):
            raise ValueError("A valid clip_model_path is required for MiniCPM-o-2_6.")
        self.clip_model_path = clip_model_path

    def __call__(self, messages: List[Dict]) -> List[ChatFormatterResponseItem]:
        prompt = ""
        images = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                continue  # system prompts not used

            if role == "user":
                prompt += "<|user|>\n"
                for part in content:
                    if part["type"] == "text":
                        prompt += part["text"] + "\n"
                    elif part["type"] == "image_url":
                        img_url = part["image_url"]["url"]
                        if not img_url.startswith("file://"):
                            raise ValueError("Only local image paths (file://) are supported.")
                        image_path = img_url.replace("file://", "")
                        images.append(image_path)
                        prompt += "<image>\n"

            elif role == "assistant":
                prompt += "<|assistant|>\n" + content + "\n"

        prompt += "<|assistant|>\n"  # signal model to respond

        return [ChatFormatterResponseItem(
            prompt=prompt,
            images=images
        )]
