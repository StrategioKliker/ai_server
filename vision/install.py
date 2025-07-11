import os
import sys
import requests


def init_projector_if_not_exists():
    os.makedirs('minicpm', exist_ok=True)
    proj_path = os.path.join('minicpm', "mmproj-model-f16.gguf")
    if os.path.isfile(proj_path):
        print(f"Projector model already at {proj_path}")
        return proj_path

    print("⬇️  Downloading projector model…")
    url = (
        "https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/"
        "main/mmproj-model-f16.gguf"
    )
    download_stream(url, proj_path)
    return proj_path

def init_projector_o_if_not_exists():
    os.makedirs('minicpm-o', exist_ok=True)
    proj_path = os.path.join('minicpm-o', "mmproj-model-f16.gguf")
    if os.path.isfile(proj_path):
        print(f"Projector model already at {proj_path}")
        return proj_path

    print("⬇️  Downloading projector model…")
    url = (
        "https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf/resolve/"
        "main/mmproj-model-f16.gguf"
    )
    download_stream(url, proj_path)
    return proj_path

def init_main_model_if_not_exists():
    os.makedirs('minicpm', exist_ok=True)
    model_path = os.path.join('minicpm', "ggml-model-f16.gguf")
    if os.path.isfile(model_path):
        print(f"Main model already at {model_path}")
        return model_path

    print("⬇️  Downloading main MiniCPM model…")
    url = (
        "https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/"
        "main/ggml-model-f16.gguf"
    )
    download_stream(url, model_path)
    return model_path


# https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf?library=llama-cpp-python
def init_main_model_o_if_not_exists():
    os.makedirs('minicpm-o', exist_ok=True)
    model_path = os.path.join('minicpm-o', "Model-7.6B-Q6_K.gguf")
    if os.path.isfile(model_path):
        print(f"Main model already at {model_path}")
        return model_path

    print("⬇️  Downloading main MiniCPM model…")
    url = (
        "https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf/resolve/"
        "main/Model-7.6B-Q6_K.gguf"
    )
    download_stream(url, model_path)
    return model_path

def download_stream(url: str, dest: str):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", 0))
    downloaded = 0
    chunk_size = 1 << 20  # 1 MiB

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                sys.stdout.write(
                    f"\r  {downloaded/(1<<20):.1f}/{total/(1<<20):.1f} MiB ({pct}%)"
                )
                sys.stdout.flush()
    
    print(f"Downloaded to {dest}")

if __name__ == "__main__":
    init_projector_if_not_exists()
    init_main_model_if_not_exists()

    init_projector_o_if_not_exists()
    init_main_model_o_if_not_exists()