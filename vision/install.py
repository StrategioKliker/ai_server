import os
import requests


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


if __name__ == '__main__':
    init_projector_if_not_exists()