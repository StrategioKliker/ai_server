import base64
import requests
import json

# ─────── ADJUSTED MODEL NAME HERE ───────
MODEL = "qwen2.5vl"  # (was "minicpm-v")

# This assumes Ollama is serving HTTP on localhost:11434
OLLAMA_URL = "http://localhost:11434/api/generate"

def encode_image(path):
    """
    Reads an image file from disk, encodes it to Base64, and returns
    the resulting string. Yes, you need this if you want to send images.
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def ask_qwen(prompt, images):
    """
    Sends a JSON payload to Ollama’s /api/generate endpoint.
    - 'model' must match the Ollama model name exactly.
    - 'prompt' is whatever instruction you want Qwen to follow.
    - 'images' is a list of Base64-encoded images, or [] if no images.
    Returns the raw text response from Qwen.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "images": images
    }
    # If this errors out, check that Ollama is actually running on port 11434.
    response = requests.post(OLLAMA_URL, json=payload)
    # print(response.text)


    # Don’t blindly trust the API—inspect the response.
    if response.status_code != 200:
        raise Exception(f"Ollama API returned {response.status_code}: {response.text}")
    
    data = json.loads(response.text)
    # Ollama’s generate endpoint always returns something under "response"
    return data.get("response", "").strip()

if __name__ == "__main__":
    # ─────── Customize Your Prompt Here ───────
    prompt = "Explain the image in detail"

    # ─────── List your local image filenames here ───────
    images = ["images/doggo.png"]  # Change this to your actual filenames
    encoded_images = []
    for img_path in images:
        try:
            encoded_images.append(encode_image(img_path))
        except FileNotFoundError:
            print(f"Error: Couldn’t find image at '{img_path}'. Double-check the path.")
            exit(1)

    # Actually call the function and print the answer
    try:
        answer = ask_qwen(prompt, encoded_images)
        print("Answer:", answer)
    except Exception as e:
        print("🚨 Failed to get a response:", e)
