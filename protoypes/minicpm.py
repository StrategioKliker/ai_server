import base64
import requests
import json

MODEL = "minicpm-v"
OLLAMA_URL = "http://localhost:11434/api/generate"

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def ask_minicpm(prompt, images):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "images": images
    }
    response = requests.post(OLLAMA_URL, json=payload)
    # print("Res", response.text)
    data = json.loads(response.text)
    # print("DATA: ", data)

    # The generate endpoint returns text under "response"
    return data.get("response", "").strip()

if __name__ == "__main__":
    # prompt = """
    #     You are a visual analysis assistant. Analyze the provided image carefully and extract the following information in strict JSON format. Do not include any explanation or extra text. Only return a well-formatted JSON object.

    #     TASKS:
    #     1. Determine if there is a bottle of ink in the image.
    #     2. Count how many bottles are present.
    #     3. Identify the color(s) of the bottles.
    #     4. Detect the printer brand, if any printer is visible.
    #     5. Extract any text that is visible in the image.

    #     JSON OUTPUT FORMAT:
    #     {
    #         "ink_bottle_present": true,
    #         "bottle_count": 3,
    #         "bottle_colors": ["cyan", "magenta", "yellow"],
    #         "printer_brand": "Canon",
    #         "extracted_text": "PIXMA G2020"
    #     }

    #     If any field is not applicable or not visible in the image, use null or an empty array as appropriate.
    #     """

    # prompt = """
    #     You are a vision model comparing two images (image 1 and image 2). Your task is to determine if both images show the **same product**.

    #     Follow these steps:
    #     1. Analyze both images and compare their visual features, labels, shapes, colors, and branding.
    #     2. Decide if they represent the **same product** (e.g., same brand, model, or item).
    #     3. If yes, try to **identify the product** by name, brand, or category if possible.
    #     4. Respond in strict JSON format with your findings.

    #     JSON OUTPUT FORMAT:
    #     {
    #     "same_product": true,
    #     "product_name": "Canon PIXMA G2020"
    #     }

    #     If they are not the same product, use:
    #     {
    #     "same_product": false,
    #     "product_name": null
    #     }
    #     """

    prompt = "Extract all the product text from the image and return it as json"

    images = ["screenshot.png"]
    encode_images = []
    for image in images:
        encoded_img = encode_image(image)
        encode_images.append(encoded_img)

    answer = ask_minicpm(prompt, encode_images)
    print("Anwser:", answer)
