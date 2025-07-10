import io
import requests
from PIL import Image 

def image_to_png(image_url, save_path="test_output.png"):
    res = requests.get(image_url, timeout=10)
    if res.status_code != 200:
        raise Exception(f"Failed to fetch image: {res.status_code}")

    img = Image.open(io.BytesIO(res.content)).convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    # Save to disk to visually verify
    with open(save_path, "wb") as f:
        f.write(buf.getvalue())

    print(f"Saved image to {save_path}")

    return buf.getvalue()


image_to_png('https://www.tehnomedia.rs/image/67035.jpg?tip=thumb&tip_slike=0')
image_to_png('https://www.bigbang.si/media/images/no-image-285.webp', 'test_output2.png')
