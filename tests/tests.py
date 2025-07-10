from vision.cpp import ImageInference

def run_vqa_test(prompt, images):
    vision = ImageInference()
    anwser = vision.prompt(prompt=prompt, images=images)
    print("Anwsered: ", anwser)
    print(f"Time taken: {vision.elapsed_minutes} mins", "\n")

t1_prompt = """
        You are a visual analysis assistant. Analyze the provided image carefully and extract the following information in strict JSON format. Do not include any explanation or extra text. Only return a well-formatted JSON object.

        TASKS:
        1. Determine if there is a bottle of ink in the image.
        2. Count how many bottles are present. Bottle does not have to be fully visible.
        3. Identify the color(s) of the bottles.
        4. Detect the printer brand, if any printer is visible.
        5. Extract any text that is visible in the image.

        JSON OUTPUT FORMAT:
        {
            "ink_bottle_present": true,
            "bottle_count": 2,
            "bottle_colors": ["color1", "color1"],
            "printer_brand": "Brand",
            "extracted_text": "Some text"
        }

        If any field is not applicable or not visible in the image, use null or an empty array as appropriate.
        """

print("\n------T1 test running:------")
print("Extracting data from provided image")
run_vqa_test(t1_prompt, ["images/printer.jpg"])

t2_prompt = """
    You are a vision model comparing two images (image 1 and image 2). Your task is to determine if both images show the **same product**.

    Follow these steps:
    1. Analyze both images and compare their visual features, labels, shapes, colors, and branding.
    2. Decide if they represent the **same product** (e.g., same brand, model, or item).
    3. If yes, try to **identify the product** by name, brand, or category if possible.
    4. Respond in strict JSON format with your findings.

    JSON OUTPUT FORMAT:
    {
    "same_product": true,
    "product_name": "Canon PIXMA G2020"
    }

    If they are not the same product, use:
    {
    "same_product": false,
    "product_name": null
    }
    """

print("\n------T2 test 1 running:------")
print("Testing image matching results, comparing two images")
run_vqa_test(t2_prompt, ["images/product1.jpg", "images/product2.jpg"])

print("\n------T2 test 2 running:------")
print("Testing image matching results, comparing two images")
run_vqa_test(t2_prompt, ["images/product1.jpg", "images/product4.jpg"])


t3_prompt = "Extract all the product text from the image and return it as json"

# print("\n------T3 test  running:------")
# print("Trying to extract data from screenshot")
# run_vqa_test(t3_prompt, ["images/screenshot.png"])