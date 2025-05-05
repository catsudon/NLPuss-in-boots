import time
import io
import re
import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import cv2
import numpy as np
import os

# === Setup ===
GEMINI_API_KEY = "AIzaSyC41esgj-53XXtVbfd709fjKeejf6vdOYA"
genai.configure(api_key=GEMINI_API_KEY)
ocr = PaddleOCR(lang='en', use_gpu=True)
MAX_TRIES = 10

# === Launch Chrome ===
options = Options()
options.add_argument("--start-maximized")
options.add_argument("--disable-infobars")
options.add_argument("--disable-extensions")
options.add_experimental_option("detach", True)
driver = webdriver.Chrome(options=options)

# === Screenshot function ===
def get_screenshot_from_driver(driver, save_path="screen.png"):
    screenshot = driver.get_screenshot_as_png()
    image = Image.open(io.BytesIO(screenshot)).convert("RGB")
    image.save(save_path)
    return save_path

# === OCR processing ===
def extract_screen_data(driver):
    path = get_screenshot_from_driver(driver)
    result = ocr.ocr(path, cls=False)
    elements = []
    
    boxes = []
    txts = []
    scores = []

    try:
        for line in result:
            for box, (text, confidence) in line:
                if confidence > 0.5:
                    x1 = int(min(p[0] for p in box))
                    y1 = int(min(p[1] for p in box))
                    x2 = int(max(p[0] for p in box))
                    y2 = int(max(p[1] for p in box))
                    elements.append({
                        "text": text,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence
                    })
                    boxes.append(box)
                    txts.append(text)
                    scores.append(confidence)
    except Exception as e:
        print(f"Error during OCR processing: {e}")

    # === Drawing the OCR result ===
    import os
    import numpy as np
    from paddleocr import draw_ocr
    from PIL import Image

    default_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    font_path = next((f for f in default_fonts if os.path.exists(f)), None)

    if font_path:
        image = Image.open(path).convert("RGB")
        im_show = draw_ocr(np.array(image), boxes, txts, scores, font_path=font_path)
        Image.fromarray(im_show).save("drawing.png")
    else:
        print("⚠️ No font found for drawing boxes. Skipped image annotation.")

    return elements


# === Ask Gemini what to do ===
def ask_llm_for_action(goal, elements):
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    prompt = f"""
You are controlling a browser screen using pixel coordinates and detected text boxes.

User's goal: {goal}

Here are the visible text elements:
""" + "\n".join(
        [f"{i}. '{el['text']}' at {el['bbox']}" for i, el in enumerate(elements)]
    ) + """

Decide which index (if any) should be clicked or typed into.

If typing is needed, also provide the text.

Respond in this exact format:
ACTION: click or type
INDEX: index of the element
TEXT: (only if typing, otherwise leave blank)

Examples:
ACTION: click
INDEX: 3
TEXT:

ACTION: type
INDEX: 2
TEXT: python tutorial
"""

    res = model.generate_content(prompt).text.strip()
    action = {"action": None, "index": -1, "text": ""}
    try:
        for line in res.splitlines():
            if line.startswith("ACTION:"):
                action["action"] = line.split(":", 1)[1].strip().lower()
            elif line.startswith("INDEX:"):
                action["index"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("TEXT:"):
                action["text"] = line.split(":", 1)[1].strip()
    except Exception as e:
        print("⚠️ Failed to parse LLM response:", res)
    return action

# === Click using JavaScript at pixel coordinates ===
def click_at_pixel(driver, x, y):
    script = f"""
    const el = document.elementFromPoint({x}, {y});
    if (el) {{
        // Visual marker
        const circle = document.createElement('div');
        circle.style.position = 'fixed';
        circle.style.left = '{x - 10}px';
        circle.style.top = '{y - 10}px';
        circle.style.width = '20px';
        circle.style.height = '20px';
        circle.style.borderRadius = '50%';
        circle.style.backgroundColor = 'rgba(0, 150, 255, 0.6)';
        circle.style.zIndex = 9999;
        circle.style.pointerEvents = 'none';
        circle.style.boxShadow = '0 0 10px rgba(0, 150, 255, 0.8)';
        circle.style.transition = 'transform 0.4s ease-out, opacity 0.4s ease-out';
        document.body.appendChild(circle);
        setTimeout(() => {{
            circle.style.transform = 'scale(3)';
            circle.style.opacity = '0';
        }}, 10);
        setTimeout(() => {{
            circle.remove();
        }}, 500);

        // Simulate real click
        const rect = el.getBoundingClientRect();
        const evt = new MouseEvent('click', {{
            view: window,
            bubbles: true,
            cancelable: true,
            clientX: {x},
            clientY: {y}
        }});
        el.dispatchEvent(evt);
    }}
    """
    driver.execute_script(script)



# === Type using JavaScript ===
from selenium.webdriver.common.keys import Keys

def type_into_active_element(driver, text, press_enter=True):
    try:
        el = driver.switch_to.active_element
        el.clear()
        el.send_keys(text)
        if press_enter:
            el.send_keys(Keys.ENTER)
    except Exception as e:
        print(f"❌ Typing failed: {e}")


# === Ask Gemini if task is done ===
def ask_if_task_is_done(goal, elements):
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    visible = "\n".join([el["text"] for el in elements])
    prompt = f"""
A user is trying to complete this task:

{goal}

The current visible screen contains:
{visible}

Has the task been completed? Reply only YES or NO. If NO, give advice.
"""
    return model.generate_content(prompt).text.strip().upper()

# === Main Loop ===
start_url = input("🌍 Enter the URL to visit: ").strip()
if start_url:
    if not start_url.startswith("http"):
        start_url = "https://" + start_url
    driver.get(start_url)
    time.sleep(2)

print("🌐 Chrome started. Ready to take tasks.")

while True:
    user_input = input("💬 Goal: ").strip()
    if not user_input:
        continue

    tries = 0
    task_completed = False

    while tries < MAX_TRIES:
        print(f"\n🔁 Attempt {tries + 1}:")
        elements = extract_screen_data(driver)
        if not elements:
            print("❌ No OCR elements found.")
            break

        action = ask_llm_for_action(user_input, elements)
        target_idx = action["index"]

        if target_idx == -1 or target_idx >= len(elements):
            print("🤷 LLM didn't find anything to act on.")
            break

        bbox = elements[target_idx]["bbox"]
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2

        if action["action"] == "click":
            print(f"🖱️ Clicking: {elements[target_idx]['text']} at ({center_x}, {center_y})")
            click_at_pixel(driver, center_x, center_y)

        elif action["action"] == "type":
            print(f"⌨️ Typing '{action['text']}' at ({center_x}, {center_y})")
            click_at_pixel(driver, center_x, center_y)
            time.sleep(0.5)
            type_into_active_element(driver, action["text"])

        time.sleep(1.5)
        updated_elements = extract_screen_data(driver)
        judge = ask_if_task_is_done(user_input, updated_elements)

        if "YES" in judge and "NO" not in judge:
            print("✅ Task completed.")
            task_completed = True
            break
        else:
            print("🕵️ Not done. Judge says:", judge)
            tries += 1

    if not task_completed:
        print("❌ Failed after max tries.")
