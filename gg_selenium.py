import io
import re
import time
import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from htmlrag import clean_html
from PIL import Image

GEMINI_API_KEY = "AIzaSyC41esgj-53XXtVbfd709fjKeejf6vdOYA"
genai.configure(api_key=GEMINI_API_KEY)

options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


# 0=INFO, 1=WARNING, 2=LOG_ERROR, 3=LOG_FATAL
options.add_argument("--log-level=3")
options.add_argument("--disable-logging")

driver = webdriver.Chrome(options=options)
generation_config = genai.GenerationConfig(
    temperature=0,
)

model = genai.GenerativeModel(
    "models/gemini-2.0-flash", generation_config=generation_config)

MAX_TRIES = 10


def prune_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "meta", "link", "noscript", "iframe", "head"]):
        tag.decompose()
    for tag in soup.find_all(lambda t: t.has_attr("style") and (
            "display:none" in t["style"] or "opacity:0" in t["style"])
            or t.has_attr("hidden")
            or (t.has_attr("aria-hidden") and t["aria-hidden"] == "true")):
        tag.unwrap()
    relevant_tags = ["a", "button", "input", "textarea", "select", "option", "label", "form",
                     "table", "tr", "td", "th", "h1", "h2", "h3", "p", "li", "ul", "ol"]
    for tag in soup.find_all(True):
        if tag.name not in relevant_tags:
            tag.unwrap()
    for tag in soup.find_all(relevant_tags):
        for attr in ["class", "id", "style", "onclick", "onmouseover", "data-*", "aria-*"]:
            tag.attrs.pop(attr, None)
    simplified_html = str(soup)

    simplified_html = clean_html(simplified_html)
    simplified_html = BeautifulSoup(simplified_html, "html.parser")
    simplified_html = simplified_html.prettify()
    return simplified_html[:100000]

def get_screenshot_from_driver(driver, save_path="screen.png"):
    screenshot = driver.get_screenshot_as_png()
    image = Image.open(io.BytesIO(screenshot)).convert("RGB")
    image.save(save_path)
    return save_path


# def prune_html(html):
#     soup = BeautifulSoup(html, "html.parser")
#     for tag in soup(["script", "style", "meta", "link", "noscript", "iframe", "head"]):
#         tag.decompose()

#     visible_tags = soup.select(
#         "a, button, input, textarea, select, option, label, form, table, tr, td, th, h1, h2, h3, p, li, ul, ol")
#     new_soup = BeautifulSoup("", "html.parser")
#     for tag in visible_tags:
#         new_soup.append(tag)

#     for tag in new_soup.find_all(True):
#         for attr in list(tag.attrs.keys()):
#             if attr in ["class", "id", "style", "onclick", "onmouseover"] or attr.startswith("data-") or attr.startswith("aria-"):
#                 del tag.attrs[attr]
#     simplified_html = str(new_soup)[:100000]

#     # simplified_html = clean_html(simplified_html)
#     # simplified_html = BeautifulSoup(simplified_html, "html.parser")
#     # simplified_html = simplified_html.prettify()
#     return str(simplified_html)


def get_visible_texts():
    elements = driver.find_elements(
        By.XPATH, "//*[normalize-space(text()) != '']")
    visible_texts = [elem.text.strip()
                     for elem in elements if elem.is_displayed()]
    return "\n".join(visible_texts)


def get_current_html():
    raw_html = driver.page_source
    with open("raw_html.html", "w", encoding="utf-8") as f:
        f.write(raw_html)
    pruned_html = prune_html(raw_html)
    with open("pruned_html.html", "w", encoding="utf-8") as f:
        f.write(pruned_html)
    return pruned_html


def extract_python_code(text):
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def send_to_gemini(prompt, html, history, advice=None):
    history_text = "\n".join(
        [f"- Step {i + 1}: Command: {h['command']}, Code: (truncated)\n{h['code'][:200]}..."
         for i, h in enumerate(history)]) if history else "None yet"

    advice_text = f"\nJudge's latest advice: {advice}" if advice else ""

    context = f"""
        You are an AI assistant controlling a Selenium browser.
        DO NOT create a new 'webdriver.Chrome()'. Use existing 'driver'.
        This is the current cleaned HTML:
        {html}

        User's goal: {prompt}

        Past steps:
        {history_text}

        {advice_text}

        Generate VALID Selenium Python code to perform the next action.
        Only return a single Python code block, nothing else.
        """
    response = model.generate_content(context)
    # print(context)
    return response.text


def ask_if_task_is_done(goal, html, last_action, history):

    # model = genai.GenerativeModel("models/gemini-2.0-flash")
    history_text = (
        "\n".join(
            [
                f"- Step {i + 1}: Command: {h['command']}, Code: (truncated)\n{h['code'][:200]}..."
                for i, h in enumerate(history)
            ]
        )
        if history
        else "None yet"
    )
    visible_text = get_visible_texts()
    action_summary = f"""
Last action:
Command: {last_action['command']}
Generated code: (truncated)\n{last_action['code'][:200]}...
"""
    prompt = f"""
A user wants to complete this goal: {goal}

System limitations:
- Only visible HTML is available.
- User's last action is shown.
- Past steps are listed.
- Current visible HTML:

Past steps:
{history_text}

Current visble text shown in HTML: {visible_text}

{action_summary}



Has the goal been completed?

criteria for completion (one of these must be true):
- The html contains the expected elements or changes.
- The user can see the expected elements or changes.
- The user can interpret the changes made by the last action that resulted in a visible change to the goal.

Respond in this format ONLY:
DONE: YES or NO
ADVICE: (If NO, explain next possible step or issue)
"""
    result = model.generate_content(prompt)
    # print("prompt:", prompt)
    return result.text.strip()


def execute_code(code):
    try:
        exec(code, {"driver": driver, "By": By, "Keys": Keys})
    except Exception as e:
        print("❌ Execution error:", e)


# === Main loop ===
print("✅ Selenium is running!")


while True:
    user_goal = input("💬 What is the goal? (type 'exit' to quit): ").strip()
    if user_goal.lower() == "exit":
        break

    history = []
    judge_response = ""
    tries = 0
    task_completed = False
    driver.get('data:,')

    while tries < MAX_TRIES:
        print(f"\n🔁 Attempt {tries + 1}")

        html = get_current_html()
        gemini_response = send_to_gemini(
            user_goal, html, history, advice=judge_response)

        generated_code = extract_python_code(gemini_response)

        # print(f"\n🤖 Gemini generated code:\n{generated_code}\n")
        execute_code(generated_code)

        last_action = {"command": user_goal, "code": generated_code}
        history.append(last_action)

        time.sleep(2)  # Let browser update

        html_after = get_current_html()
        judge_response = ask_if_task_is_done(
            user_goal, html_after, last_action, history)

        print("\n📝 Judge says:\n", judge_response)

        if "DONE: YES" in judge_response:
            print("✅ Task completed!")
            task_completed = True
            break
        else:
            print("🕵️ Not done yet. Continuing...")
            tries += 1

    if not task_completed:
        print("❌ Max tries reached. Task not completed.")

driver.quit()
