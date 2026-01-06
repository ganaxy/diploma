import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from utils import make_row

SOURCE_NAME = "example-dynamic-site.mn"

PAGE_LOAD_WAIT = 6

REQUEST_DELAY_SECONDS = 3

SELECTORS = {
    "comment_block": "div.comment-item",

    "comment_text": "span.comment-text",

    "likes": "button.like-btn span",

    "dislikes": "button.dislike-btn span",

    "reply_block": "div.reply-item",

    "reply_text": "span.reply-text",

    "reply_likes": "button.like-btn span",
    "reply_dislikes": "button.dislike-btn span",
}

def build_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    options.add_argument("--lang=mn")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def safe_text(element, selector: str) -> str:
    try:
        return element.find_element(By.CSS_SELECTOR, selector).text.strip()
    except Exception:
        return ""

def safe_int(element, selector: str) -> int:
    text = safe_text(element, selector)
    digits = "".join(filter(str.isdigit, text))
    return int(digits) if digits else 0

def click_load_more(driver, button_selector: str, max_clicks: int = 5) -> None:
    for i in range(max_clicks):
        try:
            btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, button_selector))
            )
            btn.click()
            print(f"  Clicked 'Load more' ({i+1}/{max_clicks})")
        except Exception:
            print(f"  No more 'Load more' button after {i} clicks")
            break

def scrape_selenium_site(article_url: str) -> list:
    rows = []
    row_id = 1
    driver = None

    try:
        driver = build_driver()
        print(f"  Opening: {article_url}")
        driver.get(article_url)

        try:
            WebDriverWait(driver, PAGE_LOAD_WAIT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, SELECTORS["comment_block"]))
            )
        except Exception:
            print(f"  [WARNING] Comments did not load within {PAGE_LOAD_WAIT}s at {article_url}")

        comment_elements = driver.find_elements(By.CSS_SELECTOR, SELECTORS["comment_block"])
        print(f"  Found {len(comment_elements)} top-level comment elements")

        for comment_el in comment_elements:
            text = safe_text(comment_el, SELECTORS["comment_text"])
            likes = safe_int(comment_el, SELECTORS["likes"])
            dislikes = safe_int(comment_el, SELECTORS["dislikes"])

            rows.append(make_row(
                row_id=row_id,
                text=text,
                likes=likes,
                dislikes=dislikes,
                source=SOURCE_NAME,
                relation=0,
            ))
            row_id += 1

            reply_elements = comment_el.find_elements(By.CSS_SELECTOR, SELECTORS["reply_block"])
            for reply_el in reply_elements:
                reply_text = safe_text(reply_el, SELECTORS["reply_text"])
                reply_likes = safe_int(reply_el, SELECTORS["reply_likes"])
                reply_dislikes = safe_int(reply_el, SELECTORS["reply_dislikes"])

                rows.append(make_row(
                    row_id=row_id,
                    text=reply_text,
                    likes=reply_likes,
                    dislikes=reply_dislikes,
                    source=SOURCE_NAME,
                    relation=1,
                ))
                row_id += 1

    except Exception as e:
        print(f"  [ERROR] Unexpected error: {e}")

    finally:
        if driver:

    time.sleep(REQUEST_DELAY_SECONDS)
    return rows
