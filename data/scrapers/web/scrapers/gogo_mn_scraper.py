import re
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from driver_setup import build_driver
from utils import make_row

RE_CYRILLIC = re.compile(r'[\u0400-\u04FF]{3,}')

SOURCE_NAME = "gogo.mn"
COMMENT_LOAD_WAIT = 10
REQUEST_DELAY_SECONDS = 1.5

SELECTORS = {
    "comments_section":  "div.news-detail-comment-container",
    "comment_block":     "div.comment-item.comment-level-0",
    "comment_text":      "div.comment-body p",
    "likes":             "a.comment-like",
    "dislikes":          "a.comment-dislike",
    "reply_block":       "div.comment-item.comment-level-1",
    "reply_text":        "div.comment-body p",
    "reply_likes":       "a.comment-like",
    "reply_dislikes":    "a.comment-dislike",
    "load_more":         "a.uk-display-block.uk-text-bold.text-13.text-blue",
}

def parse_int_from_element(el) -> int:
    try:
        raw = el.text.strip()
        digits = re.sub(r"\D", "", raw)
        return int(digits) if digits else 0
    except Exception:
        return 0

def click_load_more(driver, max_clicks: int = 10) -> None:
    for i in range(max_clicks):
        try:
            btn = WebDriverWait(driver, 4).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, SELECTORS["load_more"]))
            )
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(1)
            print(f"    Clicked 'Бусад сэтгэгдэл' ({i+1})")
        except TimeoutException:
            print(f"    No more 'Бусад сэтгэгдэл' button after {i} click(s)")
            break

def scrape_gogo_mn(article_url: str, category: str = "", driver=None) -> list:
    rows = []
    row_id = 1
    owns_driver = driver is None

    try:
        if owns_driver:
            driver = build_driver(headless=True)
        print(f"  Opening: {article_url}")
        driver.get(article_url)

        try:
            WebDriverWait(driver, COMMENT_LOAD_WAIT).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, SELECTORS["comments_section"])
                )
            )
            print("  ✓ Comment section loaded")
        except TimeoutException:
            print(f"  ⚠ Comment section did not appear within {COMMENT_LOAD_WAIT}s")

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        click_load_more(driver, max_clicks=10)

        comment_els = driver.find_elements(By.CSS_SELECTOR, SELECTORS["comment_block"])
        print(f"  Found {len(comment_els)} top-level comment(s)")

        for comment_el in comment_els:
            try:
                text_el = comment_el.find_element(By.CSS_SELECTOR, SELECTORS["comment_text"])
                text = text_el.text.strip()
            except NoSuchElementException:
                text = ""

            try:
                likes = parse_int_from_element(
                    comment_el.find_element(By.CSS_SELECTOR, SELECTORS["likes"])
                )
            except NoSuchElementException:
                likes = 0
            try:
                dislikes = parse_int_from_element(
                    comment_el.find_element(By.CSS_SELECTOR, SELECTORS["dislikes"])
                )
            except NoSuchElementException:
                dislikes = 0

            if not text or not RE_CYRILLIC.search(text):
                continue

            rows.append(make_row(row_id, text, likes, dislikes, SOURCE_NAME, relation=0, category=category))
            row_id += 1

            reply_els = comment_el.find_elements(By.CSS_SELECTOR, SELECTORS["reply_block"])
            for reply_el in reply_els:
                try:
                    r_text = reply_el.find_element(
                        By.CSS_SELECTOR, SELECTORS["reply_text"]
                    ).text.strip()
                except NoSuchElementException:
                    r_text = ""
                try:
                    r_likes = parse_int_from_element(
                        reply_el.find_element(By.CSS_SELECTOR, SELECTORS["reply_likes"])
                    )
                except NoSuchElementException:
                    r_likes = 0
                try:
                    r_dislikes = parse_int_from_element(
                        reply_el.find_element(By.CSS_SELECTOR, SELECTORS["reply_dislikes"])
                    )
                except NoSuchElementException:
                    r_dislikes = 0

                if not r_text or not RE_CYRILLIC.search(r_text):
                    continue

                rows.append(make_row(row_id, r_text, r_likes, r_dislikes, SOURCE_NAME, relation=1, category=category))
                row_id += 1

    except Exception as e:
        print(f"  [ERROR] {e}")
    finally:
        if owns_driver and driver:
            driver.quit()

    time.sleep(REQUEST_DELAY_SECONDS)
    return rows
