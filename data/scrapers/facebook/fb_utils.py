import re
import time
import random
import logging
from typing import Optional, List

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException,
    StaleElementReferenceException, ElementClickInterceptedException,
    ElementNotInteractableException,
)

logger = logging.getLogger("fb_scraper.utils")

MIN_TEXT_LENGTH = 3

RE_CYRILLIC = re.compile(r"[\u0400-\u04FF]{2,}")

RE_REACTION_LABEL = re.compile(
    r"([\d,\.]+)\s*[Kk]?\s*(?:reactions?|хариу үйлдэл|likes?)",
    re.IGNORECASE,
)
RE_K_SUFFIX = re.compile(r"^([\d\.]+)[Kk]$")

def short_sleep(base: float = 0.5) -> None:
    time.sleep(base * random.uniform(0.7, 1.3))

def medium_sleep(base: float = 2.0) -> None:
    time.sleep(base * random.uniform(0.8, 1.2))

def scroll_to_bottom(driver: webdriver.Chrome) -> None:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

def scroll_element_into_view(driver: webdriver.Chrome, element) -> None:
    try:
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", element)
    except Exception:
        pass

def safe_find(driver_or_el, by, selector: str):
    try:
        return driver_or_el.find_element(by, selector)
    except (NoSuchElementException, StaleElementReferenceException):
        return None

def safe_find_all(driver_or_el, by, selector: str) -> list:
    try:
        return driver_or_el.find_elements(by, selector)
    except (NoSuchElementException, StaleElementReferenceException):
        return []

def find_with_fallbacks(driver_or_el, selectors: List[str], by=By.CSS_SELECTOR):
    for sel in selectors:
        el = safe_find(driver_or_el, by, sel)
        if el:
            return el
    return None

def find_all_with_fallbacks(driver_or_el, selectors: List[str], by=By.CSS_SELECTOR) -> list:
    for sel in selectors:
        results = safe_find_all(driver_or_el, by, sel)
        if results:
            return results
    return []

def extract_comment_text(comment_el) -> str:
    try:
        text_divs = comment_el.find_elements(By.CSS_SELECTOR, "div[dir='auto']")
        for div in text_divs:
            t = div.get_attribute("innerText") or div.text or ""
            t = t.strip()
            if len(t) >= MIN_TEXT_LENGTH:
                return t
    except StaleElementReferenceException:
        pass

    try:
        full_text = comment_el.get_attribute("innerText") or ""
        lines = [ln.strip() for ln in full_text.split("\n") if ln.strip()]
        if lines:
            return max(lines, key=len)
    except StaleElementReferenceException:
        pass

    return ""

def clean_comment_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_valid_comment(text: str) -> bool:
    return bool(text and len(text.strip()) >= MIN_TEXT_LENGTH)

def parse_reaction_count(text: str) -> int:
    if not text:
        return 0
    text = text.strip()

    m = RE_REACTION_LABEL.search(text)
    if m:
        raw = m.group(1).replace(",", "")
        return _parse_k(raw)

    bare = re.sub(r"[^\d\.Kk]", "", text)
    if bare:
        return _parse_k(bare)

    return 0

def _parse_k(raw: str) -> int:
    m = RE_K_SUFFIX.match(raw)
    if m:
        return int(float(m.group(1)) * 1000)
    try:
        return int(float(raw))
    except ValueError:
        return 0

def get_reaction_count(driver_or_el, reaction_selectors: List[str]) -> int:
    for sel in reaction_selectors:
        els = safe_find_all(driver_or_el, By.CSS_SELECTOR, sel)
        for el in els:
            try:
                label = el.get_attribute("aria-label") or el.text or ""
                count = parse_reaction_count(label)
                if count > 0:
                    return count
            except StaleElementReferenceException:
                continue
    return 0

def safe_click(driver: webdriver.Chrome, element) -> bool:
    try:
        scroll_element_into_view(driver, element)
        element.click()
        return True
    except (ElementClickInterceptedException, ElementNotInteractableException):
        try:
            driver.execute_script("arguments[0].click();", element)
            return True
        except Exception:
            return False
    except StaleElementReferenceException:
        return False
    except Exception:
        return False

def click_button_by_text(
    driver: webdriver.Chrome,
    text_fragments: List[str],
    container=None,
    timeout: float = 4.0,
) -> bool:
    root = container if container is not None else driver
    try:
        fragments_xpath = " or ".join(
            f"contains(translate(normalize-space(.), "
            f"'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), "
            f"'{frag.lower()}')"
            for frag in text_fragments
        )
        xpath = (
            f".//*[@role='button' or @tabindex='0']"
            f"[{fragments_xpath}]"
        )
        elements = safe_find_all(root, By.XPATH, xpath)
        for el in elements:
            try:
                if safe_click(driver, el):
                    logger.debug("Clicked button with text matching: %s", text_fragments)
                    return True
            except Exception:
                continue
    except Exception as e:
        logger.debug("click_button_by_text error: %s", e)
    return False

def wait_for_any(driver: webdriver.Chrome, selectors: List[str], timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        for sel in selectors:
            el = safe_find(driver, By.CSS_SELECTOR, sel)
            if el:
                return el
        time.sleep(0.5)
    return None

def wait_for_comments(driver: webdriver.Chrome, timeout: float = 8.0) -> bool:
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[role='article']"))
        )
        return True
    except TimeoutException:
        return False

def normalize_fb_url(url: str) -> str:
    clean = url.split("?")[0].rstrip("/")
    return clean
