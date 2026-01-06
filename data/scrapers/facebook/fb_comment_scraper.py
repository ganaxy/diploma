import time
import logging
from typing import List, Set

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException

from fb_utils import (
    extract_comment_text, clean_comment_text, is_valid_comment,
    get_reaction_count, safe_click, safe_find_all,
    click_button_by_text, short_sleep,
    scroll_to_bottom, scroll_element_into_view, wait_for_comments,
    medium_sleep,
)
from fb_selectors import (
    REACTION_COUNT, VIEW_MORE_COMMENTS_TEXT,
    VIEW_REPLIES_TEXT, HIDE_REPLIES_TEXT,
)
from fb_compat import make_row

logger = logging.getLogger("fb_scraper.comment_scraper")

_SORT_BUTTON_TEXT = [
    "most relevant", "top comments", "all comments",
    "хамгийн хамааралтай", "шилдэг сэтгэгдэл", "бүх сэтгэгдэл",
    "эрэмбэлэх", "эрэмбэ",
]
_ALL_COMMENTS_OPTION_TEXT = [
    "all comments", "бүх сэтгэгдэл",
    "newest first", "хамгийн шинэ",
]

def scrape_post_comments(
    driver: webdriver.Chrome,
    post_url: str,
    source_name: str,
    source_type: str,
    max_comments: int = 200,
    max_replies_per_comment: int = 10,
) -> List[dict]:
    rows: List[dict] = []

    logger.info("Scraping post: %s", post_url)
    try:
        driver.get(post_url)
        if not wait_for_comments(driver, timeout=8.0):
            logger.warning("Comments did not appear within 8s — %s", post_url)
            medium_sleep(1.5)
    except Exception as e:
        logger.error("Failed to navigate to %s: %s", post_url, e)
        return rows

    _try_switch_to_all_comments(driver)
    expand_all_comments(driver, max_clicks=15)

    comment_elements = _get_comment_elements(driver)
    logger.info("Found %d comment elements", len(comment_elements))

    seen_texts: Set[str] = set()
    row_id = 1

    for comment_el in comment_elements[:max_comments]:
        try:
            raw = extract_comment_text(comment_el)
            text = clean_comment_text(raw)

            if not is_valid_comment(text):
                continue
            if text in seen_texts:
                continue
            seen_texts.add(text)

            likes = get_reaction_count(comment_el, REACTION_COUNT)

            rows.append(make_row(
                row_id=row_id,
                text=text,
                likes=likes,
                dislikes=0,
                source=source_name,
                relation=0,
                category=source_type,
            ))
            row_id += 1

        except StaleElementReferenceException:
            continue
        except Exception as e:
            logger.debug("Comment parse error: %s", e)
            continue

    top_level_count = len(rows)
    logger.info("Top-level comments collected: %d", top_level_count)

    reply_count = _collect_replies(
        driver, rows, seen_texts, row_id,
        source_name, source_type,
        max_replies_per_comment,
    )

    logger.info("Post done → %d comments + %d replies = %d total",
                top_level_count, reply_count, len(rows))
    return rows

def _collect_replies(
    driver: webdriver.Chrome,
    rows: List[dict],
    seen_texts: Set[str],
    next_row_id: int,
    source_name: str,
    source_type: str,
    max_replies_per_comment: int,
) -> int:
    articles_before = set()
    for el in safe_find_all(driver, By.CSS_SELECTOR, "[role='article']"):
        try:
            articles_before.add(id(el))
        except Exception:
            pass

    expanded = _expand_reply_buttons(driver, max_rounds=3)
    if expanded == 0:
        return 0

    short_sleep(0.8)

    all_articles = safe_find_all(driver, By.CSS_SELECTOR, "[role='article']")
    reply_elements = []
    for el in all_articles:
        if id(el) not in articles_before:
            reply_elements.append(el)

    if not reply_elements:
        try:
            texts_batch = driver.execute_script(
                "return arguments[0].map(function(el){ return (el.innerText||'').trim(); });",
                all_articles,
            )
        except Exception:
            texts_batch = [""] * len(all_articles)

        for el, text in zip(all_articles, texts_batch):
            length = len(text or "")
            if 3 <= length <= 8000:
                cleaned = clean_comment_text(text)
                if cleaned and cleaned not in seen_texts:
                    reply_elements.append(el)

    logger.info("Found %d potential reply elements", len(reply_elements))

    reply_count = 0
    row_id = next_row_id

    max_total_replies = max_replies_per_comment * 50
    for reply_el in reply_elements:
        if reply_count >= max_total_replies:
            break

        try:
            raw = extract_comment_text(reply_el)
            text = clean_comment_text(raw)

            if not is_valid_comment(text):
                continue
            if text in seen_texts:
                continue
            seen_texts.add(text)

            likes = get_reaction_count(reply_el, REACTION_COUNT)

            rows.append(make_row(
                row_id=row_id,
                text=text,
                likes=likes,
                dislikes=0,
                source=source_name,
                relation=1,
                category=source_type,
            ))
            row_id += 1
            reply_count += 1

        except StaleElementReferenceException:
            continue
        except Exception as e:
            logger.debug("Reply parse error: %s", e)
            continue

    return reply_count

def _expand_reply_buttons(driver: webdriver.Chrome, max_rounds: int = 3) -> int:
    total_clicks = 0

    for round_num in range(max_rounds):
        fragments_xpath = " or ".join(
            f"contains(translate(normalize-space(.), "
            f"'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), "
            f"'{frag.lower()}')"
            for frag in VIEW_REPLIES_TEXT
        )
        hide_xpath = " or ".join(
            f"contains(translate(normalize-space(.), "
            f"'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), "
            f"'{frag.lower()}')"
            for frag in HIDE_REPLIES_TEXT
        )
        xpath = (
            f".//*[@role='button' or @tabindex='0']"
            f"[{fragments_xpath}]"
            f"[not({hide_xpath})]"
        )

        buttons = safe_find_all(driver, By.XPATH, xpath)
        reply_buttons = []
        for btn in buttons:
            try:
                btn_text = (btn.get_attribute("innerText") or btn.text or "").strip().lower()
                if any(h in btn_text for h in HIDE_REPLIES_TEXT):
                    continue
                if any(r in btn_text for r in VIEW_REPLIES_TEXT):
                    reply_buttons.append(btn)
            except StaleElementReferenceException:
                continue

        if not reply_buttons:
            break

        clicks_this_round = 0
        for btn in reply_buttons:
            try:
                scroll_element_into_view(driver, btn)
                if safe_click(driver, btn):
                    clicks_this_round += 1
                    short_sleep(0.6)
            except Exception:
                continue

        total_clicks += clicks_this_round
        logger.info("  Reply expansion round %d: clicked %d buttons",
                    round_num + 1, clicks_this_round)

        if clicks_this_round == 0:
            break
        short_sleep(1.0)

    return total_clicks

def expand_all_comments(driver: webdriver.Chrome, max_clicks: int = 15) -> None:
    scroll_to_bottom(driver)

    prev_count = 0
    for i in range(max_clicks):
        clicked = click_button_by_text(driver, VIEW_MORE_COMMENTS_TEXT)
        if not clicked:
            break

        deadline = time.time() + 3.0
        while time.time() < deadline:
            cur = len(safe_find_all(driver, By.CSS_SELECTOR, "[role='article']"))
            if cur > prev_count:
                prev_count = cur
                break
            time.sleep(0.4)

        scroll_to_bottom(driver)
        logger.debug("Loaded comment batch %d", i + 1)

def _get_comment_elements(driver: webdriver.Chrome) -> list:
    all_articles = safe_find_all(driver, By.CSS_SELECTOR, "[role='article']")
    if not all_articles:
        return []

    try:
        texts = driver.execute_script(
            "return arguments[0].map(function(el){ return (el.innerText||'').trim(); });",
            all_articles,
        )
    except Exception:
        texts = [""] * len(all_articles)

    result = []
    for el, text in zip(all_articles, texts):
        length = len(text or "")
        if 3 <= length <= 8000:
            result.append(el)

    return result

def _try_switch_to_all_comments(driver: webdriver.Chrome) -> bool:
    logger.info("  Switching to 'All comments' sort...")

    conditions = " or ".join(
        f"contains(translate(normalize-space(.), "
        f"'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{t.lower()}')"
        for t in _SORT_BUTTON_TEXT
    )
    sort_btns = safe_find_all(driver, By.XPATH, f"//*[@role='button'][{conditions}]")
    logger.info("  Sort button candidates: %d", len(sort_btns))

    for btn in sort_btns:
        try:
            label = (btn.get_attribute("innerText") or btn.text or "").strip()
            logger.info("  Trying: '%s'", label[:60])
            scroll_element_into_view(driver, btn)
            if safe_click(driver, btn):
                time.sleep(1.0)
                items = safe_find_all(driver, By.XPATH,
                                      "//*[@role='menuitem' or @role='option']")
                for item in items:
                    item_text = (item.get_attribute("innerText") or item.text or "").strip().lower()
                    if any(opt in item_text for opt in _ALL_COMMENTS_OPTION_TEXT):
                        logger.info("  Clicking option: '%s'", item_text[:60])
                        if safe_click(driver, item):
                            time.sleep(1.0)
                            logger.info("  ✓ Switched to All comments")
                            return True
                if click_button_by_text(driver, _ALL_COMMENTS_OPTION_TEXT):
                    time.sleep(1.0)
                    logger.info("  ✓ Switched to All comments (fallback)")
                    return True
        except Exception as e:
            logger.debug("  Sort switch error: %s", e)
            continue

    logger.info("  Could not switch sort — proceeding with default")
    return False
