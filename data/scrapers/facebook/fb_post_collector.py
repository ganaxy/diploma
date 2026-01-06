import re
import time
import logging
from datetime import datetime, timedelta
from typing import List, Set

from selenium import webdriver
from selenium.webdriver.common.by import By

from fb_utils import medium_sleep, normalize_fb_url, safe_find_all
from fb_selectors import PAGE_POST_LINK_PATTERNS, GROUP_POST_LINK_PATTERNS

logger = logging.getLogger("fb_scraper.post_collector")

_POST_HREF_RE = re.compile(
    r"facebook\.com/(?:[^/]+/(?:posts|videos|permalink|photos)|groups/[^/]+/(?:posts|permalink))/",
    re.IGNORECASE,
)
_EXCLUDE_HREF_RE = re.compile(
    r"(facebook\.com/login|/events/|/marketplace/|/watch/|/reels/|javascript:|mailto:|#)",
    re.IGNORECASE,
)

MAX_SCROLL_ATTEMPTS = 700

MONTHS_TO_TRY = 6

def collect_post_urls_from_page(
    driver: webdriver.Chrome,
    page_url: str,
    max_posts: int = 80,
    done_posts: set = None,
) -> List[str]:
    done_posts = done_posts or set()
    logger.info("Collecting from PAGE: %s  (need %d new)", page_url, max_posts)
    driver.get(page_url)
    medium_sleep(2.5)

    new_posts = _scroll_and_collect(driver, PAGE_POST_LINK_PATTERNS, max_posts, done_posts)

    if len(new_posts) < max_posts:
        remaining = max_posts - len(new_posts)
        logger.info("  Main feed yielded %d — trying month-based navigation for %d more",
                    len(new_posts), remaining)
        month_posts = _collect_from_monthly_feeds(
            driver, page_url, remaining, done_posts, set(new_posts),
        )
        new_posts.extend(month_posts)

    return new_posts[:max_posts]

def collect_post_urls_from_group(
    driver: webdriver.Chrome,
    group_url: str,
    max_posts: int = 80,
    done_posts: set = None,
) -> List[str]:
    done_posts = done_posts or set()
    logger.info("Collecting from GROUP: %s  (need %d new)", group_url, max_posts)
    driver.get(group_url)
    medium_sleep(3.0)
    selectors = GROUP_POST_LINK_PATTERNS + PAGE_POST_LINK_PATTERNS
    return _scroll_and_collect(driver, selectors, max_posts, done_posts)

def _collect_from_monthly_feeds(
    driver: webdriver.Chrome,
    page_url: str,
    max_posts: int,
    done_posts: set,
    already_found: Set[str],
) -> List[str]:
    new_posts: List[str] = []
    base_url = page_url.rstrip("/")

    now = datetime.now()
    for months_back in range(1, MONTHS_TO_TRY + 1):
        if len(new_posts) >= max_posts:
            break

        target_date = now - timedelta(days=30 * months_back)
        year = target_date.year
        month = target_date.month
        month_url = f"{base_url}/posts/?year={year}&month={month}"

        logger.info("  Trying monthly feed: %d/%d → %s", year, month, month_url)
        try:
            driver.get(month_url)
            medium_sleep(2.5)
        except Exception as e:
            logger.warning("  Failed to load %s: %s", month_url, e)
            continue

        month_found = _scroll_and_collect(
            driver, PAGE_POST_LINK_PATTERNS,
            max_posts=max_posts - len(new_posts),
            done_posts=done_posts,
            already_found=already_found,
            max_scrolls=80,
        )
        new_posts.extend(month_found)
        already_found.update(month_found)
        logger.info("  Month %d/%d → %d new posts", year, month, len(month_found))

    return new_posts[:max_posts]

def _scroll_and_collect(
    driver: webdriver.Chrome,
    selectors: List[str],
    max_posts: int,
    done_posts: set,
    already_found: Set[str] = None,
    max_scrolls: int = MAX_SCROLL_ATTEMPTS,
) -> List[str]:
    already_found = already_found or set()
    new_posts: List[str] = []
    seen_all:  Set[str]  = set(already_found)
    stale_scrolls = 0

    for scroll_num in range(max_scrolls):
        links = _extract_post_links(driver, selectors)
        added_any = False

        for url in links:
            if url not in seen_all:
                seen_all.add(url)
                added_any = True
                if url not in done_posts:
                    new_posts.append(url)

        if scroll_num % 5 == 0 or added_any:
            logger.info(
                "  Scroll %d — seen %d posts total, %d new (need %d)",
                scroll_num + 1, len(seen_all), len(new_posts), max_posts,
            )

        if len(new_posts) >= max_posts:
            logger.info("  Quota met: %d new posts collected", len(new_posts))
            break

        fraction_done = len(new_posts) / max_posts if max_posts > 0 else 1.0
        stale_threshold = max(4, int(4 + 16 * (1.0 - fraction_done)))

        if not added_any:
            stale_scrolls += 1
            logger.debug(
                "  Stale scroll %d/%d (collected %d/%d)",
                stale_scrolls, stale_threshold, len(new_posts), max_posts,
            )
            if stale_scrolls >= stale_threshold:
                logger.info(
                    "  Feed exhausted — %d stale scrolls at %.0f%% of quota. "
                    "Collected %d/%d new posts.",
                    stale_scrolls, fraction_done * 100, len(new_posts), max_posts,
                )
                break
        else:
            stale_scrolls = 0

        _scroll_and_wait_for_growth(driver)

    result = new_posts[:max_posts]
    skipped = len([u for u in seen_all if u in done_posts])
    logger.info(
        "  Done — collected %d new posts, skipped %d already-done",
        len(result), skipped,
    )
    return result

def _scroll_and_wait_for_growth(driver: webdriver.Chrome, timeout: float = 4.0) -> None:
    try:
        before_height = driver.execute_script("return document.body.scrollHeight")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(0.5)
            after_height = driver.execute_script("return document.body.scrollHeight")
            if after_height > before_height:
                return
    except Exception:
        time.sleep(1.5)

def _extract_post_links(driver: webdriver.Chrome, selectors: List[str]) -> Set[str]:
    found: Set[str] = set()
    anchors = []
    for sel in selectors:
        anchors.extend(safe_find_all(driver, By.CSS_SELECTOR, sel))
    anchors.extend(safe_find_all(driver, By.CSS_SELECTOR, "a[href]"))

    for a in anchors:
        try:
            href = a.get_attribute("href") or ""
        except Exception:
            continue
        if not href or _EXCLUDE_HREF_RE.search(href) or not _POST_HREF_RE.search(href):
            continue
        found.add(normalize_fb_url(href))

    return found
