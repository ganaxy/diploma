import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "mn,en;q=0.9",
}
RE_CYRILLIC = re.compile(r'[\u0400-\u04FF]{3,}')

_session = requests.Session()
_session.headers.update(HEADERS)

def fetch(url, retries=2):
    for attempt in range(retries + 1):
        try:
            r = _session.get(url, timeout=(5, 8))
            r.raise_for_status()
            r.encoding = "utf-8"
            return BeautifulSoup(r.text, "lxml")
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
                continue
            print(f"    [fetch error] {url}: {e}")
            return None

NEWS_MN_CATEGORIES = [
    ("https://news.mn/angilal/uls-tur/",               "Улс төр"),
    ("https://news.mn/angilal/ediin-zasag/",            "Эдийн засаг"),
    ("https://news.mn/angilal/niigem/",                 "Нийгэм"),
    ("https://news.mn/angilal/delhii/",                 "Дэлхий"),
    ("https://news.mn/angilal/sport/",                  "Спорт"),
    ("https://news.mn/angilal/entertainment/",          "Соёл урлаг"),
    ("https://news.mn/angilal/niigem/eruul-mend/",      "Эрүүл мэнд"),
    ("https://news.mn/angilal/niigem/bolovsrol/",       "Боловсрол"),
    ("https://news.mn/angilal/ediin-zasag/business/",   "Бизнес"),
]

def news_mn_has_comments(url):
    soup = fetch(url)
    if not soup:
        return False
    divs = [d for d in soup.find_all("div", class_="comment")
            if "depth-" in " ".join(d.get("class", []))]
    for div in divs:
        ct = div.select_one("div.comment-text")
        if ct:
            for p in ct.select("p"):
                if not p.select("p") and RE_CYRILLIC.search(p.get_text()):
                    return True
    return False

def _check_url_for_comments(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=(5, 8))
        r.raise_for_status()
        r.encoding = "utf-8"
        soup = BeautifulSoup(r.text, "lxml")
    except Exception:
        return (url, False)
    divs = [d for d in soup.find_all("div", class_="comment")
            if "depth-" in " ".join(d.get("class", []))]
    for div in divs:
        ct = div.select_one("div.comment-text")
        if ct:
            for p in ct.select("p"):
                if not p.select("p") and RE_CYRILLIC.search(p.get_text()):
                    return (url, True)
    return (url, False)

def get_news_mn_urls(target=200, skip_urls=None):

    urls = []
    seen = set(skip_urls or [])
    per_cat = max(1, target // len(NEWS_MN_CATEGORIES))
    print(f"  Collecting news.mn URLs (target: {target}, ~{per_cat}/category, skipping {len(seen)} done)...")
    for cat_url, cat_name in NEWS_MN_CATEGORIES:
        if len(urls) >= target:
            break
        cat_count = 0
        page = 1
        bail = False
        while cat_count < per_cat and len(urls) < target and page <= MAX_PAGES_PER_CAT and not bail:
            listing = f"{cat_url}page/{page}/" if page > 1 else cat_url
            soup = fetch(listing)
            if not soup:
                break
            raw_links = {a["href"] for a in soup.find_all("a", href=re.compile(r"/r/\d+"))}

            if not raw_links:
                empty += 1
                if empty >= 2:
                    break
                page += 1
                continue

            empty = 0
            candidates = []
            for href in raw_links:
                full = (href if href.startswith("http") else f"https://news.mn{href}").split("?")[0]
                if href not in seen and full not in seen:
                    seen.add(href)
                    seen.add(full)
                    candidates.append(full)
            if not candidates:
                page += 1
                continue

            print(f"    [{cat_name}] page {page}: checking {len(candidates)} articles ({NEWS_CHECK_WORKERS} parallel)...")
            with ThreadPoolExecutor(max_workers=NEWS_CHECK_WORKERS) as pool:
                futures = {pool.submit(_check_url_for_comments, u): u for u in candidates}
                for future in as_completed(futures):
                    url_result, has = future.result()
                    if has:
                        dry_candidates = 0
                        if cat_count < per_cat and len(urls) < target:
                            cat_count += 1
                            urls.append((url_result, cat_name))
                            print(f"      ✓ {len(urls)}/{target} [{cat_name}] ({cat_count}/{per_cat})  {url_result}")
                    else:
                        dry_candidates += 1
                        if dry_candidates >= MAX_DRY_CANDIDATES:
                            print(f"    ⏭ [{cat_name}] {dry_candidates} articles checked with no comments — skipping rest")
                            bail = True
                            break

            page += 1
        if cat_count < per_cat:
            print(f"    ⚠ [{cat_name}] only found {cat_count}/{per_cat}")
            time.sleep(DELAY)
    print(f"  → {len(urls)} news.mn URLs")
    return urls[:target]

GOGO_MN_CATEGORIES = [
    ("https://gogo.mn/i/2",    "Улс төр"),
    ("https://gogo.mn/i/3",    "Эдийн засаг"),
    ("https://gogo.mn/i/4",    "Эрүүл мэнд"),
    ("https://gogo.mn/i/5",    "Соёл урлаг"),
    ("https://gogo.mn/i/6",    "Спорт"),
    ("https://gogo.mn/i/7",    "Нийгэм"),
    ("https://gogo.mn/i/8",    "Бизнес"),
    ("https://gogo.mn/i/9",    "Боловсрол"),
    ("https://gogo.mn/i/72",   "Дэлхий"),
    ("https://gogo.mn/i/6876", "Технологи"),
]

GOGO_COMMENT_SELECTOR    = "div.comment-item"
GOGO_COMMENT_TEXT_SEL    = "div.comment-body p"

def gogo_has_cyrillic_comments(driver, url):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    try:
        driver.get(url)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        try:
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "div.news-detail-comment-container")
                )
            )
        except Exception:
            return False

        from bs4 import BeautifulSoup as BS
        soup = BS(driver.page_source, "lxml")

        for sel, text_sel in [
            ("div.comment-item", "div.comment-body p"),
            ("div.comment-level-0", "div.comment-body p"),
            ("div.news-detail-comment-container div", "p"),
        ]:
            items = soup.select(sel)
            for item in items:
                t = item.select_one(text_sel)
                if t and RE_CYRILLIC.search(t.get_text()):
                    return True

        for el in soup.find_all(class_=re.compile("comment")):
            if RE_CYRILLIC.search(el.get_text()) and len(el.get_text(strip=True)) > 10:
                return True
    except Exception:
        pass
    return False

def get_gogo_mn_urls(target=200, skip_urls=None):
    from driver_setup import build_driver
    from selenium.webdriver.common.by import By
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

    RESTART_EVERY = 30
    MAX_DRY_CANDIDATES = 80

    _JS_GET_HREFS = 'return Array.from(document.querySelectorAll(\'a[href*="/r/"]\')).map(a=>a.href);'

    urls = []
    seen = set(skip_urls or [])
    driver = None
    articles_checked = 0
    per_cat = max(1, target // len(GOGO_MN_CATEGORIES))
    print(f"  Collecting gogo.mn URLs (target: {target}, ~{per_cat}/category, Selenium, skipping {len(seen)} done)...")

    def _force_kill(d):
        if not d:
            return
        try:
            d.service.process.kill()
        except Exception:
            pass
        try:
            d.quit()
        except Exception:
            pass

    def _rebuild():
        nonlocal driver
        _force_kill(driver)
        driver = build_driver(headless=True)
        driver.set_page_load_timeout(15)
        driver.set_script_timeout(10)

    def _safe_call(fn, *args, timeout=CALL_TIMEOUT, default=None):
        with ThreadPoolExecutor(1) as ex:
            future = ex.submit(fn, *args)
            try:
                return future.result(timeout=timeout)
            except (FutTimeout, Exception):
                return default

    try:
        driver = build_driver(headless=True)

        for cat_url, cat_name in GOGO_MN_CATEGORIES:
            if len(urls) >= target:
                break
            cat_count = 0
            dry_candidates = 0

            _rebuild()

            ok = _safe_call(driver.get, cat_url, timeout=20)
            time.sleep(3)
            test = _safe_call(driver.execute_script, "return 1;")
            if test is None:
                _rebuild()
                _safe_call(driver.get, cat_url, timeout=20)
                time.sleep(3)
                test = _safe_call(driver.execute_script, "return 1;")
                if test is None:
                    print(f"    [{cat_name}] failed to load category page — skipping")
                    continue

            raw = _safe_call(driver.execute_script, _JS_GET_HREFS)
            collected_hrefs = set(raw) if raw else set()
            prev_count = len(collected_hrefs)

            for click_num in range(MAX_LOAD_MORE_CLICKS):
                clicked = _safe_call(driver.execute_script, """
                    var btn = document.querySelector('a.btn-more');
                    if (btn) { btn.click(); return true; }
                    return false;
                """)
                if not clicked:
                time.sleep(2.5)

                raw = _safe_call(driver.execute_script, _JS_GET_HREFS)
                if raw:
                    collected_hrefs |= set(raw)
                else:

                if len(collected_hrefs) == prev_count:
                print(f"    [{cat_name}] load more #{click_num+1}: {prev_count} → {len(collected_hrefs)} articles")
                prev_count = len(collected_hrefs)

            all_candidates = []
            for href in collected_hrefs:
                full = href if href.startswith("http") else f"https://gogo.mn{href}"
                if href not in seen and full not in seen:
                    all_candidates.append((href, full))

            print(f"    [{cat_name}] {len(collected_hrefs)} total articles, {len(all_candidates)} new to check")

            if not all_candidates:
                print(f"    ⚠ [{cat_name}] only found 0/{per_cat}")
                continue

            _rebuild()

            for href, full in all_candidates:
                if cat_count >= per_cat or len(urls) >= target:
                    break

                seen.add(href)
                seen.add(full)
                articles_checked += 1

                if articles_checked % RESTART_EVERY == 0:
                    _rebuild()

                if gogo_has_cyrillic_comments(driver, full):
                    urls.append((full, cat_name))
                    cat_count += 1
                    dry_candidates = 0
                    print(f"    ✓ {len(urls)}/{target}  {full}  [{cat_name}] ({cat_count}/{per_cat})")
                else:
                    dry_candidates += 1
                    if dry_candidates >= MAX_DRY_CANDIDATES:
                        print(f"    ⏭ [{cat_name}] {dry_candidates} articles checked with no comments — skipping rest")
                        break

            if cat_count < per_cat:
                print(f"    ⚠ [{cat_name}] only found {cat_count}/{per_cat}")

    except Exception as e:
        print(f"  [gogo.mn error] {e}")
    finally:
        _force_kill(driver)

    print(f"  → {len(urls)} gogo.mn URLs")
    return urls[:target]
