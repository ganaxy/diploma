import os
import time
import shutil
import logging
from typing import List
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

logger = logging.getLogger("fb_scraper.browser")

FB_PROFILE_DIR = os.path.abspath("fb_profile")

def build_fb_driver(headless: bool = False, profile_dir: str = "") -> webdriver.Chrome:
    target_dir = profile_dir or FB_PROFILE_DIR
    os.makedirs(target_dir, exist_ok=True)

    options = Options()

    options.add_argument(f"--user-data-dir={target_dir}")

    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    options.add_argument("--lang=mn-MN,mn")
    options.add_argument("--window-size=1366,900")

    try:
        driver = webdriver.Chrome(options=options)
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
        )
        logger.info("Browser started (profile: %s, headless: %s)", target_dir, headless)
        return driver
    except Exception as e:
        logger.error("Failed to start browser: %s", e)
        raise

def prepare_worker_profiles(num_workers: int) -> List[str]:
    worker_dirs: List[str] = []
    for i in range(num_workers):
        worker_dir = os.path.abspath(f"fb_profile_worker_{i}")
        if os.path.exists(worker_dir):
            shutil.rmtree(worker_dir, ignore_errors=True)
        shutil.copytree(
            FB_PROFILE_DIR, worker_dir,
            ignore=shutil.ignore_patterns(
                'Cache', 'Code Cache', 'GPUCache', 'Service Worker',
                'ShaderCache', 'GrShaderCache', 'blob_storage',
            ),
            ignore_dangling_symlinks=True,
        )
        worker_dirs.append(worker_dir)
    logger.info("Created %d worker profile copies", num_workers)
    return worker_dirs

def cleanup_worker_profiles(worker_dirs: List[str]) -> None:
    for d in worker_dirs:
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception as e:
            logger.warning("Could not clean up %s: %s", d, e)
    logger.info("Cleaned up %d worker profiles", len(worker_dirs))

def check_logged_in(driver: webdriver.Chrome) -> bool:
    try:
        driver.get("https://www.facebook.com/")
        time.sleep(3)
        if "login" in driver.current_url.lower():
            return False
        page_src = driver.page_source
        if 'id="facebook"' in page_src or '"userID"' in page_src or "composer" in page_src.lower():
            return True
        return False
    except Exception as e:
        logger.warning("check_logged_in error: %s", e)
        return False

if __name__ == "__main__":
    """
    Run this file directly to perform the one-time manual login.
    Usage: python fb_browser.py
    """
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("=" * 60)
    print("  Facebook session setup — one-time login")
    print("=" * 60)
    print(f"  Profile directory: {FB_PROFILE_DIR}")
    print()
    print("  A browser window will open.")
    print("  Log in to your Facebook bot/test account.")
    print("  Then close this terminal window or press Ctrl+C.")
    print()

    drv = build_fb_driver(headless=False)
    drv.get("https://www.facebook.com/")
    print("  Browser is open. Log in now...")
    try:
        input("  Press ENTER after you have logged in to save the session.\n")
    except KeyboardInterrupt:
        pass
    print("  Session saved to:", FB_PROFILE_DIR)
    drv.quit()
    sys.exit(0)
