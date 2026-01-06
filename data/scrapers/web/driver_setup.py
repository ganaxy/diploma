import sys
import shutil
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

def get_chrome_options(headless: bool = True) -> Options:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    options.add_argument("--lang=mn-MN")
    options.add_argument("--window-size=1920,1080")
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
        "profile.managed_default_content_settings.fonts": 2,
    }
    options.add_experimental_option("prefs", prefs)
    return options

def build_driver(headless: bool = True) -> webdriver.Chrome:
    options = get_chrome_options(headless=headless)

    try:
        driver = webdriver.Chrome(options=options)
        return driver
    except Exception as e1:
        print(f"  [Method 1 failed] Selenium auto-manager: {e1}")

    chromedriver_path = shutil.which("chromedriver")
    if chromedriver_path:
        try:
            service = Service(executable_path=chromedriver_path)
            driver = webdriver.Chrome(service=service, options=options)
            print(f"  Using chromedriver from PATH: {chromedriver_path}")
            return driver
        except Exception as e2:
            print(f"  [Method 2 failed] PATH chromedriver: {e2}")

    print("""
╔══════════════════════════════════════════════════════════════╗
║  ChromeDriver not found or wrong version                     ║
╠══════════════════════════════════════════════════════════════╣
║  FIX — Do ONE of the following:                              ║
║                                                              ║
║  Option A (Recommended — automatic):                         ║
║    pip install --upgrade selenium                            ║
║    (Selenium 4.10+ manages ChromeDriver automatically)       ║
║                                                              ║
║  Option B (Manual):                                          ║
║    1. Check your Chrome version:                             ║
║       Chrome menu → Help → About Google Chrome              ║
║    2. Download matching ChromeDriver for Windows:            ║
║       https://googlechromelabs.github.io/chrome-for-testing/ ║
║    3. Extract chromedriver.exe to this folder:               ║
║       C:\\Users\\M Tech\\Desktop\\diplom\\mongolian-scraper\\      ║
║    4. Run the script again                                   ║
╚══════════════════════════════════════════════════════════════╝
""")
    sys.exit(1)
