import os
import sys
import time
import queue
import logging
import argparse
import threading
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from selenium import webdriver

from fb_browser import (
    build_fb_driver, check_logged_in,
    prepare_worker_profiles, cleanup_worker_profiles,
)
from fb_sources import SOURCES
from fb_post_collector import collect_post_urls_from_page, collect_post_urls_from_group
from fb_comment_scraper import scrape_post_comments
from fb_checkpoint import (
    load_done_posts, mark_post_done, load_existing_rows,
    save_checkpoint, print_summary,
)
from fb_utils import normalize_fb_url

CONFIG = {
    "max_posts_per_source": 400,

    "max_comments_per_post": 200,

    "max_replies_per_comment": 10,

    "workers": 3,

    "save_every": 15,

    "restart_browser_every": 80,

    "headless": False,

    "post_delay": 1.0,

    "retry_failed_posts": True,
}

def setup_logging(log_file: str = "output/fb_scraper.log") -> None:
    os.makedirs("output", exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

logger = logging.getLogger("fb_scraper.runner")

class ScrapingCoordinator:

    def __init__(self, done_posts: set, all_rows: list, save_every: int):
        self._lock = threading.Lock()
        self.done_posts = done_posts
        self.all_rows = all_rows
        self.failed_posts = 0
        self._posts_since_save = 0
        self._save_every = save_every

    def mark_done(self, post_url: str, failed: bool = False) -> None:
        with self._lock:
            self.done_posts.add(post_url)
            mark_post_done(post_url)
            if failed:
                self.failed_posts += 1

    def add_rows(self, post_url: str, rows: list) -> None:
        with self._lock:
            self.all_rows.extend(rows)
            self.done_posts.add(post_url)
            mark_post_done(post_url)
            self._posts_since_save += 1
            if self._posts_since_save >= self._save_every:
                save_checkpoint(self.all_rows)
                self._posts_since_save = 0

    @property
    def total_rows(self) -> int:
        with self._lock:
            return len(self.all_rows)

def run_facebook_scraper(
    sources: Optional[List[dict]] = None,
    config: Optional[dict] = None,
) -> List[dict]:
    cfg = {**CONFIG, **(config or {})}
    active_sources = sources or SOURCES
    num_workers = cfg.get("workers", 1)

    setup_logging()
    logger.info("=" * 55)
    logger.info("Starting Facebook scraper")
    logger.info("Sources: %d | max_posts: %d | max_comments: %d | workers: %d",
                len(active_sources), cfg["max_posts_per_source"],
                cfg["max_comments_per_post"], num_workers)
    logger.info("=" * 55)

    if num_workers > 1:
        return _run_parallel(active_sources, cfg)
    else:
        return _run_sequential(active_sources, cfg)

def _run_sequential(
    active_sources: List[dict],
    cfg: dict,
) -> List[dict]:
    done_posts = load_done_posts()
    all_rows = load_existing_rows()
    if all_rows:
        logger.info("Resuming — %d existing rows, %d posts done", len(all_rows), len(done_posts))

    failed_posts = 0
    posts_since_save = 0
    posts_since_restart = 0
    driver = None

    try:
        driver = build_fb_driver(headless=cfg["headless"])

        if not check_logged_in(driver):
            logger.error(
                "\n" + "=" * 55 +
                "\n  NOT LOGGED IN TO FACEBOOK" +
                "\n  Run:  python fb_browser.py" +
                "\n  Then log in manually and re-run this script." +
                "\n" + "=" * 55
            )
            return all_rows

        logger.info("✓ Facebook session active")

        for source in active_sources:
            source_name = source["name"]
            source_type = source["type"]
            source_url  = source["url"]
            is_group    = source.get("is_group", False)

            logger.info("-" * 55)
            logger.info("SOURCE: %s (%s) → %s", source_name, source_type, source_url)

            try:
                if is_group:
                    post_urls = collect_post_urls_from_group(
                        driver, source_url,
                        max_posts=cfg["max_posts_per_source"],
                        done_posts=done_posts,
                    )
                else:
                    post_urls = collect_post_urls_from_page(
                        driver, source_url,
                        max_posts=cfg["max_posts_per_source"],
                        done_posts=done_posts,
                    )
            except Exception as e:
                logger.error("Failed to collect post URLs from %s: %s", source_url, e)
                continue

            logger.info("  %d new posts to scrape", len(post_urls))

            for post_url in post_urls:
                if posts_since_restart >= cfg["restart_browser_every"]:
                    logger.info("Restarting browser after %d posts", posts_since_restart)
                    try:
                        driver.quit()
                    except Exception:
                        pass
                    driver = build_fb_driver(headless=cfg["headless"])
                    posts_since_restart = 0
                    time.sleep(2.0)

                logger.info("  POST: %s", post_url)
                rows = _scrape_with_retry(
                    driver, post_url, source_name, source_type, cfg,
                )
                if rows is None:
                    failed_posts += 1
                    mark_post_done(post_url)
                    done_posts.add(post_url)
                    rows = []

                all_rows.extend(rows)
                done_posts.add(post_url)
                mark_post_done(post_url)
                posts_since_save += 1
                posts_since_restart += 1

                logger.info("  +%d comments | total: %d", len(rows), len(all_rows))

                if posts_since_save >= cfg["save_every"]:
                    save_checkpoint(all_rows)
                    posts_since_save = 0

                time.sleep(cfg["post_delay"])

    except KeyboardInterrupt:
        logger.info("\nScraping interrupted by user — saving progress...")
    except Exception as e:
        logger.exception("Unexpected error in main pipeline: %s", e)
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

    save_checkpoint(all_rows)
    print_summary(all_rows, failed_posts)
    return all_rows

def _run_parallel(
    active_sources: List[dict],
    cfg: dict,
) -> List[dict]:
    num_workers = cfg["workers"]

    done_posts = load_done_posts()
    all_rows = load_existing_rows()
    if all_rows:
        logger.info("Resuming — %d existing rows, %d posts done", len(all_rows), len(done_posts))

    logger.info("Phase 1 — Collecting post URLs (single browser)...")
    collector = build_fb_driver(headless=cfg["headless"])

    if not check_logged_in(collector):
        logger.error(
            "\n" + "=" * 55 +
            "\n  NOT LOGGED IN TO FACEBOOK" +
            "\n  Run:  python fb_browser.py" +
            "\n  Then log in manually and re-run this script." +
            "\n" + "=" * 55
        )
        collector.quit()
        return all_rows

    logger.info("✓ Facebook session active")

    for source in active_sources:
        source_name = source["name"]
        source_type = source["type"]
        source_url  = source["url"]
        is_group    = source.get("is_group", False)

        logger.info("-" * 55)
        logger.info("SOURCE: %s (%s) → %s", source_name, source_type, source_url)

        try:
            if is_group:
                post_urls = collect_post_urls_from_group(
                    collector, source_url,
                    max_posts=cfg["max_posts_per_source"],
                    done_posts=done_posts,
                )
            else:
                post_urls = collect_post_urls_from_page(
                    collector, source_url,
                    max_posts=cfg["max_posts_per_source"],
                    done_posts=done_posts,
                )
        except Exception as e:
            logger.error("Failed to collect post URLs from %s: %s", source_url, e)
            continue

        logger.info("  %d new posts to scrape", len(post_urls))
        for url in post_urls:
            post_tasks.append((url, source_name, source_type))

    collector.quit()
    logger.info("Phase 1 complete — %d posts queued for scraping", len(post_tasks))

    if not post_tasks:
        logger.info("No new posts to scrape.")
        save_checkpoint(all_rows)
        print_summary(all_rows, 0)
        return all_rows

    logger.info("Phase 2 — Parallel scraping with %d workers...", num_workers)

    worker_dirs = prepare_worker_profiles(num_workers)
    coordinator = ScrapingCoordinator(done_posts, all_rows, cfg["save_every"])
    shutdown_event = threading.Event()

    post_queue: queue.Queue = queue.Queue()
    for task in post_tasks:
        post_queue.put(task)

    try:
        with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="scraper") as pool:
            futures = [
                pool.submit(_worker_fn, i, worker_dirs[i], post_queue,
                            coordinator, shutdown_event, cfg)
                for i in range(num_workers)
            ]
            for f in futures:
                try:
                    f.result()
                except Exception as e:
                    logger.error("Worker raised: %s", e)

    except KeyboardInterrupt:
        logger.info("\nInterrupted — signalling workers to stop...")
        shutdown_event.set()
        time.sleep(3.0)

    finally:
        save_checkpoint(coordinator.all_rows)
        cleanup_worker_profiles(worker_dirs)

    print_summary(coordinator.all_rows, coordinator.failed_posts)
    return coordinator.all_rows

def _worker_fn(
    worker_id: int,
    profile_dir: str,
    post_queue: queue.Queue,
    coordinator: ScrapingCoordinator,
    shutdown_event: threading.Event,
    cfg: dict,
) -> None:
    MAX_CRASH_RESTARTS = 3
    wlog = logging.getLogger(f"fb_scraper.worker.{worker_id}")
    driver = None
    posts_scraped = 0
    crash_restarts = 0
    tag = f"[W{worker_id}]"

    try:
        driver = build_fb_driver(headless=cfg["headless"], profile_dir=profile_dir)
        wlog.info("%s Started (profile: %s)", tag, profile_dir)

        while not shutdown_event.is_set():
            try:
                post_url, source_name, source_type = post_queue.get(timeout=2.0)
            except queue.Empty:

            if posts_scraped > 0 and posts_scraped % cfg["restart_browser_every"] == 0:
                wlog.info("%s Restarting browser after %d posts", tag, posts_scraped)
                try:
                    driver.quit()
                except Exception:
                    pass
                driver = build_fb_driver(headless=cfg["headless"], profile_dir=profile_dir)
                time.sleep(2.0)

            wlog.info("%s POST: %s", tag, post_url)
            try:
                rows = _scrape_with_retry(driver, post_url, source_name, source_type, cfg)
            except Exception as e:
                wlog.error("%s Crash on %s: %s", tag, post_url, e)
                coordinator.mark_done(post_url, failed=True)
                crash_restarts += 1
                if crash_restarts > MAX_CRASH_RESTARTS:
                    wlog.error("%s Too many crashes (%d) — giving up", tag, crash_restarts)
                    break
                try:
                    if driver:
                        driver.quit()
                except Exception:
                    pass
                try:
                    driver = build_fb_driver(headless=cfg["headless"], profile_dir=profile_dir)
                    wlog.info("%s Restarted after crash (%d/%d)", tag, crash_restarts, MAX_CRASH_RESTARTS)
                    time.sleep(2.0)
                except Exception:
                    wlog.error("%s Could not restart browser — giving up", tag)
                    break
                continue

            if rows is None:
                coordinator.mark_done(post_url, failed=True)
                wlog.info("%s FAILED: %s", tag, post_url)
            else:
                coordinator.add_rows(post_url, rows)
                wlog.info("%s +%d rows | total: %d", tag, len(rows), coordinator.total_rows)

            posts_scraped += 1
            time.sleep(cfg["post_delay"])

    except Exception as e:
        wlog.error("%s Fatal error: %s", tag, e, exc_info=True)

    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
        wlog.info("%s Finished (%d posts scraped)", tag, posts_scraped)

def _scrape_with_retry(
    driver: webdriver.Chrome,
    post_url: str,
    source_name: str,
    source_type: str,
    cfg: dict,
) -> Optional[List[dict]]:
    attempts = 2 if cfg.get("retry_failed_posts", True) else 1
    for attempt in range(1, attempts + 1):
        try:
            return scrape_post_comments(
                driver=driver,
                post_url=post_url,
                source_name=source_name,
                source_type=source_type,
                max_comments=cfg["max_comments_per_post"],
                max_replies_per_comment=cfg.get("max_replies_per_comment", 10),
            )
        except Exception as e:
            logger.error("  [FAILED attempt %d/%d] %s — %s", attempt, attempts, post_url, e)
            if attempt < attempts:
                logger.info("  Retrying in 3s...")
                time.sleep(3.0)
    return None

def parse_args():
    p = argparse.ArgumentParser(
        description="Scrape Mongolian Facebook comments for diploma dataset"
    )
    p.add_argument(
        "--max-posts", type=int, default=CONFIG["max_posts_per_source"],
        help="Max NEW posts per source per run (default: %(default)s)"
    )
    p.add_argument(
        "--max-comments", type=int, default=CONFIG["max_comments_per_post"],
        help="Max comments per post (default: %(default)s)"
    )
    p.add_argument(
        "--save-every", type=int, default=CONFIG["save_every"],
        help="Save checkpoint every N posts (default: %(default)s)"
    )
    p.add_argument(
        "--headed", action="store_true", default=not CONFIG["headless"],
        help="Run in headed (visible) mode. Recommended for Facebook. "
             "Default: headed=True (headless=False) per CONFIG."
    )
    p.add_argument(
        "--workers", type=int, default=CONFIG["workers"],
        help="Number of parallel browser workers. "
             "1 = sequential mode, 2+ = parallel mode (default: %(default)s)"
    )
    p.add_argument(
        "--sources", nargs="+", metavar="NAME",
        help="Scrape only sources with these names (e.g. 'IKON.mn' 'Gogo.mn')"
    )
    p.add_argument(
        "--post-delay", type=float, default=CONFIG["post_delay"],
        help="Seconds to wait between posts per worker (default: %(default)s)"
    )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    run_config = {
        "max_posts_per_source":  args.max_posts,
        "max_comments_per_post": args.max_comments,
        "max_replies_per_comment": CONFIG["max_replies_per_comment"],
        "save_every":            args.save_every,
        "headless":              not args.headed,
        "post_delay":            args.post_delay,
        "restart_browser_every": CONFIG["restart_browser_every"],
        "workers":               args.workers,
        "retry_failed_posts":    CONFIG["retry_failed_posts"],
    }

    selected_sources = SOURCES
    if args.sources:
        selected_sources = [s for s in SOURCES if s["name"] in args.sources]
        if not selected_sources:
            print(f"[ERROR] No sources matched: {args.sources}")
            print(f"Available: {[s['name'] for s in SOURCES]}")
            sys.exit(1)

    run_facebook_scraper(sources=selected_sources, config=run_config)
