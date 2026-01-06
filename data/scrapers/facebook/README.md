# Facebook Comment Scraper — Diploma Project Extension

This module extends the existing Mongolian comment scraping pipeline
to collect comments from Facebook pages and groups.

---

## Architecture Overview

```
facebook_scraper/
├── fb_browser.py          # Browser/session setup — persistent Chrome profile
├── fb_sources.py          # Centralized source list (pages + groups)
├── fb_selectors.py        # ALL CSS/XPath selectors in one place (update here on DOM changes)
├── fb_utils.py            # Low-level DOM helpers, text extraction, reaction parsing
├── fb_post_collector.py   # Discovers post URLs from page/group feeds
├── fb_comment_scraper.py  # Scrapes comments + replies from a single post
├── fb_checkpoint.py       # Save/resume support + summary
└── fb_runner.py           # Main entry point — orchestrates the full pipeline
```

The module reuses:
- `utils.py` → `make_row()`, `save_to_excel()`, `normalize_whitespace()`
- `driver_setup.py` architecture pattern
- Same output schema: `id, text, likes, dislikes, SOURCE, RELATION, CATEGORY`
- Same done-URL/checkpoint pattern as `main.py`

---

## Data Schema

| Column   | Description                                  |
|----------|----------------------------------------------|
| id       | Row number (regenerated on save)             |
| text     | Comment or reply text (Mongolian/UTF-8)      |
| likes    | Reaction count (0 if none)                   |
| dislikes | Always 0 (Facebook has no dislike button)    |
| SOURCE   | Page/group name e.g. "IKON.mn"               |
| RELATION | 0 = top-level comment, 1 = reply             |
| CATEGORY | "news" / "government" / "community"          |

---

## Setup — One-Time Login

Facebook requires login. The session is stored in `fb_profile/` and reused.

**Step 1** — Install dependencies:
```bash
pip install selenium>=4.10.0 openpyxl pandas
```

**Step 2** — Run the login setup script:
```bash
cd facebook_scraper
python fb_browser.py
```

A Chrome window will open. Log in to your bot/test Facebook account.
Press ENTER in the terminal when done. The session is now saved to `fb_profile/`.

---

## Running the Scraper

```bash
# Basic run (uses all sources, default limits)
python fb_runner.py

# Headed mode (recommended — Facebook often blocks headless)
python fb_runner.py --headed

# Custom limits
python fb_runner.py --headed --max-posts 15 --max-comments 80 --max-replies 10

# Scrape specific sources only
python fb_runner.py --headed --sources "IKON.mn" "Gogo.mn"

# See all options
python fb_runner.py --help
```

Output is saved to `output/fb_comments.xlsx`.
Progress is checkpointed every 5 posts (configurable).
Done posts are tracked in `output/fb_done_posts.txt` — safe to re-run after crashes.

---

## File Tree

```
project_root/
├── main.py                       ← existing news scraper entry point
├── utils.py                      ← shared utils (reused by FB scraper)
├── driver_setup.py               ← shared driver builder (reference only)
├── output/
│   ├── mongolian_comments.xlsx   ← news scraper output
│   ├── fb_comments.xlsx          ← Facebook scraper output (new)
│   ├── fb_done_posts.txt         ← resume tracker (new)
│   └── fb_scraper.log            ← debug log (new)
├── fb_profile/                   ← Chrome session for Facebook login (new)
└── facebook_scraper/
    ├── fb_browser.py
    ├── fb_sources.py
    ├── fb_selectors.py
    ├── fb_utils.py
    ├── fb_post_collector.py
    ├── fb_comment_scraper.py
    ├── fb_checkpoint.py
    └── fb_runner.py
```

---

## Important: Reaction Filtering in utils.py

The existing `clean_dataframe()` in `utils.py` **drops rows where likes==0 AND dislikes==0**.

Facebook has **no dislike button**. Many comments will have 0 reactions.
If you merge the Facebook dataset into the main dataset and run `clean_dataframe()`,
you will lose most Facebook rows.

**Solution**: The Facebook scraper uses `fb_checkpoint.clean_dataframe_fb()` instead,
which does NOT filter by reactions.

If you ever combine datasets, keep the two Excel files separate, or modify
`clean_dataframe()` in `utils.py` to make the reaction filter optional.

---

## Selector Maintenance

If Facebook changes its DOM layout (which it does regularly), update `fb_selectors.py`.

Key things to check when selectors break:

1. **Comment elements not found** → Inspect a comment with Chrome DevTools (F12).
   Update `COMMENT_LIST_CONTAINER` and `COMMENT_BLOCK` in `fb_selectors.py`.

2. **"View more comments" not clicked** → Check `VIEW_MORE_COMMENTS_TEXT` list.
   Add the Mongolian button label that Facebook shows in your locale.

3. **Reaction counts always 0** → Inspect the reaction element.
   Update `REACTION_COUNT` selector list. Look for `aria-label` attributes.

4. **Replies not expanded** → Check `VIEW_REPLIES_TEXT` list.
   Add the exact text Facebook shows for "View X replies" in Mongolian.

5. **Headless mode gets login wall** → Always use `--headed` flag.
   Facebook actively detects headless Chrome.

---

## Known Limitations

1. **Facebook ToS**: Automated scraping violates Facebook Terms of Service.
   This scraper is for academic research purposes only.

2. **Login required**: Public pages technically don't require login to browse,
   but the comment section is often hidden without login.

3. **DOM instability**: Facebook rebuilds its frontend regularly.
   Expect to update selectors every few months.

4. **No dislikes**: Facebook removed the dislike button. `dislikes` is always 0.

5. **Reaction count accuracy**: Facebook sometimes lazy-loads reaction counts.
   If counts appear as 0 more than expected, try increasing `medium_sleep` delays.

6. **Groups**: Group posts may require group membership to view comments.
   "Мэдэхгүй зүйлээ асуу" must be joined with the scraper account.
