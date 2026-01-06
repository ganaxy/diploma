PAGE_POST_LINK_PATTERNS = [
    "a[href*='/posts/']",
    "a[href*='/permalink/']",
    "a[href*='/videos/']",
    "a[href*='/photo/']",
    "a[href*='/story_fbid=']",
]

GROUP_POST_LINK_PATTERNS = [
    "a[href*='/groups/'][href*='/posts/']",
    "a[href*='/groups/'][href*='/permalink/']",
]

COMMENT_SORT_BUTTON = [
    "div[aria-label='Comment ordering options']",
]

ALL_COMMENTS_OPTION = [
    "div[role='menuitem']",
]

VIEW_MORE_COMMENTS = [
    "span[class*='x1lliihq']",
]

COMMENT_LIST_CONTAINER = [
]

COMMENT_BLOCK = [
]

COMMENT_TEXT_SPANS = [
    "div[dir='auto'] span[dir='auto']",
    "div[dir='auto']",
    "span[dir='auto']",
]

REACTION_COUNT = [
]

VIEW_REPLIES_BUTTON = [
    "span[class*='x1i10hfl']",
]

VIEW_MORE_COMMENTS_TEXT = [
    "view more comments",
    "more comments",
    "бусад сэтгэгдэл",
    "сэтгэгдэл харах",
    "дараагийн сэтгэгдлүүд",
]

VIEW_REPLIES_TEXT = [
    "repl",
    "хариу харах",
    "хариу үзэх",
    "харах",
    "хариу",
]

HIDE_REPLIES_TEXT = [
    "hide",
    "нуух",
]

ALL_COMMENTS_TEXT = [
    "all comments",
    "бүх сэтгэгдэл",
    "newest first",
    "хамгийн шинэ",
]
