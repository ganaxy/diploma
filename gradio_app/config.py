"""Static configuration for the Gradio inference app.

Holds absolute paths to the three trained MN-BERT model folders, English -> Mongolian
label display tables, and the curated example texts shown as one-click buttons in
the UI. Nothing in this module loads weights or touches the GPU.
"""

from pathlib import Path

# ── Project root and model paths (absolute, no copying) ──────────────────────
DIPLOM_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SCORES = DIPLOM_ROOT / "sample scores"

# Best-by-test-metric checkpoints. The Flat model now points at the
# retrained-on-corrected-data version (whose training script itself
# reported test_accuracy = 0.8326, test_macro_f1 = 0.8062 on the
# corrected v7 test split — that's a real training-script metric, not
# a post-hoc re-evaluation).
#
# Original references:
#   grid_ckpt_bs16_lr3e-05 — original SOTA, 0.8143 on original v7 test
#   best_mnbert_sota_corrected_model — retrained SOTA, 0.8326 on corrected test
#   best_stage1_model / best_stage2_model — Stage 1 (3-class) / Stage 2 (binary)
FLAT_MODEL_DIR = SAMPLE_SCORES / "best_mnbert_sota_corrected_model"
STAGE1_MODEL_DIR = SAMPLE_SCORES / "best_stage1_model"
STAGE2_MODEL_DIR = SAMPLE_SCORES / "best_stage2_model"

DATASET_CSV = SAMPLE_SCORES / "project" / "data" / "sampled_fully_labeled_new_version_v1.csv"

# Final (v7) split-tagged dataset CSV that the live Flat model was trained on.
# Columns used here: text_light_clean, text_normalized, source, label, split.
# Backs the random-sample button and the evaluation tab.
TEST_CSV = SAMPLE_SCORES / "relabeled_v7_corrected.csv"

# ── Per-model serving parameters (MUST mirror each training script) ──────────
# Flat model = best_mnbert_sota_corrected_model, trained by
# sample scores/retrain_sota_corrected.py with:
#     input_text = f"[{source}] {normalize_mongolian_v2(text_light_clean)}"
#     max_length = 256
# Serving therefore prepends the same source tag (we use the single most
# common training source, news.mn — 1369/5112 rows) and normalises the user
# text with the ported normalize_mongolian_v2 (see preprocess.py). Verified:
# faithful path scores 0.8326/0.8062 on the corrected test set (= thesis),
# whereas the old bare-text@128 path only scored 0.8136 — a ~1.9 pt skew.
FLAT_MAX_LENGTH = 256
FLAT_SOURCE_PREFIX = "[news.mn] "

# Stage 1 / Stage 2 training scripts (step2_train_stage1.py / step3_train_
# stage2.py) operated on text_light_clean with no source prefix; their
# generation scripts are no longer on disk, so we keep the light path and
# the conservative 128 length. This is documented as an explicit assumption.
TWO_STAGE_MAX_LENGTH = 128

# Backwards-compat alias (explain.py imports config.MAX_LENGTH).
MAX_LENGTH = FLAT_MAX_LENGTH

# ── Label mappings ───────────────────────────────────────────────────────────
# These checkpoints were saved with generic "LABEL_0..LABEL_n" id2label fields,
# so the *training scripts* are the authority for what each id means:
#   FLAT_LABELS    — step_focal_cosine_grid.py:54   LABEL_NAMES = ["POSITIVE","NEUTRAL","CONSTRUCTIVE","TOXIC"]
#   STAGE1_LABELS  — step2_train_stage1.py:56       LABEL_NAMES = ["POSITIVE","NEUTRAL","NEGATIVE"]
#   STAGE2_LABELS  — step3_train_stage2.py:57       LABEL_NAMES = ["CONSTRUCTIVE","TOXIC"]
FLAT_LABELS = ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"]
STAGE1_LABELS = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
STAGE2_LABELS = ["CONSTRUCTIVE", "TOXIC"]

# Display strings shown to the user (Mongolian, matching thesis Chapter 5 figure).
LABEL_MN = {
    "POSITIVE": "Эерэг",
    "NEUTRAL": "Саармаг",
    "NEGATIVE": "Сөрөг",
    "TOXIC": "Хортой сөрөг",
    "CONSTRUCTIVE": "Бүтээлч шүүмжлэл",
}

# Stable colour per class for charts and the highlight pill.
LABEL_COLOR = {
    "POSITIVE": "#2ca02c",
    "NEUTRAL": "#7f7f7f",
    "NEGATIVE": "#ff7f0e",
    "TOXIC": "#d62728",
    "CONSTRUCTIVE": "#1f77b4",
}

# ── Reported metrics (final dataset v7, thesis Abstract / Chapter 6) ─────────
# The dataset was built and iteratively refined over several annotation
# passes; these are the live Flat model's results on the final (v7) test
# split, faithful serving path ([source] + normalize @256).
FLAT_ACCURACY = 0.8326
FLAT_MACRO_F1 = 0.8062
N_TEST = 1529
STAGE2_ACCURACY = 0.9144

# ── Curated example texts (one per class, drawn from the labelled dataset) ──
# Real, sentence-length comments (not trivial one-liners) chosen to show the
# model discriminating genuinely ambiguous cases — e.g. constructive criticism
# that contains negative wording but proposes a concrete fix. Every example is
# verified to be predicted as its labelled class by BOTH the live Flat
# checkpoint and the Two-stage pipeline (see gradio_app/_smoke_test.py).
EXAMPLES = [
    ("Эерэг",
     "Ёстой их зориг шүү бүтэн 4 цаг далайд сэлнэ гэдэг төсөөлшгүй хүүхэд "
     "болохоор уян хатан цуцдаггүй юм байхдаа."),
    ("Саармаг",
     "Сайн байна уу. Та 11-11 дугаарт холбогдон мэдээлэл авах боломжтой."),
    ("Бүтээлч шүүмжлэл",
     "Сугалааны азтан тодорлоо, гээд байх юм үүнд итгэх хүн нэг их байхгүй л "
     "байх даа. Түүний оронд буцаан олголтоо 5% болгосон нь дээр."),
    ("Хортой сөрөг",
     "Ийм эргүү тэнэг, балмад хүмүүс байх ч гэж. Хэлэх муу үг алга."),
]

# ── CSV batch tab — accepted column names (auto-detect, first match wins) ────
TEXT_COLUMN_CANDIDATES = ["text", "comment", "text_light_clean", "text_raw", "comments"]
