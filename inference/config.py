from pathlib import Path

DIPLOM_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SCORES = DIPLOM_ROOT / "sample scores"

FLAT_MODEL_DIR = SAMPLE_SCORES / "grid_ckpt_bs16_lr3e-05"
STAGE1_MODEL_DIR = SAMPLE_SCORES / "best_stage1_model"
STAGE2_MODEL_DIR = SAMPLE_SCORES / "best_stage2_model"

DATASET_CSV = SAMPLE_SCORES / "project" / "data" / "sampled_fully_labeled_new_version_v1.csv"

MAX_LENGTH = 128

FLAT_LABELS = ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"]
STAGE1_LABELS = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
STAGE2_LABELS = ["CONSTRUCTIVE", "TOXIC"]

LABEL_MN = {
    "POSITIVE": "Эерэг",
    "NEUTRAL": "Саармаг",
    "NEGATIVE": "Сөрөг",
    "TOXIC": "Хортой сөрөг",
    "CONSTRUCTIVE": "Бүтээлч шүүмжлэл",
}

LABEL_COLOR = {
    "POSITIVE": "#2ca02c",
    "NEUTRAL": "#7f7f7f",
    "NEGATIVE": "#ff7f0e",
    "TOXIC": "#d62728",
    "CONSTRUCTIVE": "#1f77b4",
}

FLAT_ACCURACY = 0.8143
FLAT_MACRO_F1 = 0.7760
STAGE2_ACCURACY = 0.9144

EXAMPLES = [
    ("Эерэг", "Үнэн ялна аа. Баяр хүргэе"),
    ("Саармаг", "Энэ хэзээ зүгээр болох вэ"),
    ("Бүтээлч шүүмжлэл", "Нөөцийн мах бэлддэг асуудлыг болих хэрэгтэй. Зах зээлд нь даатгаад үлдээ."),
    ("Хортой сөрөг", "Тийм шүү ухаан муутай мал"),
]

TEXT_COLUMN_CANDIDATES = ["text", "comment", "text_light_clean", "text_raw", "comments"]
