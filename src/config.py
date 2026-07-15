from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "news_classifier.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

LABEL_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Science/Technology"
}