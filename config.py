from pathlib import Path

MODEL_PATH = Path("models/news_classifier.joblib")
METADATA_PATH = Path("models/metadata.json")

LABEL_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Science/Technology"
}