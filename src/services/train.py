from src.config import MODEL_PATH, METADATA_PATH, LABEL_NAMES

import argparse
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
from datetime import datetime, timezone
from time import perf_counter
import pandas as pd

def format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, remaining_minutes = divmod(int(minutes), 60)

    if hours:
        return (
            f"{hours}h {remaining_minutes}m "
            f"{remaining_seconds:.2f}s"
        )

    if minutes >= 1:
        return f"{int(minutes)}m {remaining_seconds:.2f}s"

    return f"{seconds:.2f}s"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if a saved model exists.",
    )

    args = parser.parse_args()

    if MODEL_PATH.exists() and not args.force:
        print(f"✓ Model already exists: {MODEL_PATH}")
        print("Skipping training.")
        print("Run with --force to retrain.")
        return
    elif MODEL_PATH.exists() and args.force:
        print("Forced retraining enabled!")

    print("Loading News for training")

    dataset = load_dataset("fancyzhx/ag_news")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]

    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=100_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    print(f"Training on {len(train_texts):,} articles...")

    training_started_at = datetime.now(timezone.utc)
    stopwatch_start = perf_counter()

    model.fit(train_texts, train_labels)

    training_seconds = perf_counter() - stopwatch_start
    training_duration = format_duration(training_seconds)

    print(f"Training completed in {training_duration}")

    print("Eval...")
    predictions = model.predict(test_texts)

    report = classification_report(
        test_labels,
        predictions,
        target_names=list(LABEL_NAMES.values()),
        digits=4,
        output_dict=True,
    )

    print(
        classification_report(
            test_labels,
            predictions,
            target_names=list(LABEL_NAMES.values()),
            digits=4,
        )
    )
    
    cm = confusion_matrix(test_labels, predictions)
    labels = list(LABEL_NAMES.values())
    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )
    
    cm_df.columns.name = "Predicted/Actual"

    print("Confusion matrix:")
    print(cm_df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "dataset": "fancyzhx/ag_news",
        "training_samples": len(train_texts),
        "test_samples": len(test_texts),
        "labels": LABEL_NAMES,
        "model": "LogisticRegression",
        "solver": "saga",
        "training_started_at": training_started_at.isoformat(),
        "training_seconds": round(training_seconds, 3),
        "training_duration": training_duration,
        "test_accuracy": round(report["accuracy"], 4),
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")

if __name__ == "__main__":
    main()
