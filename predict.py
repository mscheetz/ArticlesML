from config import MODEL_PATH, LABEL_NAMES

import joblib

def classify_article(model, text: str) -> dict[str, float | str]:
    probabilities = model.predict_proba([text])[0]
    predicted_label = int(probabilities.argmax())

    scores = {
        LABEL_NAMES[index]: round(float(probability),4)
        for index, probability in enumerate(probabilities)
    }

    return {
        "category": LABEL_NAMES[predicted_label],
        "confidence": round(float(probabilities[predicted_label]), 4),
        "scores": scores
    }

def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run `uv python train.py` first")

    model = joblib.load(MODEL_PATH)

    print("News Classifier")
    print("Paste a headline or article. Enter `quit` to exit")

    while True:
        text = input("\nHeadline or Article text: ").strip()

        if text.lower() in {"quit", "exit"}:
            print("\nCome back soon!")
            break

        if not text:
            print("Please enter some text")
            continue

        result = classify_article(model, text)

        print(f"\nPrediction: {result['category']}")
        print(f"Confidence: {result['confidence']:.1%}")

        print("\nAll category scores:")
        for category, score in sorted(
            result["scores"].items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"  {category:<20} {score:.1%}")

if __name__ == "__main__":
    main()