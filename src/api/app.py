from src.config import MODEL_PATH, LABEL_NAMES
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import joblib

class Request(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)

class Response(BaseModel):
    category: str
    confidence: float
    scores: dict[str, float]

@asynccontextmanager
async def lifespan(_: FastAPI):
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    yield

    model = None

app = FastAPI(
    title="News Classifier",
    version="0.0.1",
    lifespan=lifespan
)

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}

@app.post("/api/predict", response_model=Response)
def predict(request: Request) -> Response:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    probabilities = model.predict_proba([request.text])[0]

    predicted_index = int(probabilities.argmax())

    scores = {
        LABEL_NAMES[index]: round(float(probability), 6)
        for index, probability in enumerate(probabilities)
    }

    return Response(
        category=LABEL_NAMES[predicted_index],
        confidence=round(float(probabilities[predicted_index]), 6),
        scores=scores
    )