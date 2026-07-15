# ArticlesML

News headline/article classifier — 4 categories: World, Sports, Business, Science/Technology.

Uses TF-IDF + Logistic Regression on the AG News dataset. **92.24% test accuracy.**

## Setup

```bash
uv sync
```

## Train

```bash
uv run train.py
```

Downloads AG News (~120k training samples) and trains a TF-IDF + Logistic Regression pipeline. Saves model to `models/news_classifier.joblib`.

Retrain (overwrite existing model):

```bash
uv run train.py --force
```

Training output:
```bash
Loading News for training
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Training on 120,000 articles...
Training completed in 1m 8.79s
Eval...
                    precision    recall  f1-score   support

             World     0.9372    0.9111    0.9239      1900
            Sports     0.9559    0.9821    0.9688      1900
          Business     0.8980    0.8900    0.8940      1900
Science/Technology     0.8978    0.9063    0.9020      1900

          accuracy                         0.9224      7600
         macro avg     0.9222    0.9224    0.9222      7600
      weighted avg     0.9222    0.9224    0.9222      7600

Confusion matrix:
Predicted/Actual    World  Sports  Business  Science/Technology
World                1731      53        69                  47
Sports                 15    1866        14                   5
Business               51      14      1691                 144
Science/Technology     50      19       109                1722
```

## Predict

```bash
uv run predict.py
```

Interactive mode — paste a headline or article text. Shows predicted category, confidence, and per-category scores.

Prediction example:
```bash
News Classifier
Paste a headline or article. Enter `quit` to exit

Headline or Article text: JOHANNESBURG, July 15 (Reuters) - Amazon's (AMZN.O), opens new tab low-earth orbit satellite internet venture Amazon Leo has signed an agreement with South Africa's Herotel to launch a ​new broadband service aimed at connecting underserved rural communities, it ‌said on Wednesday.Under the agreement, Herotel, South Africa's largest fixed internet service provider, will use Amazon Leo's satellite technology to offer a new service called evry, which ​is expected to launch commercially in 2027 for residential customers.

Prediction: Science/Technology
Confidence: 95.8%

All category scores:
  Science/Technology   95.8%
  Business             3.8%
  World                0.4%
  Sports               0.1%
```

## Test samples

`tests/` contains 5 sample headlines you can paste into the predictor:

```
The central bank raised interest rates following another month of high inflation.
The team secured the championship after scoring twice in the final minutes.
Researchers unveiled a new processor designed for artificial intelligence workloads.
Diplomats met Wednesday to negotiate a ceasefire between the two countries.
President meets delegates in Geneva.
```

## Model

| Component | Detail |
|-----------|--------|
| Dataset | fancyzhx/ag_news |
| Vectorizer | TfidfVectorizer (1-2 ngrams, sublinear tf, max 100k features) |
| Classifier | LogisticRegression (lbfgs, 1000 max iter) |
| Training samples | 120,000 |
| Test samples | 7,600 |
| Test accuracy | 92.24% |

## FastAPI

Start the API server:

```bash
uv run uvicorn src.api.app:app --reload
```

### Endpoints

`GET /api/health` — health check

```json
{ "status": "healthy" }
```

`POST /api/predict` — classify a headline or article

Request:

```json
{ "text": "Your headline or article text here" }
```

Response:

```json
{
  "category": "Science/Technology",
  "confidence": 0.958,
  "scores": {
    "Business": 0.038,
    "Science/Technology": 0.958,
    "Sports": 0.001,
    "World": 0.004
  }
}
```

### API tests

`tests/tests.http` contains HTTP request samples for each category. Use with VS Code REST Client extension or IntelliJ HTTP Client.

## Docker

```bash
docker build -t articles-ml .
docker run --rm -p 8000:8000 articles-ml
```