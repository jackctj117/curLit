"""FastAPI service for CB sentiment inference."""

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from .inference import CBSentimentModel

app = FastAPI(title="curLit NLP Service", version="0.1.0")
model: CBSentimentModel | None = None


class ScoreRequest(BaseModel):
    sentences: list[str]


@app.on_event("startup")
def load_model() -> None:
    global model
    model_path = Path("/opt/fx-system/models/cb-sentiment-v1/final")
    if model_path.exists():
        model = CBSentimentModel(model_path)


@app.post("/score")
def score(req: ScoreRequest) -> list[dict[str, Any]]:
    if model is None:
        return [{"error": "model not loaded"}]
    return model.predict(req.sentences)


@app.post("/score_document")
def score_document(req: ScoreRequest) -> dict[str, Any]:
    if model is None:
        return {"error": "model not loaded"}
    return model.predict_document(req.sentences)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": model is not None}
