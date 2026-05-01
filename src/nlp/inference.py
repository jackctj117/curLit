"""CB sentiment inference — load fine-tuned FinBERT and serve predictions."""

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

LABELS = {0: "dovish", 1: "neutral", 2: "hawkish"}


class CBSentimentModel:
    def __init__(
        self, model_path: Path | str, device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = Path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), num_labels=3,
        ).to(self.device)
        self.model.eval()

        # Temperature scaling (loaded if present)
        temp_path = model_path / "temperature.pt"
        self.temperature = 1.0
        if temp_path.exists():
            self.temperature = float(torch.load(temp_path, map_location="cpu"))

    @torch.no_grad()
    def predict(self, sentences: list[str], batch_size: int = 32) -> list[dict[str, Any]]:
        results = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits / self.temperature, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            for p, pred in zip(probs, preds, strict=False):
                results.append({
                    "prediction": LABELS[int(pred)],
                    "confidence": float(p[pred]),
                    "probs": {"dovish": float(p[0]), "neutral": float(p[1]), "hawkish": float(p[2])},
                    "hawkish_score": float(p[2] - p[0]),
                })
        return results

    @torch.no_grad()
    def predict_document(self, sentences: list[str]) -> dict[str, Any]:
        results = self.predict(sentences)
        scores = [r["hawkish_score"] for r in results]
        confs = [r["confidence"] for r in results]
        total_weight = sum(confs) or 1
        weighted_score = sum(s * c for s, c in zip(scores, confs, strict=False)) / total_weight

        counts = {"dovish": 0, "neutral": 0, "hawkish": 0}
        for r in results:
            counts[r["prediction"]] += 1

        return {
            "hawkish_score": weighted_score,
            "sentence_count": len(sentences),
            "pred_distribution": counts,
            "sentence_predictions": results,
        }
