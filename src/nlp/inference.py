"""CB sentiment / stance inference (CL-v1x9).

Wraps Hugging Face `transformers` to score CB statements as
hawkish / dovish / neutral / irrelevant. Two model shapes are
supported:

  4-label gtfintechlab stance models (preferred, current SOTA):
    - LABEL_0 = neutral, LABEL_1 = hawkish, LABEL_2 = dovish,
      LABEL_3 = irrelevant
    - Per-CB models (Fed / ECB / SNB / PBoC / RBA / RBI / MAS).
    - Released 2025-08 by gtfintechlab; the "irrelevant" class is
      useful because CB statements include boilerplate sentences
      that should be excluded from the document-level score.

  3-label legacy locally-fine-tuned FinBERT:
    - LABEL_0 = dovish, LABEL_1 = neutral, LABEL_2 = hawkish
    - Loaded from a local checkpoint directory; tested in the
      original FinBERT scaffold. Kept for back-compat; new
      deployments should use the 4-label HF models.

The class auto-detects shape from the loaded model's `num_labels`.

Document-level scoring aggregates per-sentence predictions weighted by
confidence, with "irrelevant" sentences dropped (4-label only). The
final score is in [-1, +1]: -1 = uniformly dovish, +1 = uniformly
hawkish, 0 = neutral or balanced.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


# 4-label scheme used by gtfintechlab stance models.
_GTFINTECHLAB_LABELS: dict[int, str] = {
    0: "neutral",
    1: "hawkish",
    2: "dovish",
    3: "irrelevant",
}
# 3-label scheme used by the legacy locally-fine-tuned FinBERT.
_LEGACY_LABELS: dict[int, str] = {0: "dovish", 1: "neutral", 2: "hawkish"}


# CB short name → Hugging Face model ID. CBs without a dedicated
# stance model fall back to the Fed model — Fed-trained classifiers
# generalize reasonably to other English-speaking CBs because the
# vocabulary of monetary-policy text is highly stylized.
CB_MODEL_REGISTRY: dict[str, str] = {
    "fed": "gtfintechlab/model_federal_reserve_system_stance_label",
    "ecb": "gtfintechlab/model_european_central_bank_stance_label",
    "pboc": "gtfintechlab/model_peoples_bank_of_china_stance_label",
    "snb": "gtfintechlab/model_swiss_national_bank_stance_label",
    "rba": "gtfintechlab/model_reserve_bank_of_australia_stance_label",
    "rbi": "gtfintechlab/model_reserve_bank_of_india_stance_label",
    "mas": "gtfintechlab/model_monetary_authority_of_singapore_stance_label",
    "nbp": "gtfintechlab/model_national_bank_of_poland_stance_label",
    # No dedicated model — fall back to Fed. (BoE / BoJ / BoC speak
    # similar policy English; calibration is good-enough for v1, and
    # CL-v1x9-followup can fine-tune dedicated models on each CB's
    # archive once we have enough labeled samples.)
    "boe": "gtfintechlab/model_federal_reserve_system_stance_label",
    "boj": "gtfintechlab/model_federal_reserve_system_stance_label",
    "boc": "gtfintechlab/model_federal_reserve_system_stance_label",
}

# Default model when no CB context is supplied — the broadest
# pre-trained CB classifier we have.
DEFAULT_MODEL_ID: str = CB_MODEL_REGISTRY["fed"]


class CBSentimentModel:
    """Per-CB stance classifier. Auto-detects 3-vs-4-label scheme."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_ID,
        device: str | None = None,
        cb_name: str | None = None,
    ) -> None:
        """Load model from a local path or a Hugging Face model ID.

        Args:
          model_path: local checkpoint directory OR HF model ID. If
                      ``cb_name`` is given and present in CB_MODEL_REGISTRY,
                      that mapping wins over ``model_path``.
          cb_name: optional CB short name ("fed" / "ecb" / etc) — looked
                   up in CB_MODEL_REGISTRY.
          device: "cpu" / "cuda" / None for auto-detect.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cb_name = cb_name

        if cb_name and cb_name.lower() in CB_MODEL_REGISTRY:
            resolved = CB_MODEL_REGISTRY[cb_name.lower()]
        else:
            resolved = str(model_path)
        self.model_id = resolved

        # Tokenizer kwargs match the gtfintechlab recommendation
        # (do_lower_case + do_basic_tokenize). Local 3-label models
        # ignore them gracefully.
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved,
            do_lower_case=True,
            do_basic_tokenize=True,
        )

        # num_labels lets transformers re-shape the head to match
        # the loaded weights' label count. It auto-detects from the
        # config, so we don't pass num_labels here.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            resolved,
        ).to(self.device)
        self.model.eval()

        n_labels = int(self.model.config.num_labels)
        if n_labels == 4:
            self.labels = _GTFINTECHLAB_LABELS
        elif n_labels == 3:
            self.labels = _LEGACY_LABELS
        else:
            msg = (
                f"Unexpected num_labels={n_labels} for {resolved}; "
                f"this class supports the 3-label legacy FinBERT and "
                f"the 4-label gtfintechlab schemes."
            )
            raise ValueError(msg)

        # Optional temperature scaling — only present for the legacy
        # locally-trained FinBERT checkpoints. HF hub models don't
        # ship a temperature.pt sidecar.
        self.temperature = 1.0
        if isinstance(model_path, Path | str):
            mp = Path(str(model_path))
            if mp.exists():
                temp_path = mp / "temperature.pt"
                if temp_path.exists():
                    self.temperature = float(
                        torch.load(temp_path, map_location="cpu"),
                    )

    @torch.no_grad()
    def predict(
        self,
        sentences: list[str],
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        """Per-sentence prediction. Returns one dict per input sentence.

        ``hawkish_score`` is in [-1, +1] mapping
        prob(hawkish) - prob(dovish). For the 4-label scheme, the
        "irrelevant" probability is reported separately so document-
        level aggregation can drop those sentences cleanly.
        """
        results: list[dict[str, Any]] = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits / self.temperature, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            for p, pred in zip(probs, preds, strict=False):
                pred_label = self.labels[int(pred)]
                row: dict[str, Any] = {
                    "prediction": pred_label,
                    "confidence": float(p[pred]),
                }
                # Build the per-class probability dict in a label-aware
                # way so the caller doesn't have to know whether this
                # is a 3-label or 4-label model.
                row["probs"] = {
                    label_name: float(p[idx]) for idx, label_name in self.labels.items()
                }
                # Hawkish score in [-1, +1]. "irrelevant" doesn't
                # contribute (4-label) — we just compare hawkish vs dovish.
                row["hawkish_score"] = row["probs"].get("hawkish", 0.0) - row["probs"].get(
                    "dovish", 0.0
                )
                results.append(row)
        return results

    @torch.no_grad()
    def predict_document(
        self,
        sentences: list[str],
        drop_irrelevant: bool = True,
    ) -> dict[str, Any]:
        """Document-level aggregation. Confidence-weighted average of
        per-sentence hawkish_score, irrelevant sentences dropped (4-
        label only)."""
        per_sentence = self.predict(sentences)

        if drop_irrelevant:
            scoring = [r for r in per_sentence if r["prediction"] != "irrelevant"]
        else:
            scoring = per_sentence

        if not scoring:
            return {
                "hawkish_score": 0.0,
                "sentence_count": len(sentences),
                "scoring_count": 0,
                "pred_distribution": {label: 0 for label in self.labels.values()},
                "sentence_predictions": per_sentence,
                "model_id": self.model_id,
            }

        total_weight = sum(r["confidence"] for r in scoring) or 1.0
        weighted = sum(r["hawkish_score"] * r["confidence"] for r in scoring) / total_weight

        counts = {label: 0 for label in self.labels.values()}
        for r in per_sentence:
            counts[r["prediction"]] += 1

        return {
            "hawkish_score": weighted,
            "sentence_count": len(sentences),
            "scoring_count": len(scoring),
            "pred_distribution": counts,
            "sentence_predictions": per_sentence,
            "model_id": self.model_id,
        }


def for_cb(cb_name: str, device: str | None = None) -> CBSentimentModel:
    """Convenience: build a model for a known CB.

    Looks up CB_MODEL_REGISTRY. Unknown names fall through to the
    default (Fed) classifier with a WARN — the call still succeeds
    with reasonable behavior, but operator should know about it.
    """
    key = cb_name.lower()
    if key not in CB_MODEL_REGISTRY:
        logger.warning(
            "No dedicated stance model for CB '%s'; falling back to Fed model. "
            "Add a mapping to CB_MODEL_REGISTRY when a dedicated "
            "model becomes available.",
            cb_name,
        )
    return CBSentimentModel(cb_name=cb_name, device=device)
