"""Tests for the CBSentimentModel CL-v1x9 wiring.

We mock transformers.AutoModel + AutoTokenizer so tests don't download
~400MB of weights per CI run. The mocks return controllable logits
that exercise the 3-label and 4-label branches plus the document-
level aggregation path.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch


def _mock_model(num_labels: int, batch_logits: list[list[float]]) -> Any:
    """Build a mock that returns the given logits when called."""
    mock = MagicMock()
    mock.config.num_labels = num_labels
    # Build a tensor of shape (batch_size, num_labels) — batch_size
    # is len(batch_logits).
    tensor = torch.tensor(batch_logits, dtype=torch.float32)

    output = MagicMock()
    output.logits = tensor
    mock.return_value = output
    mock.eval = MagicMock()
    mock.to = MagicMock(return_value=mock)
    return mock


def _mock_tokenizer() -> Any:
    """Tokenizer mock — just returns dummy tensors of the right shape."""
    tok = MagicMock()

    def _call(batch: list[str], **_kw: Any) -> Any:
        result = MagicMock()
        # Return something with `.to(device)` -> self.
        result.to = MagicMock(
            return_value={
                "input_ids": torch.zeros(len(batch), 8, dtype=torch.long),
                "attention_mask": torch.ones(len(batch), 8, dtype=torch.long),
            }
        )
        return result

    tok.side_effect = _call
    return tok


@pytest.fixture
def patch_transformers():  # type: ignore[no-untyped-def]
    """Yield the (model, tokenizer) factories so a test can configure
    return values per case."""
    with (
        patch("src.nlp.inference.AutoModelForSequenceClassification") as MM,
        patch("src.nlp.inference.AutoTokenizer") as MT,
    ):
        yield MM, MT


class Test4LabelGtfintechlab:
    def test_loads_with_correct_label_map(self, patch_transformers) -> None:  # type: ignore[no-untyped-def]
        MM, MT = patch_transformers
        # 4-label model with hawkish-leaning logits on row 0.
        MM.from_pretrained.return_value = _mock_model(
            num_labels=4,
            batch_logits=[[0.1, 5.0, 0.2, 0.1]],  # strongly hawkish
        )
        MT.from_pretrained.return_value = _mock_tokenizer()

        from src.nlp.inference import CBSentimentModel

        model = CBSentimentModel(cb_name="fed", device="cpu")

        assert model.labels == {0: "neutral", 1: "hawkish", 2: "dovish", 3: "irrelevant"}
        assert "model_federal_reserve_system" in model.model_id

    def test_predict_yields_hawkish(self, patch_transformers) -> None:  # type: ignore[no-untyped-def]
        MM, MT = patch_transformers
        MM.from_pretrained.return_value = _mock_model(
            num_labels=4,
            batch_logits=[[0.1, 5.0, 0.2, 0.1]],
        )
        MT.from_pretrained.return_value = _mock_tokenizer()

        from src.nlp.inference import CBSentimentModel

        model = CBSentimentModel(cb_name="fed", device="cpu")
        out = model.predict(["The Committee will raise rates aggressively."])

        assert len(out) == 1
        assert out[0]["prediction"] == "hawkish"
        assert out[0]["hawkish_score"] > 0.5  # strongly hawkish
        assert "irrelevant" in out[0]["probs"]  # 4-label scheme exposed

    def test_predict_document_drops_irrelevant(self, patch_transformers) -> None:  # type: ignore[no-untyped-def]
        MM, MT = patch_transformers
        # Three sentences: one hawkish, one dovish, one irrelevant.
        MM.from_pretrained.return_value = _mock_model(
            num_labels=4,
            batch_logits=[
                [0.1, 5.0, 0.2, 0.1],  # hawkish
                [0.1, 0.2, 5.0, 0.1],  # dovish
                [0.1, 0.2, 0.1, 5.0],  # irrelevant
            ],
        )
        MT.from_pretrained.return_value = _mock_tokenizer()

        from src.nlp.inference import CBSentimentModel

        model = CBSentimentModel(cb_name="fed", device="cpu")
        out = model.predict_document(["a", "b", "c"], drop_irrelevant=True)

        assert out["sentence_count"] == 3
        # Irrelevant dropped → only 2 sentences scored.
        assert out["scoring_count"] == 2
        # Hawkish + dovish ~cancel → near-zero document score.
        assert abs(out["hawkish_score"]) < 0.5
        assert out["pred_distribution"]["irrelevant"] == 1


class Test3LabelLegacy:
    def test_legacy_3label_label_map(self, patch_transformers) -> None:  # type: ignore[no-untyped-def]
        MM, MT = patch_transformers
        MM.from_pretrained.return_value = _mock_model(
            num_labels=3,
            batch_logits=[[0.1, 0.1, 5.0]],  # hawkish=label_2 in legacy
        )
        MT.from_pretrained.return_value = _mock_tokenizer()

        from src.nlp.inference import CBSentimentModel

        # Pass model_path explicitly so we don't hit the registry.
        model = CBSentimentModel(model_path="some/local/path", device="cpu")
        assert model.labels == {0: "dovish", 1: "neutral", 2: "hawkish"}

        out = model.predict(["x"])
        assert out[0]["prediction"] == "hawkish"
        # Legacy probs dict has dovish/neutral/hawkish — no irrelevant.
        assert set(out[0]["probs"].keys()) == {"dovish", "neutral", "hawkish"}


class TestRegistry:
    def test_unknown_cb_falls_back_to_fed(self, caplog) -> None:  # type: ignore[no-untyped-def]
        from src.nlp.inference import CB_MODEL_REGISTRY, for_cb

        # Avoid actually loading a model — patch the module's transformers
        with (
            patch("src.nlp.inference.AutoModelForSequenceClassification") as MM,
            patch("src.nlp.inference.AutoTokenizer") as MT,
        ):
            MM.from_pretrained.return_value = _mock_model(
                num_labels=4,
                batch_logits=[[0.0] * 4],
            )
            MT.from_pretrained.return_value = _mock_tokenizer()
            with caplog.at_level("WARNING"):
                model = for_cb("riksbank")
            assert "No dedicated stance model" in caplog.text
            assert model.model_id == CB_MODEL_REGISTRY["fed"]

    def test_known_cb_uses_dedicated_model(self) -> None:
        from src.nlp.inference import CB_MODEL_REGISTRY

        for cb in ("fed", "ecb", "snb", "rba"):
            assert "model_" in CB_MODEL_REGISTRY[cb]
        # Fallbacks
        for cb in ("boe", "boj", "boc"):
            assert CB_MODEL_REGISTRY[cb] == CB_MODEL_REGISTRY["fed"]


class TestUnsupportedNumLabels:
    def test_raises_on_unexpected_label_count(self, patch_transformers) -> None:  # type: ignore[no-untyped-def]
        MM, MT = patch_transformers
        MM.from_pretrained.return_value = _mock_model(
            num_labels=5,
            batch_logits=[[0.0] * 5],
        )
        MT.from_pretrained.return_value = _mock_tokenizer()

        from src.nlp.inference import CBSentimentModel

        with pytest.raises(ValueError, match="num_labels=5"):
            CBSentimentModel(cb_name="fed", device="cpu")
