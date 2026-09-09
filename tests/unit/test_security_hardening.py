"""CL-u59z: hostile-input fixtures, not scanner-output assertions."""

from __future__ import annotations

import pickle
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from defusedxml.common import DefusedXmlException
from sqlalchemy import create_engine, text

from src.data.alternative import EntsoeSource
from src.data.truth_social import parse_feed
from src.research.ingest import ArxivFetcher, RSSFetcher


@pytest.mark.parametrize("declaration", [
    '<!DOCTYPE root [<!ENTITY injected "untrusted">]>',
    '<!DOCTYPE root [<!ENTITY injected SYSTEM "file:///must-not-read">]>',
    '<!DOCTYPE root SYSTEM "https://must-not-fetch.invalid/schema.dtd">',
])
def test_external_xml_rejects_dtd(declaration: str) -> None:
    xml = declaration + "<root/>"
    for parse in (ArxivFetcher._parse, RSSFetcher._parse):
        with pytest.raises(DefusedXmlException):
            parse(xml, source_label="fixture")
    with pytest.raises(DefusedXmlException):
        EntsoeSource._parse_load_xml(xml, "DE")
    assert parse_feed(xml) == []


class _UnexpectedCheckpointObject:
    """Not a tensor or scalar; never allow arbitrary checkpoint classes."""


@pytest.mark.parametrize("value", [torch.tensor(2.5), 2.5])
def test_temperature_preserves_tensor_and_scalar(value: object, tmp_path: Path) -> None:
    from src.nlp.inference import CBSentimentModel

    torch.save(value, tmp_path / "temperature.pt")
    model = MagicMock()
    model.to.return_value = model
    model.config.num_labels = 3
    with (
        patch("src.nlp.inference.AutoTokenizer.from_pretrained"),
        patch("src.nlp.inference.AutoModelForSequenceClassification.from_pretrained",
              return_value=model),
    ):
        loaded = CBSentimentModel(tmp_path, device="cpu")
    assert loaded.temperature == 2.5


def test_temperature_rejects_arbitrary_pickle(tmp_path: Path) -> None:
    from src.nlp.inference import CBSentimentModel

    torch.save(_UnexpectedCheckpointObject(), tmp_path / "temperature.pt")
    model = MagicMock()
    model.to.return_value = model
    model.config.num_labels = 3
    with (
        patch("src.nlp.inference.AutoTokenizer.from_pretrained"),
        patch("src.nlp.inference.AutoModelForSequenceClassification.from_pretrained",
              return_value=model),
        pytest.raises(pickle.UnpicklingError, match="Weights only load failed"),
    ):
        CBSentimentModel(tmp_path, device="cpu")


def test_idea_id_remains_compatible() -> None:
    from src.events.idea_ledger import make_idea_id

    # Independent oracle: printf %s '123:AAPL:buy_calls' | openssl dgst -sha1.
    assert make_idea_id(123, "AAPL", "buy_calls") == "c5d76600c6cd695d"


@pytest.mark.parametrize("cb", ["fed", "ecb", "pboc", "snb", "rba", "rbi", "mas", "nbp"])
def test_both_model_artifacts_receive_same_pin(cb: str) -> None:
    from src.nlp.inference import CB_MODEL_REGISTRY, CB_MODEL_REVISIONS, CBSentimentModel

    model = MagicMock()
    model.to.return_value = model
    model.config.num_labels = 4
    with (
        patch("src.nlp.inference.AutoTokenizer.from_pretrained") as tokenizer,
        patch("src.nlp.inference.AutoModelForSequenceClassification.from_pretrained",
              return_value=model) as weights,
    ):
        loaded = CBSentimentModel(cb_name=cb, device="cpu")
    expected = CB_MODEL_REVISIONS[CB_MODEL_REGISTRY[cb]]
    assert len(expected) == 40 and int(expected, 16) > 0
    assert loaded.model_revision == expected
    for loader in (tokenizer, weights):
        assert loader.call_args.kwargs["revision"] == expected
        assert loader.call_args.kwargs["trust_remote_code"] is False


@pytest.mark.parametrize("revision", [None, "main", "v1", "abc1234", "z" * 40])
def test_unpinned_remote_model_rejected_before_load(revision: str | None) -> None:
    from src.nlp.inference import CBSentimentModel

    with (
        patch("src.nlp.inference.AutoTokenizer.from_pretrained") as loader,
        pytest.raises(ValueError, match="immutable"),
    ):
        CBSentimentModel("unregistered/model", device="cpu", revision=revision)
    loader.assert_not_called()


def test_missing_local_model_cannot_download(tmp_path: Path) -> None:
    from src.nlp.inference import CBSentimentModel

    with (
        patch("src.nlp.inference.AutoTokenizer.from_pretrained",
              side_effect=OSError("missing local checkpoint")) as loader,
        pytest.raises(OSError, match="missing local"),
    ):
        CBSentimentModel(tmp_path / "missing", device="cpu")
    assert loader.call_args.kwargs["local_files_only"] is True
    assert loader.call_args.kwargs["revision"] is None


@pytest.mark.parametrize("book", ["options", "equities"])
@pytest.mark.parametrize("require_niche", [False, True])
@pytest.mark.parametrize("require_red_team", [False, True])
def test_entry_queries_bind_confidence(
    book: str, require_niche: bool, require_red_team: bool,
) -> None:
    from src.execution.alpaca_equity_executor import (
        EquityExecConfig,
    )
    from src.execution.alpaca_equity_executor import (
        fetch_executable_ideas as equity_ideas,
    )
    from src.execution.alpaca_options_executor import (
        OptionsExecConfig,
    )
    from src.execution.alpaca_options_executor import (
        fetch_executable_ideas as option_ideas,
    )

    engine = MagicMock()
    execute = engine.connect.return_value.__enter__.return_value.execute
    execute.return_value = []
    cfg = OptionsExecConfig() if book == "options" else EquityExecConfig()
    # CL-0deu.9 adds a constructor gate; independently retain the SQL-binding
    # oracle for a forged/deserialized object that bypasses that first defense.
    payload = "0; DROP TABLE trade_ideas; --"
    with pytest.raises(ValueError, match="min_confidence"):
        replace(cfg, min_confidence=payload)
    cfg = replace(cfg, require_niche=require_niche, require_red_team=require_red_team)
    object.__setattr__(cfg, "min_confidence", payload)
    if isinstance(cfg, OptionsExecConfig):
        option_ideas(engine, cfg)
    else:
        equity_ideas(engine, cfg)
    query, params = execute.call_args.args
    assert payload not in str(query)
    assert params == {"min_conf": payload}
    assert ("LIKE '%niche%'" in str(query)) == require_niche
    assert ("LIKE '%red-team%'" in str(query)) == require_red_team


@pytest.mark.parametrize("bounded", [False, True])
def test_provider_queries_keep_hostile_symbol_bound(bounded: bool) -> None:
    import pandas as pd

    from src.data.provider import DataProvider

    engine = MagicMock()
    provider = DataProvider(engine)
    payload = "EURUSD'; DROP TABLE prices; --"
    now = datetime(2026, 9, 8, tzinfo=UTC)
    with patch("src.data.provider.pd.read_sql", return_value=pd.DataFrame()) as read:
        assert provider.get_realized_vol(payload, as_of=now if bounded else None) is None
    query = str(read.call_args.args[0])
    assert payload not in query
    assert read.call_args.kwargs["params"]["pair"] == payload
    assert ("ts <= :cutoff" in query) == bounded
    execute = engine.connect.return_value.__enter__.return_value.execute
    execute.return_value = []
    provider.get_intraday_values_batch([payload], now, 5 if bounded else None)
    query, params = execute.call_args.args
    assert payload not in str(query)
    assert params["sids"] == [payload]
    assert ("ts >= :floor" in str(query)) == bounded


@pytest.mark.parametrize("overwrite", [False, True])
def test_exit_query_updates_only_bound_idea(overwrite: bool) -> None:
    from src.execution.alpaca_equity_exit import _mark_exit

    engine = create_engine("sqlite:///:memory:")
    payload = "x' OR 1=1 --"
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE alpaca_equity_orders (idea_id TEXT, exit_status TEXT, "
                          "exit_reason TEXT, exit_order_id TEXT, exit_price FLOAT, "
                          "pnl_pct FLOAT, exited_at TEXT)"))
        conn.execute(text("INSERT INTO alpaca_equity_orders (idea_id) VALUES (:id)"),
                     [{"id": payload}, {"id": "untouched"}])
    _mark_exit(engine, payload, datetime(2026, 9, 8, tzinfo=UTC), exit_status="submitted",
               exit_reason=payload, exit_price=2.0, overwrite_fill=overwrite)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT idea_id, exit_status, exit_reason, exit_price "
                                 "FROM alpaca_equity_orders ORDER BY idea_id")).all()
    assert rows == [("untouched", None, None, None), (payload, "submitted", payload, 2.0)]


@pytest.mark.parametrize("bounded", [False, True])
def test_attribution_filters_hostile_strategy_as_literal(bounded: bool) -> None:
    from src.portfolio.attribution import PnLAttributor

    attributor = PnLAttributor(create_engine("sqlite:///:memory:"))
    now = datetime(2026, 9, 8, tzinfo=UTC)
    payload = "x' OR 1=1 --"
    attributor.attribute_fill("EURUSD", 1, 1, ts=now, strategy_id="other", fill_id="1")
    result = attributor.compute_strategy_pnl(payload, since=now if bounded else None)
    assert result.open_quantity == 0
    assert result.realized == 0
