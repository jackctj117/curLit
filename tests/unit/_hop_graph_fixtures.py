"""Shared fakes for the hop-graph tests (CL-ynuh). No providers, network or DB.

The fixture world is hand-written so the expected outcome is known
independently of the code under test:

* seed: the Strait of Hormuz (route);
* hop 1: Frontline (FRO) depends on that route - stated in FRO's 10-K;
* hop 2: Acme Marine Coatings (ACME) supplies FRO - stated ONLY in ACME's
  own 10-K (the far node's customer-concentration disclosure). FRO's filing
  never mentions Acme, so retrieving only the near/pointer filing fails;
* ACME's 10-K states its exposure, its 8-K states the dated catalyst.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from src.events.research_evidence import SourceDocument

NOW = datetime(2026, 9, 8, 18, tzinfo=UTC)
ROUTE_ID = "route:strait of hormuz"

FRO_TEXT = (
    "Frontline Ltd. operates a fleet of crude oil tankers. A majority of our voyages "
    "transit the Strait of Hormuz to load crude in the Arabian Gulf. Our vessels are "
    "employed on the spot market."
)
ACME_10K_TEXT = (
    "Acme Marine Coatings Inc. sells hull coatings to tanker operators. Frontline Ltd. "
    "accounted for 38% of our revenue in 2025. Our coatings are applied during scheduled "
    "drydock periods."
)
ACME_8K_TEXT = (
    "On August 3, 2026 Acme signed a three-year hull coating agreement with a tanker "
    "operator. Deliveries under the agreement begin in October 2026."
)
REL_PASSAGE = "Frontline Ltd. accounted for 38% of our revenue in 2025."
FRO_PASSAGE = (
    "A majority of our voyages transit the Strait of Hormuz to load crude in the Arabian Gulf."
)
CATALYST_PASSAGE = "Deliveries under the agreement begin in October 2026."


def doc(
    symbol: str, text: str, form: str = "10-K", published: str = "2026-08-01"
) -> SourceDocument:
    return SourceDocument(
        symbol,
        f"https://www.sec.gov/Archives/edgar/data/1/{symbol.lower()}-{form.lower()}.htm",
        f"{published}T23:59:59+00:00",
        "2026-09-08T12:00:00+00:00",
        text,
        f"{form} 0000000001-26-000001; normalized-text chars 0:{len(text)}",
    )


FRO_DOC = doc("FRO", FRO_TEXT)
ACME_10K = doc("ACME", ACME_10K_TEXT)
ACME_8K = doc("ACME", ACME_8K_TEXT, form="8-K", published="2026-08-04")
DOCS: dict[str, list[SourceDocument]] = {"FRO": [FRO_DOC], "ACME": [ACME_8K, ACME_10K]}

HOP1_CLAIM = "Frontline's crude tankers depend on transiting the Strait of Hormuz"
HOP2_CLAIM = "Acme Marine Coatings supplies hull coatings to Frontline"
EXPOSURE_CLAIM = "Frontline is 38% of Acme revenue"
CATALYST_CLAIM = "Acme deliveries under a new agreement begin October 2026"


def world_edges(pointer: str = "FRO") -> dict[str, list[dict[str, Any]]]:
    """Traversal replies keyed by frontier node id (order-independent)."""
    return {
        ROUTE_ID: [
            {
                "src": ROUTE_ID,
                "dst": {"kind": "company", "name": "Frontline Ltd", "ticker": "FRO"},
                "relation": "depends_on_route",
                "claim": HOP1_CLAIM,
                "where_to_look": {
                    "ticker": "FRO",
                    "section": "business",
                    "keywords": ["voyages", "transit"],
                },
                "direction": "bullish",
            }
        ],
        "company:FRO": [
            {
                "src": "company:FRO",
                "dst": {"kind": "company", "name": "Acme Marine Coatings Inc", "ticker": "ACME"},
                "relation": "supplies",
                "claim": HOP2_CLAIM,
                # The model points at the NEAR node's filing; the evidence is
                # actually in the far node's (ACME's) customer disclosure.
                "where_to_look": {
                    "ticker": pointer,
                    "section": "suppliers",
                    "keywords": ["coatings", "revenue"],
                },
                "direction": "bullish",
                "torque_reason": "single-asset coatings supplier",
                "exposure": {
                    "claim": EXPOSURE_CLAIM,
                    "where_to_look": {
                        "ticker": "ACME",
                        "section": "customers",
                        "keywords": ["38%", "revenue"],
                    },
                },
                "catalyst": {
                    "claim": CATALYST_CLAIM,
                    "where_to_look": {
                        "ticker": "ACME",
                        "section": "8-K",
                        "keywords": ["deliveries", "October 2026"],
                    },
                },
            }
        ],
    }


def world_event() -> dict[str, Any]:
    return {
        "id": 7,
        "headline": "Tanker attacked near the Strait of Hormuz",
        "theme": "hormuz",
        "hop_seeds": [{"kind": "route", "name": "Strait of Hormuz"}],
    }


MARKET = {
    "ACME": {
        "avg_dollar_volume": 5_000_000,
        "market_cap": 200_000_000,
        "observed_at": "2026-09-08T00:00:00+00:00",
        "retrieved_at": "2026-09-08T12:00:00+00:00",
    },
    "FRO": {
        "avg_dollar_volume": 90_000_000,
        "market_cap": 5_000_000_000,
        "observed_at": "2026-09-08T00:00:00+00:00",
        "retrieved_at": "2026-09-08T12:00:00+00:00",
    },
}


def reply(text: str, model: str = "fake-model") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        model=model,
        provider="fake",
        input_tokens=10,
        output_tokens=5,
        usd_cost=None,
        cost_provenance="subscription_unmetered",
    )


class FakeTraversal:
    """Text-only traversal double: edges for exactly the frontier it is shown."""

    def __init__(self, edges_by_src: Mapping[str, list[dict[str, Any]]]) -> None:
        self.edges_by_src = edges_by_src
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def complete(self, *, messages: Any, model: str, max_tokens: int, no_tools: bool) -> Any:
        payload = json.loads(messages[1].content)
        with self._lock:
            self.calls.append({"payload": payload, "no_tools": no_tools, "model": model})
        out = [e for n in payload["frontier"] for e in self.edges_by_src.get(n["id"], [])]
        return reply(json.dumps({"edges": out}), model)


class FakeEntailment:
    """Answers per claim; a passage that does not contain ``needle`` is unclear."""

    def __init__(self, answers: Mapping[str, tuple[str, str]] | None = None) -> None:
        # claim -> (answer, substring the passage must contain for that answer)
        self.answers = dict(answers or {})
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def complete(self, *, messages: Any, model: str, max_tokens: int, no_tools: bool) -> Any:
        user = messages[1].content
        claim = re.search(r"CLAIM: (.*)\n", user)
        passage = re.search(r"<<<(.*)>>>", user, re.S)
        assert claim and passage
        answer, needle = self.answers.get(claim.group(1), ("unclear", ""))
        if needle not in passage.group(1):
            answer = "unclear"
        with self._lock:
            self.calls.append(
                {"claim": claim.group(1), "passage": passage.group(1), "no_tools": no_tools}
            )
        return reply(json.dumps({"answer": answer}), model)


def world_entailment() -> FakeEntailment:
    return FakeEntailment(
        {
            HOP1_CLAIM: ("yes", "Strait of Hormuz"),
            HOP2_CLAIM: ("yes", "38% of our revenue"),
            EXPOSURE_CLAIM: ("yes", "38% of our revenue"),
            CATALYST_CLAIM: ("yes", "October 2026"),
        }
    )


class Retriever:
    """Filing retriever over a fixed document map; records each call."""

    def __init__(self, docs: Mapping[str, list[SourceDocument]] | None = None) -> None:
        self.docs = dict(DOCS if docs is None else docs)
        self.calls: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def __call__(self, ticker: str, query: str) -> list[SourceDocument]:
        with self._lock:
            self.calls.append((ticker, query))
        return list(self.docs.get(ticker, []))


class SupportiveCritic:
    """Critic double: 'supported', citing every documented passage it is shown."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, **kwargs: Any) -> Any:
        self.calls += 1
        user = kwargs["messages"][1].content
        verdicts = []
        for line in user.splitlines():
            if line.startswith("- "):
                ticker = line[2:].split(" ", 1)[0]
            if line.startswith("EVIDENCE AND OBSERVATIONS: "):
                record = json.loads(line.split(": ", 1)[1])
                verdicts.append(
                    {
                        "ticker": ticker,
                        "verdict": "supported",
                        "strongest_attack": "customer concentration",
                        "citations": [
                            {"source_id": c["source_id"], "passage": c["passage"]}
                            for c in record["claims"]
                        ],
                        "adjusted_confidence": 0.5,
                    }
                )
        return reply(json.dumps({"verdicts": verdicts}))


def capture_payload() -> dict[str, Any]:
    return {
        "captured_at": NOW.isoformat(),
        "event": world_event(),
        "symbols": {
            "FRO": {"security_name": "Frontline Ltd", "exchange": "NYSE"},
            "ACME": {"security_name": "Acme Marine Coatings Inc", "exchange": "NASDAQ"},
        },
        "sources": [d.to_dict() for d in (FRO_DOC, ACME_10K, ACME_8K)],
        "market_data": json.loads(json.dumps(MARKET)),
    }
