"""Invocation-local passage references shared by native and shadow research.

Only code resolves a reference to an exact captured passage. This proves
provenance, NOT semantic support; freshness, company, liquidity and critic gates
still apply. Opt-in until the captured-input quality experiment is approved.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.events.research_evidence import SourceDocument

PASSAGE_VERSION = "niche-passages-v1"
# Reading-size chunks within already bounded excerpts, not a trading parameter.
PASSAGE_CHARS = 800


class PassageClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["documented_fact", "inference"]
    role: Literal["relationship", "exposure", "catalyst", "disconfirming"]
    statement: str = Field(min_length=1, pattern=r"\S")
    passage_ref: str = Field(pattern=r"^S[1-9]\d*:P[1-9]\d*$")


class PassageIdea(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    ticker: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    action: Literal["long", "short", "buy_calls", "buy_puts"]
    direction: Literal["bullish", "bearish"]
    hop_count: int = Field(ge=0)
    torque_reason: str
    rationale: str
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    # Preserve the existing six-idea / 24-claim resource bounds.
    claims: list[PassageClaim] = Field(max_length=24)


class PassageEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["niche-passages-v1"]
    niche_ideas: list[PassageIdea] = Field(max_length=6)


def passage_instructions() -> str:
    """Generate the provider contract from the same schema used for validation."""
    return (
        "\nFor this invocation use the following versioned output contract instead of "
        "copying source hashes or quotations. Each claim supplies ONLY kind, role, "
        "statement and passage_ref, such as S1:P2, from a tool-provided passage. "
        "Do not invent a reference. Code supplies exact quotations and provenance. "
        "A reference is not proof that the passage supports your claim. Keep fact "
        "separate from inference. Empty niche_ideas is valid abstention. Schema: "
        + json.dumps(PassageEnvelope.model_json_schema(), sort_keys=True)
    )


class PassageRegistry:
    """New registry per invocation; only documents actually retrieved are added."""

    def __init__(self) -> None:
        self._documents: dict[str, SourceDocument] = {}
        self._references: dict[str, tuple[SourceDocument, str]] = {}

    def render(self, sources: list[SourceDocument]) -> list[dict[str, Any]]:
        rendered = []
        for doc in sources:
            if doc.source_id not in self._documents:
                self._documents[doc.source_id] = doc
                label = f"S{len(self._documents)}"
                for start in range(0, len(doc.text), PASSAGE_CHARS):
                    ref = f"{label}:P{start // PASSAGE_CHARS + 1}"
                    self._references[ref] = (doc, doc.text[start : start + PASSAGE_CHARS])
            item = doc.to_dict()
            del item["text"]  # Avoid duplicating the same source in provider context.
            item["passages"] = [
                {"passage_ref": ref, "text": text}
                for ref, (source, text) in self._references.items()
                if source.source_id == doc.source_id
            ]
            rendered.append(item)
        return rendered

    def tool_result(self, raw: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(raw)
        sources = [SourceDocument.from_dict(d) for d in raw.get("sources", [])]
        if sources:
            result["sources"] = self.render(sources)
        return result

    def normalize(self, payload: dict[str, Any]) -> str:
        envelope = PassageEnvelope.model_validate(payload)
        result = envelope.model_dump()
        for idea in result["niche_ideas"]:
            expected = "bullish" if idea["action"] in {"long", "buy_calls"} else "bearish"
            if idea["direction"] != expected:
                raise ValueError("action_direction_mismatch")
            for claim in idea["claims"]:
                ref = claim["passage_ref"]
                if ref not in self._references:
                    raise ValueError("unknown_passage_reference")
                doc, passage = self._references[ref]
                claim["source_id"] = doc.source_id
                claim["passage"] = passage
        return json.dumps(result, sort_keys=True, allow_nan=False)
