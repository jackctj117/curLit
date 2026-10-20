"""Single provider/local review contract for subscription research (CL-5hj5).

Schema conformance validates shape only. Exact-source matching and semantic
review eligibility remain the responsibility of AdversarialCritic.apply().
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

REVIEW_PROMPT_VERSION = "niche-review-v2"


class _StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ReviewCitation(_StrictRecord):
    source_id: str = Field(min_length=1)
    passage: str = Field(min_length=1)


class ReviewVerdict(_StrictRecord):
    ticker: str = Field(min_length=1)
    verdict: Literal["supported", "contradicted", "insufficient_evidence"]
    strongest_attack: str = Field(min_length=1)
    # Confidence is a probability, never a coerced string, bool or NaN.
    adjusted_confidence: float | None = Field(ge=0, le=1)
    citations: list[ReviewCitation]


class ReviewBatch(_StrictRecord):
    verdicts: list[ReviewVerdict]
