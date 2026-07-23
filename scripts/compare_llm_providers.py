"""LLM provider A/B comparison harness (CL-gyjz).

Runs the same prompts through Claude (api.anthropic.com) and DeepSeek
(api.deepseek.com) and reports cost, latency, and a deterministic
quality score per response. Designed to answer the operator question
"is DeepSeek good enough on our research-loop debate to justify the
17× cost reduction?"

The quality rubric is opinionated and intentionally simple — we score
on three dimensions a human research-loop reviewer would care about:

  1. Structure validity   — did the response follow the expected
                            format (e.g. valid JSON when JSON-mode,
                            section headers when free-form)?
  2. Information density  — character count outside boilerplate
                            (filler-word stripping, then length).
  3. FX-relevance keyword presence — fraction of expected domain
                            terms that appear (passed in by caller).

Scores are 0..1 per dimension; weighted sum is the headline score. The
weights are exposed so the operator can re-run with their own bias
(e.g. weight density higher on debate prompts, structure higher on
JSON-extract prompts).

The harness does NOT auto-promote DeepSeek. It produces a report; the
human decides. That's the right cut for "use a cheaper model on
strategy generation" — a wrong call here propagates into all of CL-h986.

Usage:
  .venv/bin/python -m scripts.compare_llm_providers \\
      --prompt-file scripts/_seed_prompts/idea_extract.json \\
      --providers claude,deepseek \\
      [--out reports/llm_compare.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.dotenv_bootstrap import load_project_env
from src.research.llm.client import LLMResponse, Message, get_client

logger = logging.getLogger(__name__)


# Default scoring weights. Sum to 1.0; tunable per --weights flag.
_WEIGHTS_STRUCTURE: float = 0.4
_WEIGHTS_DENSITY: float = 0.3
_WEIGHTS_KEYWORDS: float = 0.3

# Filler-word set for density scoring. Strip these before counting
# characters so a verbose-but-empty response doesn't outscore a
# concise informative one.
_FILLER_WORDS: tuple[str, ...] = (
    "however",
    "moreover",
    "furthermore",
    "additionally",
    "in conclusion",
    "in summary",
    "it is important to note",
    "as mentioned",
    "indeed",
    "essentially",
    "basically",
    "ultimately",
    "interestingly",
)


# Default per-provider model selections — operator overrides via flag.
# We pick the parity model on each side: Claude Opus 4.7 vs
# DeepSeek's reasoner (closest reasoning-tier comparable).
_DEFAULT_MODELS: dict[str, str] = {
    "claude": "claude-opus-4-7",
    "deepseek": "deepseek-reasoner",
}


@dataclass
class PromptSpec:
    """One prompt to compare. ``expected_keywords`` are the FX-relevance
    terms we'd want to see in any good response."""

    name: str
    system: str
    user: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_format: str = "free-form"  # "json" | "free-form"


@dataclass
class QualityScore:
    structure: float
    density: float
    keyword_presence: float
    weighted: float


@dataclass
class ComparisonRow:
    prompt_name: str
    provider: str
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    usd_cost: float
    elapsed_sec: float
    quality: QualityScore


def _score_structure(text: str, expected_format: str) -> float:
    if expected_format == "json":
        # Try to parse the response as JSON. If it parses, full credit.
        # If it parses after stripping a code-fence, half credit
        # (technically valid but wraps require post-processing).
        try:
            json.loads(text)
            return 1.0
        except json.JSONDecodeError:
            stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            try:
                json.loads(stripped)
                return 0.5
            except json.JSONDecodeError:
                return 0.0
    # Free-form: any response with at least 50 chars and 1+ paragraph
    # break gets full credit. Empty / single-line "yes" answers fail.
    if len(text) < 50:
        return 0.0
    return 1.0 if "\n" in text else 0.5


def _score_density(text: str) -> float:
    """Information density = stripped-char-count / raw-char-count.

    Caps at 1.0. Strips filler phrases case-insensitively. Empty text
    scores 0.
    """
    if not text:
        return 0.0
    stripped = text
    for filler in _FILLER_WORDS:
        stripped = re.sub(filler, " ", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if not text.strip():
        return 0.0
    return min(1.0, len(stripped) / len(text.strip()))


def _score_keywords(text: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    low = text.lower()
    hits = sum(1 for kw in expected if kw.lower() in low)
    return hits / len(expected)


def score_response(
    text: str,
    prompt: PromptSpec,
    weights: tuple[float, float, float] = (
        _WEIGHTS_STRUCTURE,
        _WEIGHTS_DENSITY,
        _WEIGHTS_KEYWORDS,
    ),
) -> QualityScore:
    s = _score_structure(text, prompt.expected_format)
    d = _score_density(text)
    k = _score_keywords(text, prompt.expected_keywords)
    w_s, w_d, w_k = weights
    weighted = w_s * s + w_d * d + w_k * k
    return QualityScore(
        structure=s,
        density=d,
        keyword_presence=k,
        weighted=weighted,
    )


def run_one(
    provider: str,
    model: str,
    prompt: PromptSpec,
    max_tokens: int = 2048,
) -> ComparisonRow:
    client = get_client(provider)
    resp: LLMResponse = client.complete(
        messages=[
            Message(role="system", content=prompt.system),
            Message(role="user", content=prompt.user),
        ],
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    quality = score_response(resp.text, prompt)
    return ComparisonRow(
        prompt_name=prompt.name,
        provider=provider,
        model=model,
        text=resp.text,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        usd_cost=resp.usd_cost,
        elapsed_sec=resp.elapsed_sec,
        quality=quality,
    )


def load_prompts(path: Path) -> list[PromptSpec]:
    """Read a JSON file containing a list of prompt specs. Schema is
    the PromptSpec dataclass."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raw = [raw]
    return [PromptSpec(**item) for item in raw]


def summarize(rows: list[ComparisonRow]) -> dict[str, Any]:
    """Aggregate by provider: avg quality, total cost, avg latency,
    cost per quality point. Last metric is the headline answer to
    "which provider gives more value per dollar"."""
    by_provider: dict[str, list[ComparisonRow]] = {}
    for r in rows:
        by_provider.setdefault(r.provider, []).append(r)

    summary: dict[str, Any] = {}
    for prov, items in by_provider.items():
        n = len(items)
        if n == 0:
            continue
        total_cost = sum(r.usd_cost for r in items)
        avg_quality = sum(r.quality.weighted for r in items) / n
        avg_latency = sum(r.elapsed_sec for r in items) / n
        # USD per "quality point" — lower is better. Floor quality at
        # 0.01 to avoid division-by-zero on a totally broken response.
        cost_per_quality = total_cost / max(0.01, avg_quality * n)
        summary[prov] = {
            "n_prompts": n,
            "total_usd": round(total_cost, 6),
            "avg_quality": round(avg_quality, 3),
            "avg_latency_sec": round(avg_latency, 2),
            "usd_per_quality": round(cost_per_quality, 6),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    load_project_env()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prompt-file",
        type=Path,
        required=True,
        help="JSON file with PromptSpec[] (see scripts/_seed_prompts/)",
    )
    p.add_argument(
        "--providers",
        default="claude,deepseek",
        help="Comma-separated list of providers to test",
    )
    p.add_argument(
        "--models",
        default="",
        help=(
            "Optional comma-separated provider:model overrides "
            "(e.g. 'claude:claude-sonnet-4-6,deepseek:deepseek-chat')"
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("reports/llm_compare.json"),
        help="Output JSON path",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    prompts = load_prompts(args.prompt_file)
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    model_overrides: dict[str, str] = {}
    if args.models:
        for piece in args.models.split(","):
            if ":" in piece:
                k, v = piece.split(":", 1)
                model_overrides[k.strip()] = v.strip()

    rows: list[ComparisonRow] = []
    for prompt in prompts:
        for provider in providers:
            model = model_overrides.get(provider) or _DEFAULT_MODELS.get(
                provider,
                "",
            )
            if not model:
                logger.error(
                    "No default model for %s and no override given",
                    provider,
                )
                return 2
            try:
                row = run_one(provider, model, prompt)
            except Exception as exc:
                logger.error(
                    "%s/%s failed on prompt %s: %s: %s",
                    provider,
                    model,
                    prompt.name,
                    type(exc).__name__,
                    exc,
                )
                continue
            rows.append(row)
            logger.info(
                "%s/%s [%s]: q=%.3f cost=$%.4f t=%.1fs",
                provider,
                model,
                prompt.name,
                row.quality.weighted,
                row.usd_cost,
                row.elapsed_sec,
            )

    summary = summarize(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": [asdict(r) for r in rows],
        "summary": summary,
    }
    args.out.write_text(json.dumps(payload, indent=2))

    print("\n=== Summary ===")
    for provider, stats in summary.items():
        print(f"\n{provider}:")
        for k, v in stats.items():
            print(f"  {k:20s} {v}")

    if "claude" in summary and "deepseek" in summary:
        cost_ratio = summary["claude"]["total_usd"] / max(
            1e-6,
            summary["deepseek"]["total_usd"],
        )
        quality_ratio = summary["claude"]["avg_quality"] / max(
            0.01,
            summary["deepseek"]["avg_quality"],
        )
        print(
            f"\nClaude / DeepSeek ratios:  cost={cost_ratio:.1f}×  quality={quality_ratio:.2f}×",
        )

    print(f"\nFull report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
