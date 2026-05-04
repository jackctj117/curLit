"""Pre-warm the Hugging Face cache with CB stance model weights (CL-v1x9).

Each model is ~400MB (RoBERTa-base) — running the whole CB roster
pre-fetches a few GB. Run once on a fresh server before the engine
starts so first-trade-cycle isn't blocked on a cold-cache download.

Per-model cache lives under ``~/.cache/huggingface/hub/`` by default.
Override with ``HF_HOME`` or ``TRANSFORMERS_CACHE`` env vars.

Usage:
  .venv/bin/python -m scripts.download_finbert_weights
  .venv/bin/python -m scripts.download_finbert_weights --cbs fed,ecb
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.nlp.inference import CB_MODEL_REGISTRY

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cbs", default=None,
        help=(
            "Comma-separated CB short names to download (default: all "
            "in CB_MODEL_REGISTRY). Example: --cbs fed,ecb,boe"
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cbs:
        names = [n.strip().lower() for n in args.cbs.split(",") if n.strip()]
        unknown = [n for n in names if n not in CB_MODEL_REGISTRY]
        if unknown:
            print(
                f"Unknown CB names: {unknown}. "
                f"Known: {sorted(CB_MODEL_REGISTRY)}",
                file=sys.stderr,
            )
            return 2
    else:
        # Dedup model IDs — boe/boj/boc all alias to fed in v1, no
        # point downloading the same weights three times.
        names = sorted(CB_MODEL_REGISTRY.keys())

    seen_ids: set[str] = set()
    targets = []
    for name in names:
        model_id = CB_MODEL_REGISTRY[name]
        if model_id not in seen_ids:
            seen_ids.add(model_id)
            targets.append((name, model_id))

    logger.info("Downloading %d distinct stance models", len(targets))
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    for name, model_id in targets:
        logger.info("[%s] %s", name, model_id)
        AutoTokenizer.from_pretrained(
            model_id, do_lower_case=True, do_basic_tokenize=True,
        )
        AutoModelForSequenceClassification.from_pretrained(model_id)
        logger.info("  cached")

    logger.info("Done. Cache lives under HF_HOME / ~/.cache/huggingface/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
