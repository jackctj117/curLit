"""Promote operator-flagged papers to the IdeaGenerator (CL-28j follow-up).

Closes the loop the dashboard's "For impl" button started:

  Operator clicks "For impl" → research_papers.read_status = 'for_implementation'
       ↓
  THIS SCRIPT (cron / Airflow nightly):
    - SELECT paper_id, title FROM research_papers
        WHERE read_status = 'for_implementation'
        ORDER BY implementation_priority ASC, relevance_score DESC
    - For each: locate matching extract at data/research/extracts/{hash}.md
        - Missing extract → log warning, skip (operator promoted before
          the LLM extract step ran; next nightly will pick it up)
        - Existing extract → call IdeaGenerator.ideate()
            - PROPOSED  → write docs/research/hypotheses/{slug}.md
                          → transition row to read_status='in_pipeline'
            - DECLINED  → transition row to read_status='idea_declined'
                          (so we don't re-run on every cron tick)
       ↓
  Bull/Bear debate (next research_loop run) reads the new hypothesis
  brief from disk and runs the promotion review.

Idempotent: each tick, only papers still in `for_implementation` are
processed. Once transitioned to `in_pipeline` or `idea_declined`,
they're invisible to subsequent runs.

Cost: 1 LLM call per promoted paper. Operator-gated so the cost is
proportional to actual triage volume, not feed firehose volume.

Usage:
  .venv/bin/python -m scripts.promote_papers_to_idea_agent
                                              [--max-papers 10]
                                              [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, cast

from sqlalchemy import text

logger = logging.getLogger(__name__)


# 10 promotions per nightly run by default. The IdeaGenerator runs on
# DeepSeek (~$0.01/call) so the cost ceiling is fine, but a runaway
# operator-pasted-everything-as-for-impl scenario shouldn't blow up
# either compute or pipeline state in a single tick. Operator can
# override with --max-papers; daily backlogs work themselves down on
# subsequent nights.
_DEFAULT_MAX_PAPERS_PER_RUN: int = 10


def _select_promotable_papers(engine: Any, limit: int) -> list[dict[str, Any]]:
    """Pull papers awaiting promotion. Sort by priority ASC (1=top of
    queue), then relevance DESC, so the operator's manual ordering
    drives the order.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT paper_id, title, source, relevance_score,
                       implementation_priority
                FROM research_papers
                WHERE read_status = 'for_implementation'
                ORDER BY
                    CASE WHEN implementation_priority = 0 THEN 999
                         ELSE implementation_priority END ASC,
                    relevance_score DESC
                LIMIT :n
            """),
            {"n": limit},
        ).fetchall()
    return [
        {
            "paper_id": r[0], "title": r[1], "source": r[2],
            "relevance_score": float(r[3] or 0),
            "implementation_priority": int(r[4] or 0),
        }
        for r in rows
    ]


def _set_status(engine: Any, paper_id: str, status: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE research_papers
                SET read_status = :s
                WHERE paper_id = :pid
            """),
            {"s": status, "pid": paper_id},
        )


def _find_extract_path(paper_id: str, extract_root: Path) -> Path | None:
    """The ExtractStore writes ``{paper_hash}.md``. paper_id IS the
    paper_hash for rows inserted via the bridge (CL-28j). Return the
    path if present, else None."""
    candidate = extract_root / f"{paper_id}.md"
    return candidate if candidate.exists() else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--max-papers", type=int, default=_DEFAULT_MAX_PAPERS_PER_RUN,
        help="Max papers to promote per run (cost ceiling)",
    )
    p.add_argument(
        "--extract-root", default="data/research/extracts",
        help="Where the ExtractStore writes paper extracts",
    )
    p.add_argument(
        "--hypothesis-dir", default="docs/research/hypotheses",
        help="Where IdeaGenerator writes accepted hypothesis briefs",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help=(
            "List promotable papers without calling the LLM or "
            "transitioning state"
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from src.dotenv_bootstrap import load_project_env
    load_project_env()

    from src.runtime.run_engine import _build_db_engine
    engine = _build_db_engine()
    extract_root = Path(args.extract_root)

    promotable = _select_promotable_papers(engine, args.max_papers)
    if not promotable:
        logger.info("No papers in for_implementation — nothing to promote")
        return 0

    logger.info(
        "Promoting %d papers (cap=%d): %s",
        len(promotable), args.max_papers,
        ", ".join(p["paper_id"][:8] for p in promotable),
    )

    if args.dry_run:
        for p_row in promotable:
            extract = _find_extract_path(p_row["paper_id"], extract_root)
            present = "YES" if extract else "NO (skip)"
            print(
                f"  prio={p_row['implementation_priority']} "
                f"score={p_row['relevance_score']:.2f} "
                f"extract={present:8s} {p_row['title'][:60]}",
            )
        return 0

    # Build the IdeaGenerator. Lazy import keeps --help cheap.
    from src.research.agents.idea import IdeaGenerator, IdeaStatus
    from src.research.config import load_config

    research_cfg = load_config("configs/research_agents.yaml")
    idea_agent = cast(IdeaGenerator, IdeaGenerator.from_config(
        name="idea_generator", research_config=research_cfg,
    ))

    n_proposed = 0
    n_declined = 0
    n_skipped = 0

    for p_row in promotable:
        paper_id = p_row["paper_id"]
        extract = _find_extract_path(paper_id, extract_root)
        if extract is None:
            logger.warning(
                "[%s] no extract on disk at %s/%s.md — skipping; will retry "
                "next run after the LLM extract step catches up",
                paper_id[:8], extract_root, paper_id,
            )
            n_skipped += 1
            continue

        try:
            result = idea_agent.ideate(
                extract, hypothesis_dir=args.hypothesis_dir,
            )
        except Exception:
            logger.exception(
                "[%s] IdeaGenerator raised — leaving status unchanged "
                "for retry next run", paper_id[:8],
            )
            n_skipped += 1
            continue

        if result.status == IdeaStatus.PROPOSED:
            _set_status(engine, paper_id, "in_pipeline")
            n_proposed += 1
            logger.info(
                "[%s] PROPOSED → %s (status=in_pipeline)",
                paper_id[:8], result.hypothesis_path,
            )
        else:  # DECLINED
            _set_status(engine, paper_id, "idea_declined")
            n_declined += 1
            logger.info(
                "[%s] DECLINED — %s (status=idea_declined)",
                paper_id[:8], result.reason[:120],
            )

    logger.info(
        "Promotion summary: proposed=%d declined=%d skipped=%d",
        n_proposed, n_declined, n_skipped,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
