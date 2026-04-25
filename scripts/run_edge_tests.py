#!/usr/bin/env python
"""Edge test runner CLI — weekly cron entrypoint (CL-4lp follow-up).

Adapts the in-process EdgeRunner orchestrator to a production CLI:

    .venv/bin/python -m scripts.run_edge_tests \
        [--output edge_reports/edge_$(date +%Y%m%d).json] \
        [--policy configs/edge_policy.yaml]

Loads strategies via run_engine.build_strategies, pulls their backtest +
live state from StrategyStateStore, runs G1→G9 via EdgeRunner, writes a
JSON report, and (when actions fire) sends a Pushover alert if
PUSHOVER_API_TOKEN + PUSHOVER_USER_KEY are present in the environment.

Schedule weekly via cron (see reference/14_edge_testing.md "Cron Schedule
Entry"); the runner is idempotent and does not mutate state.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

# Make `src.*` imports work when invoked as a script (the project ships
# `src/` as the package root, not installed under the package name).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.edge_testing.edge_policy import LiveAction, load_edge_policy  # noqa: E402
from src.edge_testing.runner import (  # noqa: E402
    EdgeRunner,
    StrategyEdgeInputs,
)
from src.monitoring.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


# Default report path — operator-friendly date-stamped filename.
def _default_output_path() -> Path:
    return (
        Path(os.environ.get("FX_EDGE_REPORT_DIR", "edge_reports"))
        / f"edge_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    )


# -----------------------------------------------------------------------------
# Strategy input adapter
# -----------------------------------------------------------------------------


def _build_inputs_from_state_store(
    state_store: object | None,
    strategies: list[object],
) -> list[StrategyEdgeInputs]:
    """Turn the (strategy, state_store) pair into the runner's input shape.

    Production state_store is StrategyStateStore. Until D6 (CL-5lq) wires
    historical returns + backtest expectations + per-strategy fill streams,
    every strategy's inputs are mostly None and the runner skips most layers
    with explanatory notes — which is the correct behavior. The runner is
    designed so the report becomes meaningful as the underlying data layers
    fill in.
    """
    out: list[StrategyEdgeInputs] = []
    for strategy in strategies:
        sid = getattr(strategy, "id", None) or "unknown"
        # state_store APIs that don't exist yet (returns_history, fills,
        # backtest_metrics) are accessed via getattr so we degrade gracefully.
        backtest_returns = None
        live_returns = None
        backtest_expectations = None
        if state_store is not None:
            try:
                getter = getattr(state_store, "get_backtest_returns", None)
                if callable(getter):
                    backtest_returns = getter(sid)
            except Exception:
                logger.exception("get_backtest_returns failed for %s", sid)
            try:
                getter = getattr(state_store, "get_live_returns", None)
                if callable(getter):
                    live_returns = getter(sid)
            except Exception:
                logger.exception("get_live_returns failed for %s", sid)
            try:
                getter = getattr(state_store, "get_backtest_expectations", None)
                if callable(getter):
                    backtest_expectations = getter(sid)
            except Exception:
                logger.exception("get_backtest_expectations failed for %s", sid)

        out.append(
            StrategyEdgeInputs(
                strategy_id=sid,
                backtest_returns=backtest_returns,
                live_returns=live_returns,
                backtest_expectations=backtest_expectations,
            ),
        )
    return out


# -----------------------------------------------------------------------------
# Pushover alerting
# -----------------------------------------------------------------------------


def _send_alerts_if_configured(
    actions: Sequence[tuple[str, LiveAction, str]],
) -> None:
    """Send a single Pushover notification listing all action items.

    No-op when PUSHOVER_API_TOKEN or PUSHOVER_USER_KEY is unset — the
    operator running ad-hoc gets stdout output via the logger.
    """
    if not actions:
        return
    token = os.environ.get("PUSHOVER_API_TOKEN")
    user = os.environ.get("PUSHOVER_USER_KEY")
    if not token or not user:
        logger.info(
            "Pushover not configured (PUSHOVER_API_TOKEN / PUSHOVER_USER_KEY missing); "
            "skipping alert dispatch",
        )
        return

    lines = [
        f"[{action.value}] {sid}: {detail}"
        for sid, action, detail in actions
    ]
    message = "Edge run actions:\n" + "\n".join(lines)
    try:
        import httpx
        priority = 1 if any(
            a == LiveAction.HALT_STRATEGY or a == LiveAction.RETIRE_STRATEGY
            for _, a, _ in actions
        ) else 0
        httpx.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": token,
                "user": user,
                "title": "FX Edge Run",
                "message": message,
                "priority": priority,
            },
            timeout=10,
        )
        logger.info("Sent Pushover alert (%d actions)", len(actions))
    except Exception:
        logger.exception("Pushover alert dispatch failed")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run weekly edge tests")
    parser.add_argument("--output", type=Path, default=None,
                        help="report output path (default: edge_reports/edge_<ts>.json)")
    parser.add_argument("--policy", type=Path, default=None,
                        help="path to edge_policy.yaml (default: configs/edge_policy.yaml)")
    args = parser.parse_args()

    log_dir = Path(os.environ.get("FX_LOG_DIR", "logs"))
    setup_logging("edge_runner", log_dir)

    policy = load_edge_policy(args.policy) if args.policy else load_edge_policy()
    runner = EdgeRunner(policy=policy)

    # Lazy-import to avoid pulling Postgres/SQLAlchemy when running --help.
    try:
        import yaml
        from sqlalchemy import create_engine

        from src.runtime.run_engine import (
            CONFIG_PATH,
            build_strategies,
        )
        from src.strategies.state import StrategyStateStore
    except Exception:
        logger.exception("Failed to import production dependencies")
        return

    if not CONFIG_PATH.exists():
        logger.error("Config not found at %s", CONFIG_PATH)
        return
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if not isinstance(config, dict):
        logger.error("Config did not parse to dict")
        return

    db_url = os.environ.get(
        "DATABASE_URL",
        f"postgresql+psycopg2://{os.environ.get('POSTGRES_USER', 'fx')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'changeme')}@"
        f"{os.environ.get('POSTGRES_HOST', 'localhost')}:5432/"
        f"{os.environ.get('POSTGRES_DB', 'fx')}",
    )
    try:
        engine = create_engine(db_url)
        state_store = StrategyStateStore(engine)
    except Exception:
        logger.exception("Failed to construct state_store; runner will skip data layers")
        state_store = None

    # Build strategies via the same path live engine does. This intentionally
    # constructs a PaperBroker (practice mode); we only call .id and .symbols
    # below, never trade.
    from src.execution.oms import OrderManager
    from src.execution.paper_broker import PaperBroker
    broker = PaperBroker()
    oms = OrderManager(broker)
    strategies = build_strategies(config, broker, oms)

    inputs = _build_inputs_from_state_store(state_store, strategies)
    report = runner.run_all(inputs)

    output_path = args.output or _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    logger.info("Wrote edge report to %s (%d strategies)", output_path, len(report.results))

    actions = [
        (sid, r.g9_action, r.g8_verdict.summary if r.g8_verdict else "")
        for sid, r in report.results.items()
        if r.g9_action and r.g9_action != LiveAction.CONTINUE
    ]
    if actions:
        for sid, action, detail in actions:
            logger.warning("Action: %s → %s (%s)", sid, action.value, detail)
        _send_alerts_if_configured(actions)
    else:
        logger.info("All strategies passed; no action items.")


if __name__ == "__main__":
    main()
