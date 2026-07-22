"""Strategy lifecycle management CLI (CL-6vv).

Operator-facing entry point for the paper → live → retire flow. The
business logic lives in PortfolioCoordinator (add/promote/remove) and
AllocationPolicy (CL-15r4 governance); this script is just the wiring.

Usage:
  # Add a new strategy in paper mode at 0% allocation.
  .venv/bin/python -m scripts.manage_strategies add \\
      --strategy rate_diff_mr --paper-days 90

  # Evaluate whether a paper strategy is ready for promotion.
  .venv/bin/python -m scripts.manage_strategies evaluate \\
      --strategy rate_diff_mr

  # Promote to live with explicit weight (overrides policy default).
  .venv/bin/python -m scripts.manage_strategies promote \\
      --strategy rate_diff_mr --weight 0.05

  # Remove (graceful liquidation + rebalance).
  .venv/bin/python -m scripts.manage_strategies remove \\
      --strategy rate_diff_mr

The script connects to the live engine state via the same coordinator
configuration as run_engine. Operations that mutate state (promote,
remove) are loud — they print every weight change before applying.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from src.dotenv_bootstrap import load_project_env
from src.portfolio.allocation_policy import (
    AllocationPolicy,
    PolicyConfig,
    PromotionVerdict,
)

logger = logging.getLogger(__name__)


def _load_paper_returns(
    coordinator: Any, strategy_id: str, lookback_days: int = 365,
) -> pd.Series:
    """Pull recent paper-mode daily returns for ``strategy_id``.

    Returns an empty Series if the coordinator's state store doesn't
    have a returns history method — the policy layer treats empty as
    HOLD ("insufficient data"), which is the right default.
    """
    state = getattr(coordinator, "state", None)
    if state is None or not hasattr(state, "load_strategy_returns_history"):
        return pd.Series(dtype=float)
    df = state.load_strategy_returns_history([strategy_id], lookback_days)
    if df is None or df.empty or strategy_id not in df.columns:
        return pd.Series(dtype=float)
    return df[strategy_id].dropna()


def _build_coordinator(broker_mode: str = "oanda-practice") -> Any:
    """Wire a coordinator using run_engine's real builders (CL-9dhg).

    Production has one running coordinator; this script must reach the
    same Postgres state and the same broker, so we reuse run_engine's
    build_broker / build_strategies / build_coordinator rather than a
    private admin hook (the previously imported
    ``_build_coordinator_for_admin`` never existed — every mutating
    subcommand died on ImportError). Heavy lift: import-on-demand to
    keep --help fast.
    """
    from src.execution.oms import OrderManager
    from src.execution.rejection import RejectionHandler
    from src.runtime.run_engine import (
        CONFIG_PATH,
        build_blackout_evaluator,
        build_broker,
        build_coordinator,
        build_strategies,
        build_trade_journal,
        load_config,
    )

    config = load_config(CONFIG_PATH)
    broker = build_broker(broker_mode)
    journal = build_trade_journal()
    oms = OrderManager(
        broker,
        rejection_handler=RejectionHandler(journal=journal),
        journal=journal,
    )
    strategies = build_strategies(config, broker, oms)
    coordinator = build_coordinator(
        config, strategies, oms, broker,
        blackout_evaluator=build_blackout_evaluator(config),
    )
    if coordinator is None:
        # build_coordinator degrades to None when the portfolio state
        # store (Postgres) is unreachable — acceptable for the engine
        # (legacy direct-OMS mode), NOT for an admin CLI that mutates
        # allocations and liquidates positions. Fail loud.
        raise RuntimeError(
            "Could not construct PortfolioCoordinator (is Postgres "
            "reachable?) — refusing to run admin commands without "
            "portfolio state.",
        )
    return coordinator


def _seed_coordinator_memory(coord: Any) -> None:
    """Seed cross-tick target memory before any position-mutating op.

    A fresh CLI process has EMPTY ``_strategy_targets`` — the engine only
    seeds it on the first ``process_intents`` tick, which never runs
    here. ``remove_strategy`` computes each co-holder's remaining share
    from that memory, so without seeding it submits target=0
    liquidations that flatten EVERY strategy's position on shared
    symbols while printing success.

    The coordinator module is orchestrator-owned, so we reach the
    private seeder from here; suggested follow-up: a public
    ``seed_targets_from_books()`` wrapper on PortfolioCoordinator
    delegating to ``_seed_targets_from_books``.
    """
    coord.seed_targets_from_books()


def cmd_add(args: argparse.Namespace) -> int:
    coord = _build_coordinator(args.broker)
    strategy = coord.strategies.get(args.strategy)
    if strategy is None:
        # In production, strategies are constructed at engine startup —
        # the script can't add an arbitrary new strategy class. Print
        # the registered set and exit.
        print(
            f"Strategy '{args.strategy}' not in registered strategies. "
            f"Available: {sorted(coord.strategies.keys())}",
            file=sys.stderr,
        )
        return 2
    coord.add_strategy(strategy, initial_paper_days=args.paper_days)
    print(
        f"Added {args.strategy} in paper mode for {args.paper_days} days. "
        "Watch attribution dashboard for paper performance.",
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    coord = _build_coordinator(args.broker)
    if args.strategy not in coord.allocations:
        print(f"Strategy '{args.strategy}' not registered", file=sys.stderr)
        return 2
    alloc = coord.allocations[args.strategy]

    # The coordinator doesn't track paper_start; infer from the state
    # store's first observation date.
    paper_returns = _load_paper_returns(coord, args.strategy)
    if paper_returns.empty:
        paper_start = datetime.now(UTC) - timedelta(days=0)
    else:
        first_ts = paper_returns.index[0]
        paper_start = (
            first_ts.to_pydatetime() if hasattr(first_ts, "to_pydatetime")
            else datetime.combine(first_ts, datetime.min.time(), UTC)
        )

    policy = AllocationPolicy(PolicyConfig())
    decision = policy.evaluate(args.strategy, paper_start, paper_returns)

    print(f"Strategy:    {args.strategy}")
    print(f"Paper mode:  {alloc.paper_mode}")
    print(f"Verdict:     {decision.verdict.value}")
    print(f"Reason:      {decision.reason}")
    print(f"Paper days:  {decision.paper_days}")
    if decision.paper_sharpe is not None:
        print(f"Paper Sharpe: {decision.paper_sharpe:.3f}")
    if decision.verdict is PromotionVerdict.PROMOTE:
        print(f"Suggested initial weight: {decision.initial_weight:.1%}")
        print(
            f"\nTo promote: scripts/manage_strategies.py promote "
            f"--strategy {args.strategy} --weight {decision.initial_weight}",
        )
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    coord = _build_coordinator(args.broker)
    coord.promote_strategy_to_live(
        args.strategy, initial_weight=args.weight,
    )
    print(f"Promoted {args.strategy} to live at {args.weight:.1%}.")
    print("Verify rebalance applied: bd remember 'check coordinator weights'")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    coord = _build_coordinator(args.broker)
    if args.strategy not in coord.strategies:
        print(f"Strategy '{args.strategy}' not registered", file=sys.stderr)
        return 2
    if not args.confirm:
        print(
            "Refusing to remove without --confirm. This liquidates all "
            f"attributed positions for {args.strategy} immediately.",
            file=sys.stderr,
        )
        return 2
    # MUST run before remove_strategy — see _seed_coordinator_memory
    # (CL-9dhg: unseeded memory => zero-target liquidations flatten
    # every co-holder on shared symbols).
    _seed_coordinator_memory(coord)
    coord.remove_strategy(args.strategy)
    print(f"Removed {args.strategy}; positions flattened, rebalance scheduled.")
    return 0


def _add_broker_arg(sub_parser: argparse.ArgumentParser) -> None:
    """Every subcommand builds the full stack, so every subcommand needs
    the broker mode. Default matches the production fleet
    (oanda-practice — see docs/CURRENT_OPERATIONS.md); choices are
    validated by run_engine.build_broker, not duplicated here."""
    sub_parser.add_argument(
        "--broker", default="oanda-practice",
        help=(
            "Broker mode (must match the running engine's mode; same "
            "values as run_engine --broker). Default: oanda-practice."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    load_project_env()

    parser = argparse.ArgumentParser(
        description="Manage strategy lifecycle (add → promote → remove).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="Add strategy in paper mode")
    p_add.add_argument("--strategy", required=True)
    _add_broker_arg(p_add)
    p_add.add_argument(
        "--paper-days", type=int, default=90,
        help="Required paper period before evaluate() will promote",
    )
    p_add.set_defaults(func=cmd_add)

    p_eval = sub.add_parser(
        "evaluate", help="Show promotion verdict from AllocationPolicy",
    )
    p_eval.add_argument("--strategy", required=True)
    _add_broker_arg(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    p_prom = sub.add_parser("promote", help="Promote paper → live")
    p_prom.add_argument("--strategy", required=True)
    p_prom.add_argument(
        "--weight", type=float, default=0.05,
        # %% — argparse help strings interpolate %-specifiers; a bare %
        # is a "badly formed help string" ValueError on Python >= 3.14.
        help="Initial allocation weight (default 5%%)",
    )
    _add_broker_arg(p_prom)
    p_prom.set_defaults(func=cmd_promote)

    p_rem = sub.add_parser("remove", help="Liquidate and remove")
    p_rem.add_argument("--strategy", required=True)
    p_rem.add_argument(
        "--confirm", action="store_true",
        help="Required: confirms immediate liquidation of all positions",
    )
    _add_broker_arg(p_rem)
    p_rem.set_defaults(func=cmd_remove)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
