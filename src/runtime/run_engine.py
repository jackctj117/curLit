"""Live trading engine entrypoint — instantiate and run the full stack."""

import argparse
import asyncio
import logging
import os
import signal
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import create_engine

from src.data.provider import DataProvider
from src.execution.oanda_broker import OandaBroker
from src.execution.oms import OrderManager
from src.execution.paper_broker import PaperBroker
from src.monitoring.logging_setup import setup_logging
from src.nlp.provider import NLPDataProvider
from src.portfolio import (
    PortfolioConstraints,
    PortfolioCoordinator,
    PortfolioStateStore,
    PositionReconciler,
    ReconciliationPolicy,
)
from src.runtime.live_engine import LiveEngine
from src.strategies.cb_sentiment_shift import CBSentimentConfig, CBSentimentShiftStrategy
from src.strategies.rate_diff_mean_reversion import RateDiffMRConfig, RateDiffMRStrategy

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(os.environ.get("FX_CONFIG", "configs/live_portfolio.yaml"))


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.warning("Config not found at %s — using defaults", path)
        return {}
    loaded = yaml.safe_load(path.read_text())
    return loaded if isinstance(loaded, dict) else {}


def build_broker(practice: bool) -> Any:
    if practice:
        return PaperBroker(initial_capital=100_000)
    oanda_key = os.environ.get("OANDA_API_KEY", "")
    oanda_id = os.environ.get("OANDA_ACCOUNT_ID", "")
    if not oanda_key:
        logger.error("OANDA_API_KEY not set — falling back to paper broker")
        return PaperBroker(initial_capital=100_000)
    return OandaBroker(oanda_key, oanda_id, practice=False)


def _build_db_engine() -> Any:
    db_url = os.environ.get(
        "DATABASE_URL",
        f"postgresql+psycopg2://{os.environ.get('POSTGRES_USER', 'fx')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'changeme')}@"
        f"{os.environ.get('POSTGRES_HOST', 'localhost')}:5432/"
        f"{os.environ.get('POSTGRES_DB', 'fx')}",
    )
    return create_engine(db_url)


def build_strategies(
    config: dict[str, Any],
    broker: Any,
    oms: OrderManager,
) -> list[Any]:
    engine = _build_db_engine()
    data_provider = DataProvider(engine)
    nlp_provider = NLPDataProvider(engine)

    strategies: list[Any] = []
    for sconf in config.get("strategies", []):
        sid = sconf.get("id", "")
        scfg = sconf.get("config", {})
        if "rate_diff" in sid:
            strategies.append(
                RateDiffMRStrategy(
                    RateDiffMRConfig(**scfg) if scfg else RateDiffMRConfig(),
                    data_provider=data_provider,
                    state_store=None,
                )
            )
        elif "sentiment" in sid or "cb" in sid:
            strategies.append(
                CBSentimentShiftStrategy(
                    CBSentimentConfig(**scfg) if scfg else CBSentimentConfig(),
                    data_provider=data_provider,
                    nlp_provider=nlp_provider,
                    state_store=None,
                )
            )
    if not strategies:
        strategies.append(
            RateDiffMRStrategy(RateDiffMRConfig(), data_provider=data_provider)
        )
    return strategies


def build_coordinator(
    config: dict[str, Any],
    strategies: list[Any],
    oms: OrderManager,
    broker: Any,
) -> PortfolioCoordinator | None:
    """Construct the PortfolioCoordinator from config.

    Returns None and logs a warning if DB is unreachable — the engine will run
    in legacy direct-OMS mode in that case rather than refusing to start.
    """
    portfolio_cfg = config.get("portfolio", {})
    constraints_cfg = portfolio_cfg.get("constraints", {})
    constraints = PortfolioConstraints(**constraints_cfg) if constraints_cfg else None

    try:
        engine = _build_db_engine()
        state = PortfolioStateStore(engine)
    except Exception:
        logger.exception(
            "Failed to construct PortfolioStateStore; engine will run in legacy mode"
        )
        return None

    coordinator = PortfolioCoordinator(
        strategies=strategies,
        oms=oms,
        broker=broker,
        state=state,
        constraints=constraints,
    )

    initial_weights = portfolio_cfg.get("initial_weights")
    if initial_weights:
        coordinator.initialize_allocations(initial_weights)
    else:
        # Default to equal weight across all strategies in live mode (paper-mode
        # promotion via D5/CL-6vv will override this once that workflow lands).
        n = len(strategies)
        coordinator.initialize_allocations({s.id: 1.0 / n for s in strategies})

    return coordinator


def build_cold_start_reconciler(
    config: dict[str, Any],
    strategies: list[Any],
    oms: OrderManager,
    broker: Any,
) -> PositionReconciler | None:
    """Construct PositionReconciler for cold-start reconciliation.

    Returns None if a strategy state store cannot be obtained (e.g. when
    strategies aren't using one or DB is unreachable). The engine still starts;
    cold-start reconciliation is just skipped.
    """
    # Attempt to find a strategy state store from one of the strategies.
    state_store: Any | None = None
    for s in strategies:
        candidate = getattr(s, "state", None) or getattr(s, "state_store", None)
        if candidate is not None and hasattr(candidate, "get_current_position"):
            state_store = candidate
            break

    if state_store is None:
        logger.info(
            "No strategy state store available — cold-start reconciliation skipped"
        )
        return None

    policy_cfg = config.get("reconciliation", {}).get("policy", {})
    policy = ReconciliationPolicy(**policy_cfg) if policy_cfg else ReconciliationPolicy()
    return PositionReconciler(broker, oms, state_store, strategies, policy)


async def run_engine(practice: bool) -> None:
    log_dir = Path(os.environ.get("FX_LOG_DIR", "logs"))
    setup_logging("live_engine", log_dir)

    config = load_config(CONFIG_PATH)
    broker = build_broker(practice)
    oms = OrderManager(broker)
    strategies = build_strategies(config, broker, oms)
    coordinator = build_coordinator(config, strategies, oms, broker)
    reconciler = build_cold_start_reconciler(config, strategies, oms, broker)
    engine = LiveEngine(
        strategies, oms, broker,
        coordinator=coordinator,
        cold_start_reconciler=reconciler,
    )

    # Wire web API to live state
    try:
        from src.web.api import set_runtime
        set_runtime(broker, oms, strategies)
        logger.info("Web API runtime wired")
    except Exception:
        pass

    loop = asyncio.get_event_loop()

    async def _shutdown() -> None:
        logger.info("Graceful shutdown initiated")
        engine.running = False
        oms.halt_new_trades()
        await asyncio.sleep(1)
        logger.info("Shutdown complete")

    def _handler(sig: signal.Signals) -> None:
        logger.info("Received %s", sig.name)
        loop.create_task(_shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handler, sig)
        except NotImplementedError:
            pass

    logger.info("Starting curLit live engine (practice=%s)", practice)
    await engine.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="curLit live trading engine")
    parser.add_argument("--practice", action="store_true", default=True)
    parser.add_argument("--live", dest="practice", action="store_false")
    args = parser.parse_args()
    asyncio.run(run_engine(args.practice))


if __name__ == "__main__":
    main()
