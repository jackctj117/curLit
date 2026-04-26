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

from src.data.economic_calendar import (
    BlackoutEvaluator,
    EconomicCalendar,
    load_calendar_from_yaml,
)
from src.data.provider import DataProvider
from src.execution.oanda_broker import OandaBroker
from src.execution.oms import OrderManager
from src.execution.paper_broker import PaperBroker
from src.execution.rejection import RejectionHandler
from src.execution.trade_journal import TradeJournal
from src.models.feature_versioning import FeatureSnapshotStore
from src.monitoring.logging_setup import setup_logging
from src.nlp.provider import NLPDataProvider
from src.portfolio import (
    PortfolioConstraints,
    PortfolioCoordinator,
    PortfolioStateStore,
    PositionReconciler,
    PreTradeValidator,
    ReconciliationPolicy,
)
from src.runtime.live_engine import LiveEngine
from src.strategies.carry_vol_filter import (
    CarryVolFilterConfig,
    CarryVolFilterStrategy,
)
from src.strategies.cb_sentiment_shift import CBSentimentConfig, CBSentimentShiftStrategy
from src.strategies.rate_diff_mean_reversion import RateDiffMRConfig, RateDiffMRStrategy
from src.strategies.state import StrategyStateStore

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


def build_trade_journal() -> TradeJournal | None:
    """Construct the audit-trail journal. Returns None if the DB is unreachable
    so the engine still starts — losing audit on a single boot is preferable
    to refusing to trade.
    """
    try:
        engine = _build_db_engine()
        return TradeJournal(engine)
    except Exception:
        logger.exception(
            "Failed to construct TradeJournal; engine will run without audit log"
        )
        return None


def build_feature_snapshot_store() -> FeatureSnapshotStore | None:
    """Construct the feature-snapshot store for reproducibility tagging.

    Returns None if the DB is unreachable so the engine still starts —
    losing snapshot capture on a single boot is preferable to refusing
    to trade. Strategies that get None skip emitting snapshots.
    """
    try:
        engine = _build_db_engine()
        return FeatureSnapshotStore(engine)
    except Exception:
        logger.exception(
            "Failed to construct FeatureSnapshotStore; intents will not "
            "carry feature-snapshot ids"
        )
        return None


def build_blackout_evaluator(
    config: dict[str, Any],
) -> BlackoutEvaluator | None:
    """Load the economic-event calendar and build a BlackoutEvaluator.

    Path resolution: explicit `calendar.path` in config wins; otherwise
    configs/economic_calendar.yaml is the default. Missing file → None
    (engine runs without blackout enforcement, which is non-fatal but logged).
    """
    cal_cfg = config.get("calendar", {})
    explicit_path = cal_cfg.get("path")
    path = Path(explicit_path) if explicit_path else Path("configs/economic_calendar.yaml")
    if not path.exists():
        logger.info(
            "Economic calendar not found at %s — blackout enforcement disabled",
            path,
        )
        return None
    try:
        calendar: EconomicCalendar = load_calendar_from_yaml(path)
    except Exception:
        logger.exception(
            "Failed to load economic calendar from %s — blackout enforcement disabled",
            path,
        )
        return None
    return BlackoutEvaluator(calendar)


def build_strategies(
    config: dict[str, Any],
    broker: Any,
    oms: OrderManager,
    snapshot_store: FeatureSnapshotStore | None = None,
) -> list[Any]:
    engine = _build_db_engine()
    data_provider = DataProvider(engine)
    nlp_provider = NLPDataProvider(engine)
    # Single shared StrategyStateStore across all strategies — the cold-start
    # reconciler discovers it via getattr(strategy, "state", ...) and uses
    # get_current_position(strategy_id) to aggregate per-symbol holdings.
    # If table-creation fails (e.g. DB unreachable), strategies degrade to
    # state_store=None and reconciliation is skipped — non-fatal.
    try:
        state_store: Any | None = StrategyStateStore(engine)
    except Exception:
        logger.exception(
            "Failed to construct StrategyStateStore; strategies start without state",
        )
        state_store = None

    strategies: list[Any] = []
    for sconf in config.get("strategies", []):
        sid = sconf.get("id", "")
        scfg = sconf.get("config", {})
        if "rate_diff" in sid:
            strategies.append(
                RateDiffMRStrategy(
                    RateDiffMRConfig(**scfg) if scfg else RateDiffMRConfig(),
                    data_provider=data_provider,
                    state_store=state_store,
                    snapshot_store=snapshot_store,
                )
            )
        elif "sentiment" in sid or "cb" in sid:
            strategies.append(
                CBSentimentShiftStrategy(
                    CBSentimentConfig(**scfg) if scfg else CBSentimentConfig(),
                    data_provider=data_provider,
                    nlp_provider=nlp_provider,
                    state_store=state_store,
                    snapshot_store=snapshot_store,
                )
            )
        elif "carry" in sid or "vol_filter" in sid:
            strategies.append(
                CarryVolFilterStrategy(
                    CarryVolFilterConfig(**scfg) if scfg else CarryVolFilterConfig(),
                    data_provider=data_provider,
                    state_store=state_store,
                    snapshot_store=snapshot_store,
                )
            )
    if not strategies:
        strategies.append(
            RateDiffMRStrategy(
                RateDiffMRConfig(),
                data_provider=data_provider,
                state_store=state_store,
                snapshot_store=snapshot_store,
            )
        )
    return strategies


def _supported_instruments(
    config: dict[str, Any],
    strategies: list[Any],
) -> set[str] | None:
    """Resolve the supported-instruments whitelist for the pre-trade gate.

    Priority: explicit `portfolio.supported_instruments` list in config wins.
    Otherwise we union every strategy's `.symbols` to derive the set the
    portfolio actually trades on. Returns None to disable the whitelist
    (validator accepts all symbols).
    """
    portfolio_cfg = config.get("portfolio", {})
    explicit = portfolio_cfg.get("supported_instruments")
    if isinstance(explicit, list) and explicit:
        return set(explicit)
    if not strategies:
        return None
    derived: set[str] = set()
    for s in strategies:
        symbols = getattr(s, "symbols", None) or []
        for sym in symbols:
            derived.add(sym)
    return derived if derived else None


def build_coordinator(
    config: dict[str, Any],
    strategies: list[Any],
    oms: OrderManager,
    broker: Any,
    blackout_evaluator: BlackoutEvaluator | None = None,
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

    # Pre-trade gate: validates margin / leverage / per-pair / per-currency /
    # tradability per intent before the OMS sees it. Supported instruments
    # default to the union of all strategy symbols + USD-base inverses; an
    # explicit set in config overrides.
    effective_constraints = constraints or PortfolioConstraints()
    supported = _supported_instruments(config, strategies)
    pre_trade_validator = PreTradeValidator(
        broker=broker,
        constraints=effective_constraints,
        supported_instruments=supported,
        blackout_evaluator=blackout_evaluator,
    )

    coordinator = PortfolioCoordinator(
        strategies=strategies,
        oms=oms,
        broker=broker,
        state=state,
        constraints=effective_constraints,
        pre_trade_validator=pre_trade_validator,
        blackout_evaluator=blackout_evaluator,
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
    journal: TradeJournal | None = None,
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
    return PositionReconciler(
        broker, oms, state_store, strategies, policy, journal=journal,
    )


async def run_engine(practice: bool) -> None:
    log_dir = Path(os.environ.get("FX_LOG_DIR", "logs"))
    setup_logging("live_engine", log_dir)

    config = load_config(CONFIG_PATH)
    broker = build_broker(practice)
    journal = build_trade_journal()
    snapshot_store = build_feature_snapshot_store()
    blackout_evaluator = build_blackout_evaluator(config)
    rejection_handler = RejectionHandler(journal=journal)
    oms = OrderManager(broker, rejection_handler=rejection_handler, journal=journal)
    strategies = build_strategies(config, broker, oms, snapshot_store=snapshot_store)
    coordinator = build_coordinator(
        config, strategies, oms, broker,
        blackout_evaluator=blackout_evaluator,
    )
    reconciler = build_cold_start_reconciler(
        config, strategies, oms, broker, journal=journal,
    )
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
    shutdown_started = False

    async def _shutdown() -> None:
        # Idempotent — multiple SIGTERMs (or both SIGINT+SIGTERM during a
        # ctrl-C+kill rollover) shouldn't trigger multiple shutdowns. Without
        # this we previously logged "Shutdown complete" twice.
        nonlocal shutdown_started
        if shutdown_started:
            return
        shutdown_started = True
        logger.info("Graceful shutdown initiated")
        await engine.graceful_shutdown(timeout=10)

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
