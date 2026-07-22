"""Live trading engine entrypoint — instantiate and run the full stack."""

import argparse
import asyncio
import contextlib
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
from src.execution.broker import build_fx_broker, effective_broker_mode
from src.execution.oms import OrderManager
from src.execution.rejection import RejectionHandler
from src.execution.trade_journal import TradeJournal
from src.models.feature_versioning import FeatureSnapshotStore
from src.monitoring.data_health import log_startup_health
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
from src.strategies.event_driven import EventDrivenConfig, EventDrivenStrategy
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


# Three broker modes the engine entrypoint accepts (CL-920k):
#   "paper"          — in-process PaperBroker, no network, $100k start
#                       capital. Pure simulation; no real spreads.
#   "oanda-practice" — OandaBroker against api-fxpractice.oanda.com.
#                       Real OANDA practice account, real spreads,
#                       real fills, but no real money. Practice
#                       account state survives engine restarts.
#   "oanda-live"     — OandaBroker against api-fxtrade.oanda.com.
#                       REAL MONEY. Requires --confirm-live.
BROKER_MODES = (
    "paper", "oanda-practice", "oanda-live",
    # CL-poly-2: paper broker against live Polymarket order books.
    "polymarket-paper",
    # CL-poly-3: testnet (Amoy) — full sign + submit chain, no real money.
    "polymarket-amoy",
    # CL-poly-3: mainnet — REAL MONEY. HARD-GATED until preflight passes.
    "polymarket-mainnet",
)


def build_broker(mode: str) -> Any:
    if mode in ("paper", "oanda-practice", "oanda-live"):
        # Fail-fast credentials policy lives in build_fx_broker (CL-qyav
        # P1): requesting an OANDA mode without credentials RAISES
        # BrokerCredentialsError instead of silently handing back a
        # PaperBroker while the whole system reports "oanda". Explicit
        # ALLOW_PAPER_FALLBACK=1 restores the old downgrade (CRITICAL log).
        return build_fx_broker(mode)
    if mode == "polymarket-paper":
        # Paper-only — no wallet, no chain, no signer. CL-poly-2.
        from src.execution.polymarket_paper_broker import PolymarketPaperBroker
        return PolymarketPaperBroker()
    if mode == "polymarket-amoy":
        # Testnet — full chain wiring, no real money. CL-poly-3 scaffold.
        from src.execution.polymarket_broker import PolymarketBroker
        from src.execution.polymarket_preflight import run as preflight_run
        # Testnet preflight is permissive about the vault path (env
        # vars are fine for dev smoke).
        failures = preflight_run("amoy", require_vault=False)
        if failures:
            msg = (
                "polymarket-amoy preflight failed:\n  - "
                + "\n  - ".join(failures)
            )
            raise RuntimeError(msg)
        return PolymarketBroker(env="amoy")
    if mode == "polymarket-mainnet":
        # HARD GATE — see CL-poly-3 acceptance for the unlock checklist.
        # The override env var POLYMARKET_MAINNET_UNLOCK is the one
        # documented gate to flip this on, and it logs WARNING when set
        # so any audit log reflects the unlock.
        if os.environ.get("POLYMARKET_MAINNET_UNLOCK") != "1":
            msg = (
                "polymarket-mainnet is HARD-GATED. Real-money trading on "
                "Polymarket is disabled until the CL-poly-3 acceptance "
                "criteria are satisfied (see bd show CL-poly-3). To "
                "explicitly unlock after operator-validated bringup, set "
                "POLYMARKET_MAINNET_UNLOCK=1 in the environment. Until then "
                "use polymarket-amoy (testnet) or polymarket-paper (no "
                "chain at all)."
            )
            raise RuntimeError(msg)
        logger.warning(
            "POLYMARKET_MAINNET_UNLOCK=1 — real-money mode active. "
            "Operator must have completed CL-poly-3 acceptance gates.",
        )
        from src.execution.polymarket_broker import PolymarketBroker
        from src.execution.polymarket_preflight import run as preflight_run
        # Mainnet preflight is strict — vault path required.
        failures = preflight_run("mainnet", require_vault=True)
        if failures:
            msg = (
                "polymarket-mainnet preflight failed:\n  - "
                + "\n  - ".join(failures)
            )
            raise RuntimeError(msg)
        return PolymarketBroker(env="mainnet")
    msg = f"unknown broker mode: {mode!r} (must be one of {BROKER_MODES})"
    raise ValueError(msg)


def _build_db_engine() -> Any:
    # Shared helper (CL-8lv6): warns once per process when the well-known
    # default Postgres password is in effect instead of silently connecting.
    from src.data.db_env import build_db_url
    return create_engine(build_db_url())


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
    # CL-q4n1: surface data-starved strategies loudly at boot (logs a WARN
    # report; never blocks startup) so a silently-gated strategy is visible.
    log_startup_health(engine)
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
        # CL-mnhw: honor an explicit `enabled: false` on any strategy
        # entry. Missing/true keeps the legacy always-on behavior.
        if sconf.get("enabled", True) is False:
            logger.info("Strategy %s disabled via config (enabled: false) — skipping", sid)
            continue
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
        elif "event" in sid:
            # CL-mnhw: current-events consumer. Gets the raw DB engine
            # (not just DataProvider) for geo_events polling + status
            # transitions. NO-OP that logs once if the producer's
            # geo_events migration hasn't been applied yet.
            strategies.append(
                EventDrivenStrategy(
                    EventDrivenConfig(**scfg) if scfg else EventDrivenConfig(),
                    data_provider=data_provider,
                    state_store=state_store,
                    snapshot_store=snapshot_store,
                    db_engine=engine,
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


def build_kill_switch_manager(broker: Any, oms: OrderManager) -> Any:
    """CL-ep0c: construct the KillSwitchManager for the live engine.

    Config comes from the active risk profile's kill_switches block; the
    DataProvider feeds the open_position_correlation switch (None-safe —
    if the DB is unreachable that switch simply never fires).

    FAIL CLOSED (CL-r8gv): a construction failure RAISES instead of
    returning None — an engine that silently boots without its automated
    brakes is exactly the "operators trust Arch §5.5 and are unprotected"
    failure the review flagged. A corrupt risk profile must stop the boot.
    """
    from dataclasses import asdict  # noqa: PLC0415

    from src.risk.kill_switches import KillSwitchManager  # noqa: PLC0415
    from src.risk.risk_profile import load_active_profile  # noqa: PLC0415

    data_provider: DataProvider | None = None
    try:
        data_provider = DataProvider(_build_db_engine())
    except Exception:
        logger.exception(
            "DataProvider unavailable for kill switches — "
            "open_position_correlation switch disabled",
        )
    try:
        return KillSwitchManager(
            broker, oms,
            config=asdict(load_active_profile().kill_switches),
            data_provider=data_provider,
        )
    except Exception as exc:
        raise RuntimeError(
            "Kill-switch construction failed — refusing to start an "
            "unprotected engine (fix the risk profile / config and retry)",
        ) from exc
        return None


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


async def run_engine(broker_mode: str = "paper") -> None:
    log_dir = Path(os.environ.get("FX_LOG_DIR", "logs"))
    setup_logging("live_engine", log_dir)

    config = load_config(CONFIG_PATH)
    broker = build_broker(broker_mode)
    # Report the mode we are ACTUALLY in (CL-qyav P1): if the explicit
    # ALLOW_PAPER_FALLBACK opt-in downgraded an OANDA request to paper,
    # every subsequent log/status line must say so.
    effective_mode = effective_broker_mode(broker_mode, broker)
    if effective_mode != broker_mode:
        logger.critical(
            "Requested broker mode %r but running %r — ALLOW_PAPER_FALLBACK "
            "downgrade is active; NOT connected to OANDA",
            broker_mode, effective_mode,
        )
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
    kill_switch_manager = build_kill_switch_manager(broker, oms)
    engine = LiveEngine(
        strategies, oms, broker,
        coordinator=coordinator,
        cold_start_reconciler=reconciler,
        kill_switch_manager=kill_switch_manager,
    )

    # Wire web API to live state. A failure here means the control plane
    # (halt/resume/close endpoints) is DEAD while the engine trades — say
    # so loudly instead of silently continuing (CL-b0ws).
    try:
        from src.web.api import set_runtime
        set_runtime(broker, oms, strategies)
        logger.info("Web API runtime wired")
    except Exception:
        logger.exception(
            "Web API runtime wiring FAILED — /api control endpoints will "
            "not reflect or control this engine",
        )

    loop = asyncio.get_running_loop()
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
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _handler, sig)

    # CL-9eli post-mortem hook: snapshot critical source files now and
    # warn if they change under us. Catches the "engine running stale
    # code because nobody restarted after a fix" failure mode.
    drift_task: asyncio.Task[Any] | None = None
    try:
        from src.runtime.source_drift import SourceDriftWatcher
        watcher = SourceDriftWatcher()
        # Retain + observe (CL-8lv6): a bare create_task can be GC'd and
        # its exceptions vanish — the watcher would die silently and the
        # stale-code canary it exists to provide would be gone.
        drift_task = loop.create_task(
            watcher.run_forever(), name="source_drift_watcher",
        )

        def _drift_done(t: asyncio.Task[Any]) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                logger.error(
                    "SourceDriftWatcher DIED — stale-code canary is gone "
                    "until restart", exc_info=exc,
                )

        drift_task.add_done_callback(_drift_done)
    except Exception:
        logger.exception("SourceDriftWatcher failed to start (non-fatal)")

    logger.info("Starting curLit live engine (broker=%s)", effective_mode)
    await engine.run()


def main() -> None:
    # Auto-load .env so OANDA / Postgres / Telegram creds are available
    # without first sourcing the file. Explicit env vars still win.
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()
    parser = argparse.ArgumentParser(description="curLit live trading engine")
    parser.add_argument(
        "--broker",
        choices=BROKER_MODES,
        default="paper",
        help=(
            "Broker selection (CL-920k). 'paper' = in-process PaperBroker "
            "(default, no network). 'oanda-practice' = OANDA practice "
            "API (real spreads, real fills, no real money). 'oanda-live' "
            "= REAL MONEY — requires --confirm-live."
        ),
    )
    parser.add_argument(
        "--confirm-live", action="store_true",
        help="Required when --broker=oanda-live (REAL MONEY guard).",
    )
    # Legacy flags — kept for backwards compatibility; map to --broker.
    parser.add_argument(
        "--practice", action="store_true",
        help="Deprecated alias: --broker=paper.",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Deprecated alias: --broker=oanda-live (still requires --confirm-live).",
    )
    args = parser.parse_args()

    # Resolve legacy aliases.
    mode = args.broker
    if args.live:
        mode = "oanda-live"
    elif args.practice:
        mode = "paper"

    if mode == "oanda-live" and not args.confirm_live:
        parser.error(
            "--broker=oanda-live requires --confirm-live (REAL MONEY guard)",
        )

    asyncio.run(run_engine(mode))


if __name__ == "__main__":
    main()
