"""Live trading engine entrypoint — instantiate and run the full stack."""

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

import yaml

from src.monitoring.logging_setup import setup_logging
from src.monitoring.metrics import start_metrics_server
from src.runtime.live_engine import LiveEngine
from src.execution.paper_broker import PaperBroker
from src.execution.oanda_broker import OandaBroker
from src.execution.oms import OrderManager
from src.strategies.rate_diff_mean_reversion import RateDiffMRStrategy, RateDiffMRConfig
from src.strategies.cb_sentiment_shift import CBSentimentShiftStrategy, CBSentimentConfig
from src.data.provider import DataProvider
from src.nlp.provider import NLPDataProvider
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(os.environ.get("FX_CONFIG", "configs/live_portfolio.yaml"))


def load_config(path: Path) -> dict:
    if not path.exists():
        logger.warning("Config not found at %s — using defaults", path)
        return {}
    return yaml.safe_load(path.read_text())


def build_broker(practice: bool) -> object:
    if practice:
        return PaperBroker(initial_capital=100_000)
    oanda_key = os.environ.get("OANDA_API_KEY", "")
    oanda_id = os.environ.get("OANDA_ACCOUNT_ID", "")
    if not oanda_key:
        logger.error("OANDA_API_KEY not set — falling back to paper broker")
        return PaperBroker(initial_capital=100_000)
    return OandaBroker(oanda_key, oanda_id, practice=False)


def build_strategies(config: dict, broker, oms) -> list:
    db_url = os.environ.get(
        "DATABASE_URL",
        f"postgresql+psycopg2://{os.environ.get('POSTGRES_USER', 'fx')}:{os.environ.get('POSTGRES_PASSWORD', 'changeme')}@{os.environ.get('POSTGRES_HOST', 'localhost')}:5432/{os.environ.get('POSTGRES_DB', 'fx')}",
    )
    engine = create_engine(db_url)
    data_provider = DataProvider(engine)
    nlp_provider = NLPDataProvider(engine)

    strategies = []
    for sconf in config.get("strategies", []):
        sid = sconf.get("id", "")
        scfg = sconf.get("config", {})
        if "rate_diff" in sid:
            strategies.append(RateDiffMRStrategy(
                RateDiffMRConfig(**scfg) if scfg else RateDiffMRConfig(),
                data_provider=data_provider, state_store=None,
            ))
        elif "sentiment" in sid or "cb" in sid:
            strategies.append(CBSentimentShiftStrategy(
                CBSentimentConfig(**scfg) if scfg else CBSentimentConfig(),
                data_provider=data_provider, nlp_provider=nlp_provider, state_store=None,
            ))
    if not strategies:
        strategies.append(RateDiffMRStrategy(RateDiffMRConfig(), data_provider=data_provider))
    return strategies


async def run_engine(practice: bool) -> None:
    log_dir = Path(os.environ.get("FX_LOG_DIR", "logs"))
    setup_logging("live_engine", log_dir)
    start_metrics_server(port=8000)

    config = load_config(CONFIG_PATH)
    broker = build_broker(practice)
    oms = OrderManager(broker)
    strategies = build_strategies(config, broker, oms)
    engine = LiveEngine(strategies, oms, broker)

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
