"""Strategy state store — persistent signal and trade history, crash recovery."""

import json
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)


class StrategyStateStore:
    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._create_tables()

    def _create_tables(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS strategy_signals (
                    ts          TIMESTAMPTZ NOT NULL,
                    strategy_id TEXT NOT NULL,
                    signal_data JSONB,
                    PRIMARY KEY (ts, strategy_id)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS strategy_trades (
                    ts          TIMESTAMPTZ NOT NULL,
                    strategy_id TEXT NOT NULL,
                    action      TEXT NOT NULL,
                    details     JSONB,
                    PRIMARY KEY (ts, strategy_id, action)
                )
            """))

    def record_signal(self, strategy_id: str, ts: datetime, signal: dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO strategy_signals (ts, strategy_id, signal_data)
                VALUES (:ts, :sid, :data)
                ON CONFLICT (ts, strategy_id) DO UPDATE
                SET signal_data = EXCLUDED.signal_data
            """), {"ts": ts, "sid": strategy_id, "data": json.dumps(signal)})

    def record_entry(
        self, strategy_id: str, ts: datetime, signal: dict[str, Any], size: float,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO strategy_trades (ts, strategy_id, action, details)
                VALUES (:ts, :sid, 'entry', :details)
            """), {"ts": ts, "sid": strategy_id, "details": json.dumps({**signal, "size": size})})

    def record_exit(self, strategy_id: str, ts: datetime, reason: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO strategy_trades (ts, strategy_id, action, details)
                VALUES (:ts, :sid, 'exit', :details)
            """), {"ts": ts, "sid": strategy_id, "details": json.dumps({"reason": reason})})

    def get_current_position(self, strategy_id: str) -> dict[str, Any] | None:
        """Return last entry if position is open, None if flat (crash recovery)."""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT action, details, ts FROM strategy_trades
                WHERE strategy_id = :sid
                ORDER BY ts DESC LIMIT 1
            """), {"sid": strategy_id}).fetchone()
        if result is None or result[0] == "exit":
            return None
        return {**json.loads(result[1]), "entry_ts": result[2]}
