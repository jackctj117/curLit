"""Durable OANDA transaction-stream catch-up (CL-pksi finding).

Oracle: a fake OANDA account whose transaction log is the single source of
truth. Every ORDER_FILL in that log must reach the journal EXACTLY once no
matter how deliveries interleave (live stream, reconnect replay, restart),
and the durable checkpoint may only name a transaction whose fill (if it is
one) is already journaled. HTTP is mocked at the httpx boundary — REST via a
fake ``client.get``, the stream via a fake ``httpx.AsyncClient``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from sqlalchemy import create_engine, text

from src.execution.fill_stream import (
    FillStreamCatchUp,
    InMemoryTransactionCheckpointStore,
    SqlTransactionCheckpointStore,
)
from src.execution.oanda_broker import OandaBroker
from src.execution.oms import OrderManager
from src.execution.trade_journal import EventType

ACC = "ACC-1"


def _txn(tid: int, kind: str = "ORDER_FILL", client: str | None = None) -> dict[str, Any]:
    t: dict[str, Any] = {"id": str(tid), "type": kind, "time": "2026-10-06T10:00:00Z"}
    if kind == "ORDER_FILL":
        t.update(
            orderID=str(tid - 1000),
            instrument="EUR_USD",
            units="1000",
            price="1.0841",
            # OANDA OrderFillTransaction field for the filled order's client id.
            clientOrderID=client or f"intent-{tid}",
        )
    return t


class _Account:
    """Fake OANDA account: the transaction log + what is 'visible' so far."""

    def __init__(self, log: list[dict[str, Any]]) -> None:
        self.log = log
        self.visible_upto = 0
        self.fail_sinceid = 0  # number of upcoming sinceid calls that 500
        self.sinceid_calls: list[int] = []

    def visible(self) -> list[dict[str, Any]]:
        return [t for t in self.log if int(t["id"]) <= self.visible_upto]


class _Resp:
    def __init__(self, status: int, body: dict[str, Any]) -> None:
        self.status_code = status
        self._body = body
        self.headers: dict[str, str] = {}
        self.text = json.dumps(body)

    def json(self) -> dict[str, Any]:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "err",
                request=httpx.Request("GET", "http://x"),
                response=httpx.Response(self.status_code),
            )


class _RestClient:
    base_url = "https://api-fxpractice.oanda.com"

    def __init__(self, acct: _Account) -> None:
        self.acct = acct

    def get(self, path: str, **_: Any) -> _Resp:
        if "/transactions/sinceid?id=" in path:
            since = int(path.rsplit("=", 1)[1])
            self.acct.sinceid_calls.append(since)
            if self.acct.fail_sinceid > 0:
                self.acct.fail_sinceid -= 1
                return _Resp(500, {"errorMessage": "boom"})
            txns = [t for t in self.acct.visible() if int(t["id"]) > since]
            return _Resp(
                200,
                {"transactions": txns, "lastTransactionID": str(self.acct.visible_upto)},
            )
        if path.endswith("/summary"):
            return _Resp(200, {"account": {}, "lastTransactionID": str(self.acct.visible_upto)})
        raise AssertionError(f"unexpected GET {path}")


def _broker(acct: _Account) -> OandaBroker:
    b = OandaBroker.__new__(OandaBroker)  # skip network __init__
    b.account_id = ACC
    b.api_key = "k"
    b.headers = {}
    b.client = _RestClient(acct)  # type: ignore[assignment]
    return b


class _Journal:
    def __init__(self) -> None:
        self.rows: list[SimpleNamespace] = []
        self.fail_fill_ids: set[str] = set()

    def record(
        self,
        *,
        event_type: EventType,
        payload: dict[str, Any],
        intent_id: str | None,
        strategy_id: str | None,
        symbol: str | None,
    ) -> None:
        if (
            event_type is EventType.ORDER_FILLED
            and str(payload.get("fill_id")) in self.fail_fill_ids
        ):
            self.fail_fill_ids.discard(str(payload.get("fill_id")))
            raise RuntimeError("journal down")
        self.rows.append(
            SimpleNamespace(event_type=event_type, payload=payload, intent_id=intent_id)
        )

    def query_by_intent(self, intent_id: str) -> list[SimpleNamespace]:
        return [r for r in self.rows if r.intent_id == intent_id]

    def fill_ids(self) -> list[str]:
        return [
            str(r.payload["fill_id"]) for r in self.rows if r.event_type is EventType.ORDER_FILLED
        ]


_CHECKED_STORES: list[_CheckedStore] = []


class _CheckedStore(InMemoryTransactionCheckpointStore):
    """Records an oracle violation on every write that would let the
    checkpoint pass a fill not yet journaled. Violations are collected (not
    raised — the catch-up swallows exceptions to keep the stream alive) and
    asserted empty after every test by the autouse fixture below."""

    def __init__(self, journal: _Journal, acct: _Account) -> None:
        super().__init__()
        self.journal = journal
        self.acct = acct
        #: Fills at or below the starting checkpoint were journaled by an
        #: earlier session (outside this test's journal).
        self.floor = -1
        self.violations: list[str] = []
        _CHECKED_STORES.append(self)

    def advance(self, account_id: str, transaction_id: int) -> None:
        journaled = set(self.journal.fill_ids())
        for t in self.acct.log:
            if (
                t["type"] == "ORDER_FILL"
                and self.floor < int(t["id"]) <= transaction_id
                and t["id"] not in journaled
            ):
                self.violations.append(
                    f"checkpoint {transaction_id} passes unjournaled fill {t['id']}"
                )
        super().advance(account_id, transaction_id)


@pytest.fixture(autouse=True)
def _no_checkpoint_violations():  # noqa: ANN202
    _CHECKED_STORES.clear()
    yield
    violations = [v for st in _CHECKED_STORES for v in st.violations]
    _CHECKED_STORES.clear()
    assert violations == []


# --- stream harness ----------------------------------------------------------


class _StreamResp:
    def __init__(self, acct: _Account, upto: int, lines: list[str], then: BaseException | None):
        self.status_code = 200
        self._acct, self._upto, self._lines, self._then = acct, upto, lines, then

    async def aiter_lines(self):  # noqa: ANN202
        for ln in self._lines:
            yield ln
        if self._then is not None:
            raise self._then


class _Ctx:
    def __init__(self, conn: dict[str, Any], acct: _Account) -> None:
        self.conn, self.acct = conn, acct

    async def __aenter__(self) -> _StreamResp:
        # Connecting makes everything up to `upto` exist at the venue.
        self.acct.visible_upto = max(self.acct.visible_upto, self.conn["upto"])
        return _StreamResp(self.acct, self.conn["upto"], self.conn["lines"], self.conn.get("then"))

    async def __aexit__(self, *a: object) -> bool:
        return False


class _AsyncClient:
    def __init__(self, script: list[dict[str, Any]], state: dict[str, int], acct: _Account):
        self.script, self.state, self.acct = script, state, acct

    async def __aenter__(self) -> _AsyncClient:
        return self

    async def __aexit__(self, *a: object) -> bool:
        return False

    def stream(self, *a: object, **k: object) -> _Ctx:
        i = self.state["i"]
        self.state["i"] += 1
        return _Ctx(self.script[min(i, len(self.script) - 1)], self.acct)


def _drive(
    monkeypatch: pytest.MonkeyPatch,
    broker: OandaBroker,
    cu: FillStreamCatchUp,
    script: list[dict[str, Any]],
    n_live: int,
) -> list[str]:
    state = {"i": 0}
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **k: _AsyncClient(script, state, broker.client.acct)
    )

    async def _no_sleep(*_a: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    async def hook() -> None:
        cu.catch_up()

    async def run() -> list[str]:
        got: list[str] = []
        gen = broker.stream_transactions(on_connect=hook)
        while len(got) < n_live:
            fill = await gen.__anext__()
            got.append(fill["transaction_id"])
            cu.handle_live(fill)
        await gen.aclose()
        return got

    return asyncio.run(run())


def _line(t: dict[str, Any]) -> str:
    return json.dumps(t)


LOG = [_txn(101), _txn(102), _txn(103, "ORDER_CREATE"), _txn(104), _txn(105)]


def _setup(log: list[dict[str, Any]] = LOG, checkpoint: int | None = 100):
    acct = _Account(log)
    broker = _broker(acct)
    journal = _Journal()
    oms = OrderManager(broker, journal=journal)  # type: ignore[arg-type]
    store = _CheckedStore(journal, acct)
    if checkpoint is not None:
        InMemoryTransactionCheckpointStore.advance(store, ACC, checkpoint)
        store.floor = checkpoint
    cu = FillStreamCatchUp(broker, oms, store, ACC)
    return acct, broker, journal, oms, store, cu


class TestReconnectReplay:
    def test_reconnect_replays_missed_fills_exactly_once(self, monkeypatch, caplog):
        acct, broker, journal, _oms, store, cu = _setup()
        script = [
            # connect 1: venue at 101; the stream delivers 101 live, then drops.
            {"upto": 101, "lines": [_line(_txn(101))], "then": httpx.ReadError("reset")},
            # connect 2: 102-104 happened while disconnected; 105 streams live.
            {"upto": 105, "lines": [_line(_txn(105))]},
        ]
        caplog.set_level("INFO")
        live = _drive(monkeypatch, broker, cu, script, n_live=2)
        assert live == ["101", "105"]
        # Every fill in the venue log journaled exactly once (101 and 105 were
        # delivered twice: replay + live).
        assert sorted(journal.fill_ids()) == ["101", "102", "104", "105"]
        assert store.load(ACC) == 105
        assert acct.sinceid_calls == [100, 101]
        # Replay 1 covered 101; replay 2 covered 102, 104 and 105 (105 was
        # already at the venue when the second replay ran).
        assert any("REPLAYED 3 missed fill" in r.getMessage() for r in caplog.records)

    def test_failed_replay_keeps_checkpoint_and_stream_alive(self, monkeypatch):
        acct, broker, journal, _oms, store, cu = _setup(checkpoint=101)
        acct.fail_sinceid = 1
        script = [
            {"upto": 104, "lines": [_line(_txn(104))], "then": httpx.ReadError("reset")},
            {"upto": 105, "lines": [_line(_txn(105))]},
        ]
        live = _drive(monkeypatch, broker, cu, script, n_live=2)
        assert live == ["104", "105"]  # stream stayed up after the failed replay
        # The second connect's replay recovered 102 (missed by the first).
        assert sorted(journal.fill_ids()) == ["102", "104", "105"]
        assert store.load(ACC) == 105

    def test_failed_replay_freezes_checkpoint_against_live_fills(self, monkeypatch):
        acct, broker, journal, _oms, store, cu = _setup(checkpoint=101)
        acct.fail_sinceid = 99  # replay keeps failing
        script = [{"upto": 104, "lines": [_line(_txn(104))]}]
        _drive(monkeypatch, broker, cu, script, n_live=1)
        # 104 journaled live, but advancing to 104 would skip unreplayed 102.
        assert journal.fill_ids() == ["104"]
        assert store.load(ACC) == 101
        assert cu.frozen

    def test_unjournaled_replay_fill_stops_before_advancing(self, monkeypatch):
        acct, broker, journal, _oms, store, cu = _setup(checkpoint=100)
        journal.fail_fill_ids = {"102"}
        script = [
            {"upto": 104, "lines": [_line(_txn(104))], "then": httpx.ReadError("reset")},
            {"upto": 105, "lines": [_line(_txn(105))]},
        ]
        _drive(monkeypatch, broker, cu, script, n_live=2)
        # Connect 1: 101 journaled, 102 failed -> checkpoint stuck at 101.
        # Connect 2: replay from 101 journals 102 (retry), 104 once, 105 once.
        assert sorted(journal.fill_ids()) == ["101", "102", "104", "105"]
        assert store.load(ACC) == 105
        assert 101 in [w[1] for w in store.writes]


class TestCatchUpUnit:
    def test_checkpoint_advances_only_after_journaling(self):
        acct, _broker_, journal, _oms, store, cu = _setup(checkpoint=100)
        acct.visible_upto = 105
        journal.fail_fill_ids = {"104"}
        assert cu.catch_up() == -1
        # 101, 102 journaled; 103 is a non-fill; 104 failed -> cursor <= 103.
        assert store.load(ACC) == 102
        assert "104" not in journal.fill_ids()
        assert cu.catch_up() == 2  # 104 (journal retry) + 105
        assert store.load(ACC) == 105
        assert sorted(journal.fill_ids()) == ["101", "102", "104", "105"]

    def test_bounded_pages(self, caplog):
        log = [_txn(100 + i) for i in range(1, 8)]  # 101..107, all fills
        acct, _b, journal, _oms, store, cu = _setup(log=log, checkpoint=100)
        acct.visible_upto = 107
        cu.page_size, cu.max_pages = 2, 2
        caplog.set_level("INFO")
        assert cu.catch_up() == 4  # two pages of two
        assert store.load(ACC) == 104
        assert cu.frozen  # truncated: live fills may not jump the backlog
        assert any("TRUNCATED" in r.getMessage() for r in caplog.records)
        assert acct.sinceid_calls == [100, 102]
        cu.max_pages = 10
        assert cu.catch_up() == 3
        assert not cu.frozen
        assert journal.fill_ids() == [str(i) for i in range(101, 108)]

    def test_no_checkpoint_seeds_from_account_last_transaction(self):
        acct, broker, journal, oms, _checked, _cu = _setup(checkpoint=None)
        # Seeding deliberately skips history, so the replay oracle store does
        # not apply here; use the plain store.
        store = InMemoryTransactionCheckpointStore()
        cu = FillStreamCatchUp(broker, oms, store, ACC)
        acct.visible_upto = 104
        assert cu.catch_up() == 0
        assert store.load(ACC) == 104
        assert journal.fill_ids() == []
        assert acct.sinceid_calls == []

    def test_restart_replay_does_not_rejournal(self):
        acct, broker, journal, _oms, store, cu = _setup(checkpoint=100)
        acct.visible_upto = 102
        assert cu.catch_up() == 2
        # Crash after journaling 102 but before its checkpoint write.
        store._rows[ACC] = 101
        restarted = FillStreamCatchUp(
            broker,
            OrderManager(broker, journal=journal),
            store,
            ACC,  # type: ignore[arg-type]
        )
        assert restarted.catch_up() == 0
        assert journal.fill_ids() == ["101", "102"]
        assert store.load(ACC) == 102


class TestReviewRound1:
    def test_fill_normalization_reads_client_order_id(self):
        b = _broker(_Account([]))
        ev = b._normalize_fill(_txn(201, client="intent-x"))
        assert ev["client_order_id"] == "intent-x"
        legacy = {"id": "202", "type": "ORDER_FILL", "clientExtensions": {"id": "intent-y"}}
        assert b._normalize_fill(legacy)["client_order_id"] == "intent-y"

    def test_restart_replay_skips_sync_fill_journaled_under_intent(self):
        """A synchronous FOK fill journaled under its intent id before a
        restart must not be journaled again when the replay delivers it."""
        acct, broker, journal, oms, store, cu = _setup(checkpoint=100)
        journal.record(
            event_type=EventType.ORDER_FILLED,
            payload={"side": "buy", "quantity": 1000.0, "attempt": 1, "fill_id": "101"},
            intent_id="intent-101",
            strategy_id="s",
            symbol="EURUSD",
        )
        acct.visible_upto = 101
        assert cu.catch_up() == 0
        assert journal.fill_ids() == ["101"]
        assert store.load(ACC) == 101

    def test_no_journal_is_never_durable(self):
        oms = OrderManager(_broker(_Account([])))  # type: ignore[arg-type]
        fill = {"transaction_id": "5", "client_order_id": None, "instrument": "EURUSD"}
        first = oms.process_fill(fill)
        assert first.newly_processed and not first.durable
        again = oms.process_fill(fill)
        assert not again.newly_processed and not again.durable

    def test_catch_up_disabled_without_trade_journal(self):
        from src.runtime.run_engine import build_fill_catch_up

        b = _broker(_Account([]))
        assert build_fill_catch_up(b, OrderManager(b), engine=object()) is None  # type: ignore[arg-type]
        with_journal = OrderManager(b, journal=_Journal())  # type: ignore[arg-type]
        assert build_fill_catch_up(b, with_journal, engine=object()) is not None

    def test_dedup_rollover_keeps_recent_ids(self, monkeypatch):
        import src.execution.oms as oms_mod

        monkeypatch.setattr(oms_mod, "_SEEN_FILLS_CAP", 3)
        journal = _Journal()
        oms = OrderManager(_broker(_Account([])), journal=journal)  # type: ignore[arg-type]
        for i in range(1, 6):
            oms.on_fill({"transaction_id": str(i), "client_order_id": None})
        # Overflow evicted only the oldest ids; recent overlap still deduped.
        for i in (3, 4, 5):
            assert oms.on_fill({"transaction_id": str(i), "client_order_id": None}) is False
        assert journal.fill_ids() == ["1", "2", "3", "4", "5"]


class TestSqlCheckpointStore:
    def test_migration_028_and_monotonic_upsert_on_sqlite(self):
        eng = create_engine("sqlite://")
        sql = Path("migrations/028_oanda_stream_checkpoint.sql").read_text()
        body = "\n".join(ln for ln in sql.splitlines() if not ln.strip().startswith("--"))
        with eng.begin() as conn:
            for stmt in (s.strip() for s in body.split(";")):
                if stmt:
                    conn.execute(text(stmt))
        store = SqlTransactionCheckpointStore(eng)
        assert store.load(ACC) is None
        store.advance(ACC, 10)
        store.advance(ACC, 7)  # never moves backwards
        assert store.load(ACC) == 10
        store.advance(ACC, 12)
        assert store.load(ACC) == 12
        assert store.load("other") is None


class TestEngineWiring:
    def test_transaction_task_replays_on_connect_and_checkpoints_live(self):
        from src.runtime.live_engine import LiveEngine

        calls: list[str] = []

        class _Broker:
            async def stream_transactions(self, on_connect=None):  # noqa: ANN001, ANN202
                await on_connect()
                yield {"transaction_id": "9"}

            async def stream_prices(self, symbols):  # noqa: ANN001, ANN202
                yield {}

            def get_account(self) -> None:
                return None

        class _Oms:
            def on_fill(self, fill: dict[str, Any]) -> bool:
                raise AssertionError("catch-up path must not bypass handle_live")

        class _CU:
            def catch_up(self) -> int:
                calls.append("catch_up")
                return 0

            def handle_live(self, fill: dict[str, Any]) -> None:
                calls.append("live:" + fill["transaction_id"])
                engine.running = False

        engine = LiveEngine([], _Oms(), _Broker(), fill_catch_up=_CU())  # type: ignore[arg-type]
        engine.running = True
        asyncio.run(engine._transaction_stream_task())
        assert calls == ["catch_up", "live:9"]
