"""Control-endpoint honesty + manual-trade validation (CL-8lv6).

The control plane must never lie: 503 when the engine runtime is not
wired (the old code returned ``{"ok": true}`` no-ops), halted state
reflected in responses, /api/system/resume re-arms the kill-switch daily
dedup when a manager is wired and SAYS SO either way, and /api/trade
rejects garbage symbols/sizes with 400.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.web.api as api

client = TestClient(api.app)

SECRET = "a-real-secret-value"
AUTH = {"X-API-Key": SECRET}


class FakeOMS:
    def __init__(self) -> None:
        self._halted = False
        self.intents: list = []

    def submit_intent(self, intent, **kwargs) -> str:
        self.intents.append(intent)
        return intent.intent_id

    def halt_new_trades(self) -> None:
        self._halted = True

    def resume_trades(self) -> None:
        self._halted = False


class FakeKSM:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset_daily(self) -> None:
        self.reset_calls += 1


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setenv("WEB_API_SECRET", SECRET)


@pytest.fixture
def runtime():
    """Snapshot/restore the module-level runtime around each test."""
    saved = dict(api._runtime)
    yield api._runtime
    api._runtime.clear()
    api._runtime.update(saved)


@pytest.fixture
def unwired(runtime):
    runtime.update(
        {"broker": None, "oms": None, "strategies": [],
         "kill_switch_manager": None},
    )
    return runtime


@pytest.fixture
def oms(unwired):
    fake = FakeOMS()
    unwired["oms"] = fake
    return fake


# ---------------------------------------------------------------- 503s

@pytest.mark.parametrize("method,path", [
    ("post", "/api/system/halt"),
    ("post", "/api/system/resume"),
    ("delete", "/api/positions/EUR_USD"),
])
def test_control_endpoints_503_when_unwired(unwired, method, path):
    r = getattr(client, method)(path, headers=AUTH)
    assert r.status_code == 503
    assert "not wired" in r.json()["detail"]


def test_trade_503_when_unwired(unwired):
    r = client.post("/api/trade", headers=AUTH,
                    json={"symbol": "EUR_USD", "target_position": 100})
    assert r.status_code == 503
    assert "not wired" in r.json()["detail"]


# ------------------------------------------------------- halt / resume

def test_halt_reports_halted_state(oms):
    r = client.post("/api/system/halt", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"ok": True, "action": "halt", "oms_halted": True}
    assert oms._halted is True


def test_resume_rearms_kill_switches_when_wired(oms, runtime):
    ksm = FakeKSM()
    runtime["kill_switch_manager"] = ksm
    oms._halted = True
    r = client.post("/api/system/resume", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["kill_switches_rearmed"] is True
    assert body["oms_halted"] is False
    assert ksm.reset_calls == 1


def test_resume_reports_unarmed_dedup_when_no_manager(oms):
    oms._halted = True
    r = client.post("/api/system/resume", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["kill_switches_rearmed"] is False
    assert "not wired" in body["note"]
    assert oms._halted is False  # OMS still resumed


def test_set_runtime_none_preserves_kill_switch_manager(runtime):
    """LiveEngine re-calls set_runtime without the manager — that must not
    unwire the resume endpoint's re-arm path."""
    ksm = FakeKSM()
    api.set_runtime("broker", "oms", [], kill_switch_manager=ksm)
    api.set_runtime("broker2", "oms2", [])
    assert api._runtime["kill_switch_manager"] is ksm
    assert api._runtime["oms"] == "oms2"


# ------------------------------------------------------- system status

def test_system_status_honest_when_unwired(unwired):
    r = client.get("/api/system", headers=AUTH)
    body = r.json()
    assert body["engine"] == "not_wired"
    assert body["oms_wired"] is False
    assert body["oms_halted"] is True
    assert body["kill_switch_manager_wired"] is False


def test_system_status_wired(oms):
    body = client.get("/api/system", headers=AUTH).json()
    assert body["engine"] == "running"
    assert body["oms_wired"] is True
    assert body["oms_halted"] is False


# ----------------------------------------------------------- close

def test_close_position_submits_flatten_intent(oms):
    r = client.delete("/api/positions/EUR_USD", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["oms_halted"] is False
    (intent,) = oms.intents
    assert intent.symbol == "EUR_USD"
    assert intent.target_position == 0
    assert intent.urgency == "urgent"


def test_close_position_rejects_garbage_symbol(oms):
    r = client.delete("/api/positions/EUR;DROP", headers=AUTH)
    assert r.status_code == 400
    assert oms.intents == []


def test_close_position_allows_non_fx_instruments(oms):
    # Index CFD event legs (e.g. SPX500_USD) must stay closable.
    r = client.delete("/api/positions/SPX500_USD", headers=AUTH)
    assert r.status_code == 200


# ---------------------------------------------------- trade validation

def _trade(payload):
    return client.post("/api/trade", headers=AUTH, json=payload)


def test_trade_valid_fx_submits_intent(oms):
    r = _trade({"symbol": "EUR_USD", "target_position": 1000})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["oms_halted"] is False
    assert "note" not in body
    (intent,) = oms.intents
    assert intent.symbol == "EUR_USD"
    assert intent.target_position == 1000
    assert intent.strategy_id == "manual"


@pytest.mark.parametrize("symbol", [
    "DOGE", "SPX500_USD", "EURUSDX", "EUR", "..", "EUR_US1",
])
def test_trade_rejects_non_fx_symbols(oms, symbol):
    r = _trade({"symbol": symbol, "target_position": 100})
    assert r.status_code == 400
    assert "FX pair" in r.json()["detail"]
    assert oms.intents == []


def test_trade_rejects_zero_size(oms):
    r = _trade({"symbol": "EUR_USD", "target_position": 0})
    assert r.status_code == 400
    assert oms.intents == []


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_trade_rejects_non_finite_size(oms, bad):
    # Raw body: Python's json.loads (used server-side) ACCEPTS these
    # non-standard tokens, and pydantic floats allow inf/nan by default —
    # exactly the hole the endpoint's isfinite check closes.
    r = client.post(
        "/api/trade", headers={**AUTH, "Content-Type": "application/json"},
        content=f'{{"symbol": "EUR_USD", "target_position": {bad}}}',
    )
    assert r.status_code == 400
    assert oms.intents == []


def test_trade_rejects_over_default_cap(oms):
    r = _trade({"symbol": "EUR_USD", "target_position": 1_000_001})
    assert r.status_code == 400
    assert "cap" in r.json()["detail"]
    assert oms.intents == []
    assert _trade(
        {"symbol": "EUR_USD", "target_position": -999_999},
    ).status_code == 200


def test_trade_cap_env_override(oms, monkeypatch):
    monkeypatch.setenv("WEB_API_MAX_TRADE_UNITS", "100")
    assert _trade(
        {"symbol": "EUR_USD", "target_position": 150},
    ).status_code == 400
    assert _trade(
        {"symbol": "EUR_USD", "target_position": 50},
    ).status_code == 200


def test_trade_cap_garbage_env_falls_back_to_default(oms, monkeypatch):
    monkeypatch.setenv("WEB_API_MAX_TRADE_UNITS", "lots")
    assert _trade(
        {"symbol": "EUR_USD", "target_position": 1_000_001},
    ).status_code == 400
    assert _trade(
        {"symbol": "EUR_USD", "target_position": 500},
    ).status_code == 200


def test_trade_rejects_unknown_urgency(oms):
    r = _trade({"symbol": "EUR_USD", "target_position": 100,
                "urgency": "high"})
    assert r.status_code == 400
    assert oms.intents == []


def test_trade_while_halted_warns(oms):
    oms._halted = True
    r = _trade({"symbol": "EUR_USD", "target_position": 100})
    assert r.status_code == 200
    body = r.json()
    assert body["oms_halted"] is True
    assert "halted" in body["note"]
