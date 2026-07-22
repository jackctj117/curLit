"""FastAPI backend — REST API wired to live engine state.

Auth (CL-pu7i): every /api/* route requires the ``X-API-Key`` header set
to ``WEB_API_SECRET``. The legacy ``?secret=`` query param was REMOVED —
query strings leak into access logs, proxies, and Referer headers.

Control endpoints (CL-8lv6): halt / resume / close / manual trade return
503 when the engine runtime is not wired instead of a lying
``{"ok": true}`` no-op, reflect the OMS halted state honestly, and
``/api/system/resume`` re-arms the kill-switch once-per-day trigger dedup
when a ``KillSwitchManager`` is wired (reported as
``kill_switches_rearmed``).
"""

import hashlib
import hmac
import logging
import math
import os
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="curLit Web API", version="0.1.0")

# Global state — set by run_engine at startup
_runtime: dict[str, Any] = {
    "broker": None,
    "oms": None,
    "strategies": [],
    "kill_switch_manager": None,
}

#: Manual-trade size cap in units (CL-8lv6). ``|target_position|`` above
#: this is refused with 400. Override via ``WEB_API_MAX_TRADE_UNITS``.
_DEFAULT_MAX_TRADE_UNITS: float = 1_000_000.0

#: Anything longer than this is not an instrument symbol in any dialect
#: this repo speaks (longest real forms are like ``SPX500_USD`` = 10).
_MAX_SYMBOL_LEN: int = 20


def set_runtime(
    broker: Any, oms: Any, strategies: list[Any],
    kill_switch_manager: Any | None = None,
) -> None:
    """Wire the live engine's objects into the module-level runtime.

    ``kill_switch_manager`` (CL-8lv6): run_engine passes the engine's
    ``KillSwitchManager`` so ``/api/system/resume`` can clear the
    once-per-day trigger dedup (``reset_daily()``). ``None`` LEAVES any
    previously wired manager in place — ``LiveEngine._web_server_task``
    re-calls ``set_runtime(broker, oms, strategies)`` after run_engine
    has wired the manager, and that second call must not silently unwire
    the resume endpoint's re-arm path.
    """
    _runtime["broker"] = broker
    _runtime["oms"] = oms
    _runtime["strategies"] = strategies
    if kill_switch_manager is not None:
        _runtime["kill_switch_manager"] = kill_switch_manager


class TradeRequest(BaseModel):
    symbol: str
    target_position: float
    urgency: str = "normal"


#: Known-default secrets that must NEVER authenticate (CL-k55b). "curlit-dev"
#: was the hardcoded fallback; "change-me..." ships in .env.example.
_FORBIDDEN_SECRETS = frozenset({
    "", "curlit-dev", "change-me-to-a-random-string",
})


def _secret_is_forbidden(secret: str) -> bool:
    """True when the configured secret is a known default (CL-9dhg).

    ``CHANGE_ME*`` placeholders (see .env.example) are defaults too —
    the read-only soak dashboard already rejects them; the trade-placing
    API must not be the weaker surface.
    """
    return secret in _FORBIDDEN_SECRETS or secret.startswith("CHANGE_ME")


def verify_secret(
    x_api_key: str | None = Header(default=None),
) -> None:
    """Auth for every /api/* route (CL-k55b hardening; CL-pu7i header-only).

    * FAIL CLOSED on a missing/default secret: if WEB_API_SECRET is unset,
      one of the known defaults, or a CHANGE_ME* placeholder (CL-9dhg —
      same policy as the read-only soak dashboard), /api/* returns 503
      for EVERYONE — these
      endpoints place trades and halt the engine; a guessable default on a
      0.0.0.0 bind was a takeover path. /health stays open.
    * X-API-Key HEADER ONLY (CL-pu7i). The legacy ``?secret=`` query param
      is gone: query strings leak into access logs, proxies, and Referer
      headers. Send ``X-API-Key: <WEB_API_SECRET>``.
    * Constant-time comparison over SHA-256 digests of BOTH sides
      (hash-then-compare-digest): equal-length inputs to compare_digest
      mean a length mismatch can neither raise nor leak length via timing.
    """
    expected = os.environ.get("WEB_API_SECRET", "")
    if _secret_is_forbidden(expected):
        raise HTTPException(
            status_code=503,
            detail="WEB_API_SECRET is unset or a known default — the API "
                   "refuses to serve control endpoints until a real secret "
                   "is configured (see .env.example).",
        )
    supplied = x_api_key if x_api_key is not None else ""
    if not hmac.compare_digest(
        hashlib.sha256(supplied.encode()).digest(),
        hashlib.sha256(expected.encode()).digest(),
    ):
        raise HTTPException(status_code=403)


def _require_oms() -> Any:
    """The OMS, or 503 — control endpoints must not pretend to act
    (CL-8lv6). ``{"ok": true}`` from an unwired API was a lie that could
    convince an operator a live engine had been halted."""
    oms = _runtime.get("oms")
    if oms is None:
        raise HTTPException(
            status_code=503,
            detail="Engine runtime is not wired (no OMS) — this control "
                   "endpoint cannot act. Start the API via the live engine "
                   "(src.runtime.run_engine), not standalone.",
        )
    return oms


def _oms_halted(oms: Any) -> bool:
    return bool(getattr(oms, "_halted", True))


def _max_trade_units() -> float:
    """Manual-trade cap: WEB_API_MAX_TRADE_UNITS or the default. A
    malformed/non-positive override falls back to the (conservative)
    default LOUDLY rather than breaking manual control entirely."""
    raw = os.environ.get("WEB_API_MAX_TRADE_UNITS")
    if raw is None:
        return _DEFAULT_MAX_TRADE_UNITS
    try:
        cap = float(raw)
    except ValueError:
        cap = float("nan")
    if not math.isfinite(cap) or cap <= 0:
        logger.error(
            "WEB_API_MAX_TRADE_UNITS=%r is not a positive finite number — "
            "using the default cap of %.0f units", raw,
            _DEFAULT_MAX_TRADE_UNITS,
        )
        return _DEFAULT_MAX_TRADE_UNITS
    return cap


@app.get("/health")
def health() -> dict[str, str]:
    """Unauthenticated liveness probe (CL-oluv) — used by the Docker
    HEALTHCHECK. Deliberately reports process-up only; no broker/DB state
    and no secret required."""
    return {"status": "ok"}


@app.get("/api/positions")
def get_positions(_: None = Depends(verify_secret)) -> list[dict[str, Any]]:
    broker = _runtime.get("broker")
    if broker is None:
        return []
    positions = broker.get_positions()
    return [{"symbol": p.symbol, "quantity": p.quantity, "avg_price": p.avg_price,
             "pnl": p.unrealized_pnl} for p in positions]


@app.delete("/api/positions/{symbol}")
def close_position(symbol: str, _: None = Depends(verify_secret)) -> dict[str, Any]:
    from src.execution.broker import canonical_symbol
    oms = _require_oms()
    # Any instrument the broker can hold is closable (FX pairs AND index
    # CFDs like SPX500_USD), so this validates shape, not FX-ness.
    if len(symbol) > _MAX_SYMBOL_LEN or not canonical_symbol(symbol).isalnum():
        raise HTTPException(
            status_code=400, detail=f"invalid symbol {symbol!r}",
        )
    from src.execution.oms import OrderIntent
    intent_id = oms.submit_intent(OrderIntent(
        strategy_id="manual", symbol=symbol, target_position=0,
        urgency="urgent",
    ))
    # A close (target 0) is risk-reducing, so it passes the OMS halt gate;
    # the halted flag is still reported so the operator sees engine state.
    return {"ok": True, "symbol": symbol, "intent_id": intent_id,
            "oms_halted": _oms_halted(oms)}


@app.get("/api/signals")
def get_signals(_: None = Depends(verify_secret)) -> dict[str, Any]:
    return {"strategies": [{"id": s.id, "symbols": s.symbols} for s in _runtime.get("strategies", [])]}


@app.get("/api/pnl")
def get_pnl(_: None = Depends(verify_secret)) -> dict[str, Any]:
    broker = _runtime.get("broker")
    if broker is None:
        return {"daily": 0, "weekly": 0, "monthly": 0}
    acc = broker.get_account()
    return {"daily": 0, "equity": acc.equity}


@app.get("/api/account")
def get_account(_: None = Depends(verify_secret)) -> dict[str, Any]:
    broker = _runtime.get("broker")
    if broker is None:
        return {"equity": 0, "margin_used": 0, "drawdown_pct": 0}
    acc = broker.get_account()
    return {"equity": acc.equity, "margin_used": acc.margin_used}


@app.post("/api/trade")
def manual_trade(req: TradeRequest, _: None = Depends(verify_secret)) -> dict[str, Any]:
    """Manual FX intent (CL-8lv6 hardening): 503 when unwired; 400 on a
    non-FX symbol, an unknown urgency, or a non-finite / zero / over-cap
    size. FX-only by design — options flow through the Alpaca executor,
    not this endpoint."""
    from src.execution.broker import currency_pair
    from src.execution.oms import OrderIntent, Urgency
    oms = _require_oms()
    if currency_pair(req.symbol) is None:
        raise HTTPException(
            status_code=400,
            detail=f"symbol {req.symbol!r} is not a canonical FX pair "
                   "(expected e.g. EUR_USD or EURUSD)",
        )
    valid_urgencies = tuple(u.value for u in Urgency)
    if req.urgency not in valid_urgencies:
        raise HTTPException(
            status_code=400,
            detail=f"urgency {req.urgency!r} is not one of {valid_urgencies}",
        )
    if not math.isfinite(req.target_position) or req.target_position == 0.0:
        raise HTTPException(
            status_code=400,
            detail="target_position must be a finite, non-zero number of "
                   "units (to flatten, use DELETE /api/positions/{symbol})",
        )
    cap = _max_trade_units()
    if abs(req.target_position) > cap:
        raise HTTPException(
            status_code=400,
            detail=f"|target_position| {abs(req.target_position):.0f} "
                   f"exceeds the manual-trade cap of {cap:.0f} units "
                   "(override via WEB_API_MAX_TRADE_UNITS)",
        )
    intent_id = oms.submit_intent(OrderIntent(
        strategy_id="manual", symbol=req.symbol,
        target_position=req.target_position, urgency=req.urgency,
    ))
    halted = _oms_halted(oms)
    resp: dict[str, Any] = {
        "ok": True, "intent_id": intent_id, "symbol": req.symbol,
        "target_position": req.target_position, "oms_halted": halted,
    }
    if halted:
        resp["note"] = (
            "OMS is halted — the halt gate rejects non-risk-reducing "
            "intents, so this intent may NOT have reached the broker; "
            "resume via POST /api/system/resume first"
        )
    return resp


@app.get("/api/config")
def get_config(_: None = Depends(verify_secret)) -> dict[str, Any]:
    return {"strategies": [{"id": s.id, "config": s.config.__dict__ if hasattr(s, "config") else {}}
            for s in _runtime.get("strategies", [])]}


@app.get("/api/system")
def system_status(_: None = Depends(verify_secret)) -> dict[str, Any]:
    oms = _runtime.get("oms")
    # "engine: running" with no OMS wired was a lie (CL-8lv6) — report the
    # wiring truthfully so /api/system can be trusted during incidents.
    return {
        "engine": "running" if oms is not None else "not_wired",
        "oms_wired": oms is not None,
        "oms_halted": _oms_halted(oms) if oms is not None else True,
        "kill_switch_manager_wired": (
            _runtime.get("kill_switch_manager") is not None
        ),
    }


@app.post("/api/system/halt")
def halt_system(_: None = Depends(verify_secret)) -> dict[str, Any]:
    oms = _require_oms()
    oms.halt_new_trades()
    return {"ok": True, "action": "halt", "oms_halted": _oms_halted(oms)}


@app.post("/api/system/resume")
def resume_system(_: None = Depends(verify_secret)) -> dict[str, Any]:
    """Resume trading AND re-arm the kill switches (CL-8lv6).

    ``KillSwitchManager`` dedups each switch to one trigger per UTC day
    (``_triggered_today``); resuming without clearing that dedup left a
    fired switch unable to re-fire until day rollover — a resumed engine
    trading WITHOUT the brake that just stopped it. When the manager is
    wired via ``set_runtime`` its ``reset_daily()`` is called here;
    ``kill_switches_rearmed`` in the response tells the operator which
    world they are in.
    """
    oms = _require_oms()
    oms.resume_trades()
    manager = _runtime.get("kill_switch_manager")
    rearmed = False
    if manager is not None:
        manager.reset_daily()
        rearmed = True
        logger.warning(
            "/api/system/resume: OMS resumed and kill-switch daily dedup "
            "cleared — all switches may fire again today",
        )
    else:
        logger.warning(
            "/api/system/resume: OMS resumed but NO kill_switch_manager is "
            "wired — the once-per-day trigger dedup was NOT cleared; an "
            "already-fired switch cannot re-fire until UTC day rollover",
        )
    resp: dict[str, Any] = {
        "ok": True, "action": "resume", "oms_halted": _oms_halted(oms),
        "kill_switches_rearmed": rearmed,
    }
    if not rearmed:
        resp["note"] = (
            "kill_switch_manager is not wired — a switch that already "
            "fired today stays deduped until UTC day rollover or engine "
            "restart"
        )
    return resp
