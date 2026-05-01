"""FastAPI backend — REST API wired to live engine state."""

import os
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="curLit Web API", version="0.1.0")

# Global state — set by run_engine at startup
_runtime: dict[str, Any] = {"broker": None, "oms": None, "strategies": []}


def set_runtime(
    broker: Any, oms: Any, strategies: list[Any],
) -> None:
    _runtime["broker"] = broker
    _runtime["oms"] = oms
    _runtime["strategies"] = strategies


class TradeRequest(BaseModel):
    symbol: str
    target_position: float
    urgency: str = "normal"


def verify_secret(secret: str = "") -> None:
    expected = os.environ.get("WEB_API_SECRET", "curlit-dev")
    if secret != expected:
        raise HTTPException(status_code=403)


@app.get("/api/positions")
def get_positions(_: str = Depends(verify_secret)) -> list[dict[str, Any]]:
    broker = _runtime.get("broker")
    if broker is None:
        return []
    positions = broker.get_positions()
    return [{"symbol": p.symbol, "quantity": p.quantity, "avg_price": p.avg_price,
             "pnl": p.unrealized_pnl} for p in positions]


@app.delete("/api/positions/{symbol}")
def close_position(symbol: str, _: str = Depends(verify_secret)) -> dict[str, Any]:
    oms = _runtime.get("oms")
    if oms:
        from src.execution.oms import OrderIntent
        oms.submit_intent(OrderIntent(strategy_id="manual", symbol=symbol, target_position=0, urgency="urgent"))
    return {"ok": True, "symbol": symbol}


@app.get("/api/signals")
def get_signals(_: str = Depends(verify_secret)) -> dict[str, Any]:
    return {"strategies": [{"id": s.id, "symbols": s.symbols} for s in _runtime.get("strategies", [])]}


@app.get("/api/pnl")
def get_pnl(_: str = Depends(verify_secret)) -> dict[str, Any]:
    broker = _runtime.get("broker")
    if broker is None:
        return {"daily": 0, "weekly": 0, "monthly": 0}
    acc = broker.get_account()
    return {"daily": 0, "equity": acc.equity}


@app.get("/api/account")
def get_account(_: str = Depends(verify_secret)) -> dict[str, Any]:
    broker = _runtime.get("broker")
    if broker is None:
        return {"equity": 0, "margin_used": 0, "drawdown_pct": 0}
    acc = broker.get_account()
    return {"equity": acc.equity, "margin_used": acc.margin_used}


@app.post("/api/trade")
def manual_trade(req: TradeRequest, _: str = Depends(verify_secret)) -> dict[str, Any]:
    oms = _runtime.get("oms")
    if oms:
        from src.execution.oms import OrderIntent
        oms.submit_intent(OrderIntent(strategy_id="manual", symbol=req.symbol,
                          target_position=req.target_position, urgency=req.urgency))
    return {"ok": True}


@app.get("/api/config")
def get_config(_: str = Depends(verify_secret)) -> dict[str, Any]:
    return {"strategies": [{"id": s.id, "config": s.config.__dict__ if hasattr(s, "config") else {}}
            for s in _runtime.get("strategies", [])]}


@app.get("/api/system")
def system_status(_: str = Depends(verify_secret)) -> dict[str, Any]:
    oms = _runtime.get("oms")
    return {"engine": "running", "oms_halted": getattr(oms, "_halted", True) if oms else True}


@app.post("/api/system/halt")
def halt_system(_: str = Depends(verify_secret)) -> dict[str, Any]:
    oms = _runtime.get("oms")
    if oms:
        oms.halt_new_trades()
    return {"ok": True, "action": "halt"}


@app.post("/api/system/resume")
def resume_system(_: str = Depends(verify_secret)) -> dict[str, Any]:
    oms = _runtime.get("oms")
    if oms:
        oms.resume_trades()
    return {"ok": True, "action": "resume"}
