"""FastAPI backend — REST API for web UI (signals, positions, PnL, config, manual trades)."""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

app = FastAPI(title="curLit Web API", version="0.1.0")


class TradeRequest(BaseModel):
    symbol: str
    target_position: float
    urgency: str = "normal"


class ConfigUpdate(BaseModel):
    value: float


# -- Auth --------------------------------------------------------------
def verify_secret(secret: str = "") -> None:
    import os
    expected = os.environ.get("WEB_API_SECRET", "curlit-dev")
    if secret != expected:
        raise HTTPException(status_code=403, detail="Forbidden")


# -- Positions ---------------------------------------------------------
@app.get("/api/positions")
def get_positions(_: str = Depends(verify_secret)) -> list[dict]:
    return []  # stub — wire to OMS


@app.delete("/api/positions/{symbol}")
def close_position(symbol: str, _: str = Depends(verify_secret)) -> dict:
    return {"ok": True, "symbol": symbol}


# -- Signals -----------------------------------------------------------
@app.get("/api/signals")
def get_signals(_: str = Depends(verify_secret)) -> dict:
    return {"strategies": []}


# -- PnL ---------------------------------------------------------------
@app.get("/api/pnl")
def get_pnl(_: str = Depends(verify_secret)) -> dict:
    return {"daily": 0, "weekly": 0, "monthly": 0}


# -- Account -----------------------------------------------------------
@app.get("/api/account")
def get_account(_: str = Depends(verify_secret)) -> dict:
    return {"equity": 0, "margin_used": 0, "drawdown_pct": 0}


# -- Manual trade -------------------------------------------------------
@app.post("/api/trade")
def manual_trade(req: TradeRequest, _: str = Depends(verify_secret)) -> dict:
    return {"ok": True, **req.dict()}


# -- Config -------------------------------------------------------------
@app.get("/api/config")
def get_config(_: str = Depends(verify_secret)) -> dict:
    return {"strategies": {}}


# -- System -------------------------------------------------------------
@app.get("/api/system")
def system_status(_: str = Depends(verify_secret)) -> dict:
    return {"engine": "ok", "services": {}, "kill_switches": []}


@app.post("/api/system/halt")
def halt_system(_: str = Depends(verify_secret)) -> dict:
    return {"ok": True, "action": "halt"}


@app.post("/api/system/resume")
def resume_system(_: str = Depends(verify_secret)) -> dict:
    return {"ok": True, "action": "resume"}
