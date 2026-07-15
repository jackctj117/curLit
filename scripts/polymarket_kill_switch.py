"""Polymarket emergency kill switch CLI (CL-983f).

Cancels ALL open CLOB orders for the configured account and revokes the
CTFExchange USDC.e allowance (ERC-20 ``approve(CTFExchange, 0)``).

Thin shim — all logic (and the unit-test surface) lives in
``src.execution.polymarket_kill_switch``. Kept as a script so the
operator can find it next to the other runbook tools.

Usage:
  .venv/bin/python -m scripts.polymarket_kill_switch --env amoy --dry-run
  .venv/bin/python -m scripts.polymarket_kill_switch --env amoy
  .venv/bin/python -m scripts.polymarket_kill_switch --env mainnet --yes-i-mean-it

Exit codes: 0 = all steps OK, 1 = a step failed (check logs), 2 = refused
(mainnet without --yes-i-mean-it).
"""

from __future__ import annotations

from src.execution.polymarket_kill_switch import main

if __name__ == "__main__":
    raise SystemExit(main())
