"""Config for the event-driven strategy (CL-mnhw; split out in CL-e6lx).

Holds :class:`EventDrivenConfig` (populated from
``configs/live_portfolio.yaml`` by ``run_engine.build_strategies``) and
the default assessment-id → OANDA instrument map. Pure data — no
behavior. Re-exported from :mod:`src.strategies.event_driven` so every
existing import path keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _default_instrument_map() -> dict[str, str]:
    """Assessment instrument id → OANDA broker instrument. Covers every
    tradable id in configs/event_playbooks.yaml plus compact aliases an
    LLM might emit; anything NOT in the map is skipped with a warning at
    trade time — never traded blind, never a crash."""
    return {
        # OANDA-native ids pass through unchanged.
        "EUR_USD": "EUR_USD",
        "GBP_USD": "GBP_USD",
        "USD_JPY": "USD_JPY",
        "USD_CAD": "USD_CAD",
        "USD_CHF": "USD_CHF",
        "AUD_USD": "AUD_USD",
        "NZD_USD": "NZD_USD",
        "USD_MXN": "USD_MXN",
        "USD_NOK": "USD_NOK",
        "USD_SEK": "USD_SEK",
        "USD_CNH": "USD_CNH",
        "EUR_GBP": "EUR_GBP",
        "EUR_JPY": "EUR_JPY",
        "XAU_USD": "XAU_USD",
        "XAG_USD": "XAG_USD",
        "BCO_USD": "BCO_USD",
        "WTICO_USD": "WTICO_USD",
        "NATGAS_USD": "NATGAS_USD",
        "SPX500_USD": "SPX500_USD",
        "NAS100_USD": "NAS100_USD",
        # Compact aliases → OANDA ids.
        "EURUSD": "EUR_USD",
        "GBPUSD": "GBP_USD",
        "USDJPY": "USD_JPY",
        "USDCAD": "USD_CAD",
        "USDCHF": "USD_CHF",
        "AUDUSD": "AUD_USD",
        "NZDUSD": "NZD_USD",
        "XAUUSD": "XAU_USD",
        "XAGUSD": "XAG_USD",
        "GOLD": "XAU_USD",
        "BRENT": "BCO_USD",
        "WTI": "WTICO_USD",
    }


@dataclass
class EventDrivenConfig:
    # ---- Confirmation (mirrors ConfluenceConfig; see src/events/confluence.py)
    min_urgency: int = 7
    min_confidence: float = 0.75
    confirm_window_min_minutes: int = 30
    confirm_window_max_minutes: int = 120
    confirm_move_frac: float = 0.25
    min_confirmed_instruments: int = 1
    realized_vol_window: int = 20
    vol_spike_check_enabled: bool = False
    vol_spike_window: int = 5
    vol_spike_ratio: float = 1.5
    # Stale-fact confidence ceiling (CL-ylak; mirrors ConfluenceConfig).
    # A CONFIRMED-candidate whose theme's playbook facts were last reviewed
    # more than stale_review_days ago has its LLM confidence capped at
    # stale_confidence_ceiling before Gate A — aging ownership/control facts
    # can't clear min_confidence on conviction alone. 0 days disables.
    stale_review_days: int = 90
    stale_confidence_ceiling: float = 0.58
    # ---- Trading -----------------------------------------------------
    # Risk per event trade as a fraction of equity. 0.005 = 50bps —
    # deliberately half the CB-sentiment risk; event assessments are the
    # least-proven signal in the book.
    event_risk_pct: float = 0.005
    # Hard stop distance from entry (fraction of entry price).
    event_stop_pct: float = 0.01
    # Max simultaneous event positions across ALL events.
    max_concurrent_event_positions: int = 2
    # Phantom-position reconciliation grace window (CL-v9g4): open_positions
    # entries OLDER than this that the broker doesn't hold are pruned each
    # cycle — see EventBook.reconcile for the full rationale.
    position_reconcile_grace_sec: int = 120
    # Hard TIME STOP: exit after this many hours regardless of P&L.
    event_max_holding_hours: float = 4.0
    # Per-instrument concentration cap (CL-wbmw): combined open notional in
    # ANY single event instrument as a fraction of equity. A CEILING that
    # catches ACCUMULATION, sized just ABOVE one base leg (base notional =
    # event_risk_pct / event_stop_pct = ~50% of equity, only 0.5% RISK at
    # the 1% stop): the first leg in a name passes intact, a SECOND leg in
    # the same instrument is trimmed/skipped. 0.55 = one base leg + buffer.
    per_instrument_max_pct: float = 0.55
    # Combined open notional across HAVEN_INSTRUMENTS (gold/silver) as a
    # fraction of equity (CL-5mkf) — the tighter CLUSTER cap correlated
    # metals need (per-name 0.55 alone would allow 1.10 of havens). Allows
    # one full haven leg (~0.50) plus a small second, never two full metal
    # legs of stacked gap risk. 0.60 = ~1.2 base legs of combined metals.
    haven_max_pct: float = 0.60
    # ---- Theme-primary scoping (CL-9nvq) ------------------------------
    # Machine legs may ONLY use instruments listed in the CONFIRMED event's
    # OWN theme playbook. Gold/USDJPY/indices appear in many themes, and the
    # impact agent's cross-theme whitelist (all_tradable_instruments) admits
    # any instrument known to ANY playbook — so without this gate a theme
    # that does not list gold could still open a gold leg ("gold on
    # everything"). Cross-theme instruments are demoted to advisory (skipped,
    # reason 'cross_theme'); the confirmed alert still lists them so the
    # operator can act manually. Fail-OPEN: an empty/unknown theme, or an
    # unloadable playbook config, leaves the pre-existing behavior intact
    # (no in-theme instrument set to scope against). Set False to disable.
    theme_primary_only: bool = True
    event_playbooks_path: str = "configs/event_playbooks.yaml"
    # ---- Event-book protection ----------------------------------------
    # Cumulative realized loss (fraction of equity) that freezes NEW
    # event entries. Exits always still flow.
    event_book_max_loss_pct: float = 0.02
    event_book_state_path: str = "data/event_book_state.json"
    # ---- Cross-asset corroboration (CL-6mzn) ---------------------------
    # Path to the per-theme corroborating-instruments config. The read is
    # a DISPLAY annotation on confirmed events, and — when the gate below
    # is enabled — an ENTRY gate on the machine legs.
    cross_asset_checks_path: str = "configs/cross_asset_checks.yaml"
    # ENTRY GATE (CL-6mzn, gating half): when True, a confirmed event whose
    # cross-asset read POSITIVELY FAILS (confirmed is False — contradictory)
    # has its machine legs SKIPPED (reason 'cross_asset_veto'). The event
    # still confirms/alerts and advisory ideas still flow — only the
    # auto-traded legs are blocked; fade risk isn't machine-traded.
    cross_asset_gate_enabled: bool = False
    # Stricter mode: ALSO veto when the cross-asset read is UNAVAILABLE
    # (confirmed is None — instruments unmapped / no price data). Default
    # False on purpose: this repo's history shows silent data gaps already
    # zeroed the book once (CL-5lpp/CL-gr8o); fail-closed-on-missing would
    # recreate that failure mode. Enable once cross-asset data coverage is
    # proven complete.
    cross_asset_block_on_missing: bool = False
    # ---- ASSESSED poll bounds (CL-9ts9) --------------------------------
    # The ASSESSED scan was an UNBOUNDED `SELECT ... WHERE status='ASSESSED'`
    # (see event_driven._POLL_SQL): it grew with the backlog, and rows with a
    # bad/never-confirming seen_at could linger forever, so every poll paid
    # to re-scan (and Gate-B re-price) a set that only ever grew. Bound it:
    #   * time window — only rows seen within the last N hours are polled.
    #     Anything older is already past confirm_window_max_minutes (2h by
    #     default) → it would only ever EXPIRE, so excluding it changes no
    #     tradable outcome. Default 6h leaves generous headroom above the 2h
    #     window for the once-per-run expired alert to still fire.
    #   * LIMIT — at most this many rows per poll, ordered freshest-first so
    #     the highest-urgency, most-recent (most tradable) events are always
    #     processed; if the cap elides rows it is LOGGED (never silent).
    assessed_poll_window_hours: float = 6.0
    assessed_poll_limit: int = 200
    # ---- Alerts --------------------------------------------------------
    # EXPIRED events at/above this urgency get a brief info alert.
    expired_alert_min_urgency: int = 8
    # ---- Plumbing ------------------------------------------------------
    # Event entries chase news moves — allow more slippage than the
    # 2bps default before refusing a fill.
    max_slippage_bps: float = 10.0
    signal_interval_seconds: int = 300
    instrument_map: dict[str, str] = field(default_factory=_default_instrument_map)
    id: str = "event_driven"
