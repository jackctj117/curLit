"""Regime-aware position sizing — VIX, drawdown, and correlation multipliers."""

from dataclasses import dataclass


@dataclass
class SizingAdjustment:
    base_multiplier: float
    correlation_multiplier: float
    drawdown_multiplier: float
    final_multiplier: float
    reason: str


class RegimeAwareSizer:
    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.vix_thresholds: list[tuple[float, float]] = cfg.get(
            "vix_thresholds",
            [(15.0, 1.0), (20.0, 0.8), (25.0, 0.6), (30.0, 0.4), (40.0, 0.2)],
        )
        self.dd_thresholds: list[tuple[float, float]] = cfg.get(
            "dd_thresholds",
            [(-0.05, 1.0), (-0.10, 0.75), (-0.15, 0.50), (-0.20, 0.25), (-0.25, 0.0)],
        )

    def compute_adjustment(
        self,
        vix: float,
        portfolio_dd: float,
        correlation_regime: str,
        max_pair_corr: float,
    ) -> SizingAdjustment:
        vol_mult = 1.0
        for threshold, m in self.vix_thresholds:
            if vix >= threshold:
                vol_mult = m

        dd_mult = 1.0
        for threshold, m in self.dd_thresholds:
            if portfolio_dd <= threshold:
                dd_mult = m

        if correlation_regime == "crisis":
            corr_mult = 0.4
        elif correlation_regime == "stressed":
            corr_mult = 0.7
        else:
            corr_mult = 1.0

        if max_pair_corr > 0.85:
            corr_mult *= 0.5

        final = vol_mult * dd_mult * corr_mult

        parts = []
        if vol_mult < 1.0:
            parts.append(f"VIX {vix:.1f}")
        if dd_mult < 1.0:
            parts.append(f"DD {portfolio_dd:.1%}")
        if corr_mult < 1.0:
            parts.append(f"corr {correlation_regime}")

        return SizingAdjustment(
            base_multiplier=vol_mult,
            correlation_multiplier=corr_mult,
            drawdown_multiplier=dd_mult,
            final_multiplier=final,
            reason=", ".join(parts) or "normal",
        )
