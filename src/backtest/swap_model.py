"""FX overnight swap / rollover model with triple-swap Wednesday and broker markup."""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np


@dataclass
class SwapModelConfig:
    broker_markup_pct: float = 0.50
    weekend_roll_weekday: int = 2  # Wednesday


class SwapModel:
    def __init__(self, config: SwapModelConfig | None = None) -> None:
        self.config = config or SwapModelConfig()

    def compute_daily_swap(
        self,
        position: float,
        target_rate: float,
        base_rate: float,
        day_of_week: int,
    ) -> float:
        if day_of_week in (5, 6):
            return 0.0

        multiplier = 3 if day_of_week == self.config.weekend_roll_weekday else 1
        rate_diff = target_rate - base_rate
        daily_rate_diff = rate_diff / 360.0
        markup = self.config.broker_markup_pct / 100.0 / 360.0
        effective = daily_rate_diff - markup
        return abs(position) * effective * multiplier * np.sign(position * rate_diff)

    def simulate_swap_over_hold(
        self,
        entry_date: date,
        exit_date: date,
        position_size: float,
        target_rate: float,
        base_rate: float,
    ) -> float:
        total = 0.0
        d = entry_date
        while d < exit_date:
            total += self.compute_daily_swap(position_size, target_rate, base_rate, d.weekday())
            d += timedelta(days=1)
        return total
