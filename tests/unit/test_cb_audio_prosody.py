"""Unit tests for the prosody feature functions in scripts/analyze_cb_audio.py
(CL-68e).

All audio is synthetic (numpy sine waves) — no downloads, no whisper, no
network. What's worth pinning down: pitch recovery on a known tone, pause
detection on constructed gaps, and the speaking-rate arithmetic including
its zero-duration edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest
from scripts.analyze_cb_audio import (
    extract_energy_features,
    extract_pause_features,
    extract_pitch_features,
    feature_market_correlations,
    head_bytes_needed,
    speaking_rate_features,
)

SR = 16_000


def _tone(freq_hz: float, seconds: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


@pytest.mark.unit
class TestPitchFeatures:
    def test_recovers_sine_frequency(self) -> None:
        y = _tone(220.0, 2.0)
        feats = extract_pitch_features(y, SR)
        assert feats["pitch_mean_hz"] == pytest.approx(220.0, abs=10.0)
        assert feats["pitch_std_hz"] < 10.0  # constant tone → near-zero spread
        assert feats["voiced_fraction"] > 0.5

    def test_silence_yields_nan_pitch(self) -> None:
        feats = extract_pitch_features(_silence(1.0), SR)
        assert np.isnan(feats["pitch_mean_hz"])
        assert feats["voiced_fraction"] == 0.0


@pytest.mark.unit
class TestPauseFeatures:
    def test_counts_constructed_gaps(self) -> None:
        # tone(1s) gap(0.5s) tone(1s) gap(0.5s) tone(1s) → 2 pauses ≈ 0.5 s
        y = np.concatenate([
            _tone(180, 1.0), _silence(0.5),
            _tone(180, 1.0), _silence(0.5),
            _tone(180, 1.0),
        ])
        feats = extract_pause_features(y, SR, top_db=30.0, min_pause_s=0.3)
        assert feats["pause_count"] == 2.0
        assert feats["pause_mean_s"] == pytest.approx(0.5, abs=0.15)
        assert feats["span_seconds"] == pytest.approx(4.0, abs=0.2)
        assert feats["speech_seconds"] == pytest.approx(3.0, abs=0.3)
        assert 0.0 < feats["pause_fraction"] < 0.5

    def test_leading_trailing_silence_excluded(self) -> None:
        y = np.concatenate([_silence(2.0), _tone(180, 1.0), _silence(2.0)])
        feats = extract_pause_features(y, SR, top_db=30.0)
        assert feats["pause_count"] == 0.0
        assert feats["span_seconds"] == pytest.approx(1.0, abs=0.2)

    def test_gaps_below_threshold_ignored(self) -> None:
        y = np.concatenate([_tone(180, 1.0), _silence(0.1), _tone(180, 1.0)])
        feats = extract_pause_features(y, SR, top_db=30.0, min_pause_s=0.3)
        assert feats["pause_count"] == 0.0

    def test_all_silence(self) -> None:
        feats = extract_pause_features(_silence(1.0), SR)
        assert feats["pause_count"] == 0.0
        assert feats["speech_seconds"] == 0.0

    def test_adaptive_threshold_with_compressed_dynamic_range(self) -> None:
        # Broadcast-style audio: "pauses" are room tone (gaussian noise)
        # only ~11 dB below speech, where a fixed 30 dB-under-peak gate
        # finds no silence at all. The adaptive default must still find
        # both gaps.
        rng = np.random.default_rng(7)
        room = (0.1 * rng.standard_normal(int(0.5 * SR))).astype(np.float32)
        y = np.concatenate([
            _tone(180, 1.0), room,
            _tone(180, 1.0), room,
            _tone(180, 1.0),
        ])
        fixed = extract_pause_features(y, SR, top_db=30.0, min_pause_s=0.3)
        adaptive = extract_pause_features(y, SR, min_pause_s=0.3)
        assert fixed["pause_count"] == 0.0  # the failure mode being fixed
        assert adaptive["pause_count"] == 2.0
        assert adaptive["pause_mean_s"] == pytest.approx(0.5, abs=0.15)


@pytest.mark.unit
class TestSpeakingRate:
    def test_wpm_arithmetic(self) -> None:
        feats = speaking_rate_features(n_words=300, speech_seconds=120.0, span_seconds=150.0)
        assert feats["wpm_speech"] == pytest.approx(150.0)
        assert feats["wpm_total"] == pytest.approx(120.0)
        assert feats["n_words"] == 300.0

    def test_zero_duration_is_safe(self) -> None:
        feats = speaking_rate_features(n_words=10, speech_seconds=0.0, span_seconds=0.0)
        assert feats["wpm_speech"] == 0.0
        assert feats["wpm_total"] == 0.0


@pytest.mark.unit
class TestEnergyFeatures:
    def test_constant_tone_has_lower_cv_than_bursty(self) -> None:
        steady = extract_energy_features(_tone(180, 2.0), SR)
        bursty = extract_energy_features(
            np.concatenate([_tone(180, 0.5), _silence(0.5)] * 2), SR,
        )
        assert steady["rms_cv"] < bursty["rms_cv"]


@pytest.mark.unit
class TestCorrelations:
    def _row(self, wpm: float, reaction: float | None) -> dict:
        return {
            "features": {"wpm_speech": wpm},
            "market": {"reaction_pct": reaction},
        }

    def test_perfect_monotone_relationship(self) -> None:
        rows = [self._row(100, -0.1), self._row(150, 0.0), self._row(200, 0.1)]
        corr = feature_market_correlations(rows, features=("wpm_speech",))
        assert corr["wpm_speech"]["pearson_r"] == pytest.approx(1.0, abs=1e-9)
        assert corr["wpm_speech"]["spearman_r"] == pytest.approx(1.0, abs=1e-9)
        assert corr["wpm_speech"]["n"] == 3

    def test_zero_variance_feature_is_none(self) -> None:
        rows = [self._row(100, -0.1), self._row(100, 0.1)]
        corr = feature_market_correlations(rows, features=("wpm_speech",))
        assert corr["wpm_speech"]["pearson_r"] is None

    def test_missing_reactions_dropped(self) -> None:
        rows = [self._row(100, -0.1), self._row(150, None), self._row(200, 0.1)]
        corr = feature_market_correlations(rows, features=("wpm_speech",))
        assert corr["wpm_speech"]["n"] == 2


@pytest.mark.unit
def test_head_bytes_needed_scales_with_bitrate() -> None:
    # 510 kbps for 900 s → well under the full ~164 MB file but > raw payload.
    n = head_bytes_needed(510_000, 900.0)
    raw = 510_000 / 8 * 900
    assert raw < n < 164_000_000
