#!/usr/bin/env python3
"""FOMC press conference audio analysis prototype — Whisper + prosody (CL-68e).

Pipeline, per press conference:
  1. Discover media on federalreserve.gov. Presser pages
     (``/monetarypolicy/fomcpresconf{YYYYMMDD}.htm``) embed the Fed's own
     Brightcove player (account 66043936001). We read the player's public
     policy key from ``players.brightcove.net/.../config.json`` and query
     the Brightcove Playback API for direct progressive-MP4 URLs. This is
     the Fed's own media distribution linked from federalreserve.gov —
     no YouTube scraping involved.
  2. Range-download only the head of the lowest-bitrate MP4 rendition
     (~15 min ≈ 60-80 MB instead of ~600 MB for the full hour). Brightcove
     pMP4s are faststart (moov atom first) so a truncated file decodes.
  3. Extract 16 kHz mono WAV of the first N minutes with the bundled
     ffmpeg from ``imageio-ffmpeg`` (host has no system ffmpeg). The first
     10-15 minutes cover the chair's statement reading — that is where the
     tone signal lives; Q&A is deliberately out of scope for the prototype.
  4. Transcribe with openai-whisper (``base.en`` by default — prototype
     speed over accuracy).
  5. Prosody features: pitch stats via ``librosa.pyin``, speaking rate
     (words/min over speech time), pause frequency/duration via
     energy-based silence detection, RMS energy variability.
  6. Text sentiment baseline: project lexicon (``src.nlp.lexicon_scorer``)
     on the transcript, plus the statement-diff ``net_shift`` from
     ``cb_diff_events`` when the DB is reachable.
  7. Market reaction: EUR/USD move after the 14:00 ET statement release.
     Granularity is best-effort and recorded per presser (documented
     limitation): yfinance 5m bars only reach back ~60 days, so recent
     pressers get a true 5-minute post-statement move; older ones fall
     back to 60m bars (~730-day window), then daily close-to-close from
     the project DB / yfinance.
  8. Correlate prosody metrics with the reaction. n = 2-3 pressers is
     anecdotal — correlations are reported for direction-finding only.

Outputs: cached media + transcripts under ``data/cb_audio/`` (gitignored),
a JSON report and a short markdown summary under ``reports/``.

Usage:
  .venv/bin/python -m scripts.analyze_cb_audio \\
      [--meetings 20260617,20260429,20260318] [--minutes 15] \\
      [--whisper-model base.en] [--data-dir data/cb_audio] \\
      [--out reports/cb_audio_prosody.json] [--force]

Requires the [audio] extra: ``.venv/bin/pip install -e '.[audio]'``.
ECB pressers (ECB hosts its own webcasts) are a natural follow-up; the
discovery layer is Fed-only for now.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

logger = logging.getLogger(__name__)

# Make src.* imports work when running this script directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Fed media discovery (federalreserve.gov + its Brightcove account)
# ---------------------------------------------------------------------------

FED_BC_ACCOUNT = "66043936001"
FED_PRESSER_URL = "https://www.federalreserve.gov/monetarypolicy/fomcpresconf{meeting}.htm"
BC_PLAYER_CONFIG_URL = (
    f"https://players.brightcove.net/{FED_BC_ACCOUNT}/default_default/config.json"
)
BC_PLAYBACK_URL = (
    f"https://edge.api.brightcove.com/playback/v1/accounts/{FED_BC_ACCOUNT}/videos/{{video_id}}"
)
_UA = {"User-Agent": "curlit-research/0.1 (FOMC presser prosody prototype)"}

# Statement drops at 14:00 ET; the press conference starts at 14:30 ET.
ET = ZoneInfo("America/New_York")
STATEMENT_TIME_ET = (14, 0)
PRESSER_TIME_ET = (14, 30)


def _http_get(url: str, headers: dict[str, str] | None = None) -> Any:
    import httpx  # noqa: PLC0415

    resp = httpx.get(
        url, headers={**_UA, **(headers or {})}, follow_redirects=True, timeout=30.0,
    )
    resp.raise_for_status()
    return resp


def fed_video_id(meeting: str) -> str:
    """Brightcove video id embedded on the Fed's presser page."""
    page = _http_get(FED_PRESSER_URL.format(meeting=meeting)).text
    m = re.search(r'data-video-id="(\d+)"', page)
    if not m:
        raise RuntimeError(f"No Brightcove video id on presser page for {meeting}")
    return m.group(1)


def brightcove_policy_key() -> str:
    """Public playback policy key from the Fed's default player config."""
    cfg = _http_get(BC_PLAYER_CONFIG_URL).json()
    key = cfg.get("video_cloud", {}).get("policy_key")
    if not key:
        raise RuntimeError("No policy_key in Brightcove player config")
    return str(key)


def lowest_mp4_source(video_meta: dict[str, Any]) -> dict[str, Any]:
    """Cheapest https progressive-MP4 rendition (we only need the audio)."""
    mp4s = [
        s
        for s in video_meta.get("sources", [])
        if s.get("container") == "MP4" and str(s.get("src", "")).startswith("https://")
    ]
    if not mp4s:
        raise RuntimeError(f"No progressive MP4 source for video {video_meta.get('id')}")
    return min(mp4s, key=lambda s: s.get("avg_bitrate", 1 << 30))


def fetch_video_meta(video_id: str, policy_key: str) -> dict[str, Any]:
    resp = _http_get(
        BC_PLAYBACK_URL.format(video_id=video_id),
        headers={**_UA, "Accept": f"application/json;pk={policy_key}"},
    )
    return dict(resp.json())


def head_bytes_needed(avg_bitrate_bps: int, seconds: float) -> int:
    """Bytes to range-download so the first ``seconds`` decode.

    avg_bitrate is the muxed rate; +90 s slack and a 1.2 safety factor
    absorb VBR peaks, plus 2 MB headroom for the moov atom.
    """
    return int(avg_bitrate_bps / 8 * (seconds + 90) * 1.2) + 2_000_000


def download_media_head(url: str, dest: Path, n_bytes: int) -> None:
    """Range-GET the first ``n_bytes`` of ``url`` to ``dest``."""
    import httpx  # noqa: PLC0415

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    headers = {**_UA, "Range": f"bytes=0-{n_bytes - 1}"}
    with httpx.stream("GET", url, headers=headers, timeout=120.0, follow_redirects=True) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes(1 << 20):
                fh.write(chunk)
    tmp.rename(dest)
    logger.info("Downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def extract_audio_wav(mp4_path: Path, wav_path: Path, seconds: float) -> None:
    """First ``seconds`` of audio → 16 kHz mono WAV via bundled ffmpeg."""
    import imageio_ffmpeg  # noqa: PLC0415

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-v", "error", "-y",
        "-i", str(mp4_path),
        "-t", f"{seconds:.0f}",
        "-vn", "-ac", "1", "-ar", "16000",
        str(wav_path),
    ]
    # Truncated-but-faststart MP4s decode fine; ffmpeg may still whine on
    # stderr about the missing tail, so only raise if no output appeared.
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if not wav_path.exists() or wav_path.stat().st_size < 1000:
        raise RuntimeError(f"ffmpeg failed for {mp4_path.name}: {proc.stderr[-500:]}")


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def transcribe_wav(wav_path: Path, model_name: str) -> dict[str, Any]:
    """Whisper transcription. Audio is loaded with soundfile and passed as an
    array so whisper's own ffmpeg-subprocess loader (needs ffmpeg on PATH,
    which this host lacks) is never invoked."""
    import soundfile as sf  # noqa: PLC0415
    import whisper  # noqa: PLC0415

    y, sr = sf.read(wav_path, dtype="float32")
    if sr != 16000:  # whisper expects 16 kHz
        import librosa  # noqa: PLC0415

        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    model = whisper.load_model(model_name)
    result = model.transcribe(y, language="en", fp16=False)
    return {
        "text": result["text"].strip(),
        "segments": [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in result["segments"]
        ],
        "model": model_name,
    }


# ---------------------------------------------------------------------------
# Prosody features (pure functions — unit-tested on synthetic audio)
# ---------------------------------------------------------------------------


def extract_pitch_features(
    y: np.ndarray,
    sr: int,
    fmin: float = 60.0,
    fmax: float = 350.0,
) -> dict[str, float]:
    """F0 statistics over voiced frames via probabilistic YIN.

    Range spans typical adult speech; pitch_range is p90-p10 (robust to
    octave-error outliers), voiced_fraction is the share of frames pyin
    marks as voiced.
    """
    import librosa  # noqa: PLC0415

    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=512,
    )
    voiced = f0[np.asarray(voiced_flag, dtype=bool) & np.isfinite(f0)]
    if voiced.size == 0:
        return {
            "pitch_mean_hz": float("nan"),
            "pitch_std_hz": float("nan"),
            "pitch_range_hz": float("nan"),
            "voiced_fraction": 0.0,
        }
    return {
        "pitch_mean_hz": float(np.mean(voiced)),
        "pitch_std_hz": float(np.std(voiced)),
        "pitch_range_hz": float(np.percentile(voiced, 90) - np.percentile(voiced, 10)),
        "voiced_fraction": float(voiced.size / max(len(f0), 1)),
    }


_PAUSE_EMPTY: dict[str, float] = {
    "pause_count": 0.0,
    "pauses_per_min": 0.0,
    "pause_mean_s": 0.0,
    "pause_fraction": 0.0,
    "speech_seconds": 0.0,
    "span_seconds": 0.0,
}


def extract_pause_features(
    y: np.ndarray,
    sr: int,
    top_db: float | None = None,
    min_pause_s: float = 0.3,
) -> dict[str, float]:
    """Pause statistics from energy-based silence detection on the frame-RMS
    envelope. A pause is a silent run >= ``min_pause_s`` between the first
    and last speech frame (leading/trailing silence excluded); rates are per
    minute of that span.

    Threshold: with ``top_db`` set, silence = frames more than ``top_db``
    below peak (librosa.effects.split semantics — fine for clean/synthetic
    audio). Default (None) is adaptive: the dB midpoint between the 5th
    percentile (pause/noise floor) and the median (speech level) of frame
    RMS. Broadcast presser audio is compressed enough that a fixed 30 dB
    gate under peak finds zero silence (observed on the Fed's own
    recordings: 5th-percentile frames sit only ~18-24 dB below peak).
    """
    import librosa  # noqa: PLC0415

    if y.size == 0 or float(np.max(np.abs(y))) < 1e-6:
        return dict(_PAUSE_EMPTY)

    # Finer envelope than the pitch tracker: 64 ms windows / 16 ms hops keep
    # boundary smear well under the 300 ms minimum pause of interest.
    hop = 256
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    if top_db is not None:
        thr = float(db.max()) - top_db
    else:
        thr = 0.5 * float(np.percentile(db, 5) + np.percentile(db, 50))
    silent = db < thr

    speech_idx = np.flatnonzero(~silent)
    if speech_idx.size == 0:
        return dict(_PAUSE_EMPTY)
    first, last = int(speech_idx[0]), int(speech_idx[-1])
    hop_s = hop / sr

    pauses: list[float] = []
    run = 0
    for is_silent in silent[first : last + 1]:
        if is_silent:
            run += 1
        elif run:
            dur = run * hop_s
            if dur >= min_pause_s:
                pauses.append(dur)
            run = 0
    span_s = (last - first + 1) * hop_s
    speech_s = float((~silent[first : last + 1]).sum()) * hop_s
    return {
        "pause_count": float(len(pauses)),
        "pauses_per_min": float(len(pauses) / (span_s / 60.0)) if span_s > 0 else 0.0,
        "pause_mean_s": float(np.mean(pauses)) if pauses else 0.0,
        "pause_fraction": float(sum(pauses) / span_s) if span_s > 0 else 0.0,
        "speech_seconds": speech_s,
        "span_seconds": float(span_s),
    }


def speaking_rate_features(
    n_words: int, speech_seconds: float, span_seconds: float,
) -> dict[str, float]:
    """Words/minute over speech time (articulation rate proxy) and over the
    whole span (includes pauses)."""
    return {
        "n_words": float(n_words),
        "wpm_speech": float(n_words / (speech_seconds / 60.0)) if speech_seconds > 0 else 0.0,
        "wpm_total": float(n_words / (span_seconds / 60.0)) if span_seconds > 0 else 0.0,
    }


def extract_energy_features(y: np.ndarray, sr: int) -> dict[str, float]:
    """RMS energy variability — flat delivery vs dynamic delivery."""
    import librosa  # noqa: PLC0415

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    mean = float(np.mean(rms))
    return {"rms_cv": float(np.std(rms) / mean) if mean > 0 else 0.0}


def prosody_features(y: np.ndarray, sr: int, n_words: int) -> dict[str, float]:
    """All prosody features for one presser's audio."""
    pauses = extract_pause_features(y, sr)
    feats: dict[str, float] = {}
    feats.update(extract_pitch_features(y, sr))
    feats.update(pauses)
    feats.update(
        speaking_rate_features(n_words, pauses["speech_seconds"], pauses["span_seconds"])
    )
    feats.update(extract_energy_features(y, sr))
    return feats


# ---------------------------------------------------------------------------
# Text sentiment baseline
# ---------------------------------------------------------------------------


def transcript_sentiment(text: str) -> dict[str, float]:
    """Project lexicon scores on the transcript — the text-only baseline the
    prosody features must beat to 'add signal'."""
    from src.nlp.lexicon_scorer import LexiconScorer  # noqa: PLC0415

    s = LexiconScorer().score_text(text)
    return {
        "text_hawkish": float(s.hawkish_count),
        "text_dovish": float(s.dovish_count),
        "text_net": float(s.net_score),
        "text_uncertainty": float(s.uncertainty_count),
    }


def statement_net_shift(meeting: str) -> float | None:
    """net_shift from cb_diff_events for the written statement (None when
    the DB is unreachable or the row isn't seeded yet)."""
    try:
        from sqlalchemy import text as sql_text  # noqa: PLC0415

        from src.runtime.run_engine import _build_db_engine  # noqa: PLC0415

        engine = _build_db_engine()
        day = datetime.strptime(meeting, "%Y%m%d").date()
        with engine.connect() as conn:
            row = conn.execute(
                sql_text(
                    "SELECT net_shift FROM cb_diff_events "
                    "WHERE cb = 'fed' AND ts::date = :d LIMIT 1"
                ),
                {"d": day},
            ).fetchone()
        return float(row[0]) if row else None
    except Exception as exc:  # non-fatal: prototype must run without the DB
        logger.warning("cb_diff_events lookup failed for %s: %s", meeting, exc)
        return None


# ---------------------------------------------------------------------------
# Market reaction — EUR/USD around the 14:00 ET statement
# ---------------------------------------------------------------------------


def _yf_close_series(ticker: str, start: str, end: str, interval: str) -> Any:
    """Close series from yfinance; empty results are retried a couple of
    times because yfinance intermittently returns nothing under repeated
    calls (observed run-to-run flakiness on the same query)."""
    import pandas as pd  # noqa: PLC0415
    import yfinance as yf  # noqa: PLC0415
    from tenacity import (  # noqa: PLC0415
        retry,
        retry_if_result,
        stop_after_attempt,
        wait_fixed,
    )

    @retry(
        retry=retry_if_result(lambda df: df is None or df.empty),
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry_error_callback=lambda state: state.outcome.result() if state.outcome else None,
    )
    def _download() -> Any:
        return yf.download(
            ticker, start=start, end=end, interval=interval,
            progress=False, auto_adjust=False,
        )

    df = _download()
    if df is None or df.empty:
        return None
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance MultiIndex for single ticker
        close = close.iloc[:, 0]
    return close.dropna()


def _bar_close_at(
    close: Any, label: datetime, bar: timedelta,
) -> float | None:
    """Close of the bar labeled ``label`` (bar start), asof-fallback
    bounded to one bar interval — a fallback that reaches further back
    than ``bar`` silently substitutes stale prices, so return None
    instead and let the caller drop the granularity."""
    import pandas as pd  # noqa: PLC0415

    ts = pd.Timestamp(label)
    if ts in close.index:
        return float(close.loc[ts])
    prior = close.loc[:ts]
    if not len(prior):
        return None
    if (ts - prior.index[-1]) > pd.Timedelta(bar):
        return None
    return float(prior.iloc[-1])


def eurusd_reaction(meeting: str) -> dict[str, Any]:
    """EUR/USD move after the statement, at the finest granularity available.

    Preference order (documented limitation — yfinance intraday history is
    capped at ~60 days for 5m and ~730 days for 60m):
      * 5m  : close(13:55-14:00 ET bar) → close(14:00-14:05 ET bar)
      * 60m : close(13:00-14:00 ET bar) → close(14:00-15:00 ET bar)
      * 1d  : prior-day close → meeting-day close (project DB, then yfinance)
    Also reports a 30-min presser-window move (14:30→15:00 ET) when 5m bars
    exist, since the tone being scored is spoken in the presser itself.
    """
    day = datetime.strptime(meeting, "%Y%m%d").replace(tzinfo=ET)
    stmt = day.replace(hour=STATEMENT_TIME_ET[0], minute=STATEMENT_TIME_ET[1])
    presser = day.replace(hour=PRESSER_TIME_ET[0], minute=PRESSER_TIME_ET[1])
    start = (day - timedelta(days=4)).strftime("%Y-%m-%d")
    end = (day + timedelta(days=1)).strftime("%Y-%m-%d")
    out: dict[str, Any] = {"meeting": meeting}

    # -- 5m bars ------------------------------------------------------------
    try:
        close = _yf_close_series("EURUSD=X", start, end, "5m")
    except Exception as exc:
        logger.info("yfinance 5m unavailable for %s: %s", meeting, exc)
        close = None
    if close is not None and len(close):
        pre = _bar_close_at(close, stmt - timedelta(minutes=5), timedelta(minutes=5))
        post = _bar_close_at(close, stmt, timedelta(minutes=5))
        if pre and post:
            out.update(
                reaction_pct=(post / pre - 1.0) * 100.0,
                reaction_window="5m_post_statement",
                pre_px=pre, post_px=post,
            )
            p_pre = _bar_close_at(close, presser - timedelta(minutes=5), timedelta(minutes=5))
            p_post = _bar_close_at(close, presser + timedelta(minutes=25), timedelta(minutes=5))
            if p_pre and p_post:
                out["presser_30m_pct"] = (p_post / p_pre - 1.0) * 100.0
            return out

    # -- 60m bars -----------------------------------------------------------
    try:
        close = _yf_close_series("EURUSD=X", start, end, "60m")
    except Exception as exc:
        logger.info("yfinance 60m unavailable for %s: %s", meeting, exc)
        close = None
    if close is not None and len(close):
        pre = _bar_close_at(close, stmt - timedelta(hours=1), timedelta(hours=1))
        post = _bar_close_at(close, stmt, timedelta(hours=1))
        if pre and post:
            out.update(
                reaction_pct=(post / pre - 1.0) * 100.0,
                reaction_window="60m_post_statement",
                pre_px=pre, post_px=post,
            )
            return out

    # -- daily close-to-close (project DB, then yfinance) --------------------
    try:
        from sqlalchemy import text as sql_text  # noqa: PLC0415

        from src.runtime.run_engine import _build_db_engine  # noqa: PLC0415

        engine = _build_db_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                sql_text(
                    "SELECT ts, close FROM prices WHERE symbol = 'EURUSD' "
                    "AND ts <= :d ORDER BY ts DESC LIMIT 2"
                ),
                {"d": day.date() + timedelta(days=1)},
            ).fetchall()
        if len(rows) == 2 and rows[0][0].date() >= day.date():
            out.update(
                reaction_pct=(float(rows[0][1]) / float(rows[1][1]) - 1.0) * 100.0,
                reaction_window="1d_close_to_close_db",
                pre_px=float(rows[1][1]), post_px=float(rows[0][1]),
            )
            return out
    except Exception as exc:
        logger.info("DB daily fallback failed for %s: %s", meeting, exc)

    close = _yf_close_series("EURUSD=X", start, end, "1d")
    if close is not None and len(close) >= 2:
        out.update(
            reaction_pct=(float(close.iloc[-1]) / float(close.iloc[-2]) - 1.0) * 100.0,
            reaction_window="1d_close_to_close_yf",
            pre_px=float(close.iloc[-2]), post_px=float(close.iloc[-1]),
        )
        return out

    out.update(reaction_pct=None, reaction_window="unavailable")
    return out


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

CORR_FEATURES: tuple[str, ...] = (
    "pitch_mean_hz", "pitch_std_hz", "pitch_range_hz",
    "pauses_per_min", "pause_mean_s", "pause_fraction",
    "wpm_speech", "wpm_total", "rms_cv",
    "text_net",
)


def feature_market_correlations(
    rows: list[dict[str, Any]], features: tuple[str, ...] = CORR_FEATURES,
) -> dict[str, dict[str, float | None]]:
    """Pearson + Spearman of each feature vs the EUR/USD reaction.

    With n = 2-3 these are direction-finding numbers only (n=2 is a sign,
    n=3 has one degree of freedom); no p-values are reported on purpose.
    """
    from scipy import stats  # noqa: PLC0415

    out: dict[str, dict[str, float | None]] = {}
    y = np.array(
        [r["market"]["reaction_pct"] for r in rows if r["market"]["reaction_pct"] is not None],
        dtype=float,
    )
    for feat in features:
        x = np.array(
            [
                r["features"].get(feat, float("nan"))
                for r in rows
                if r["market"]["reaction_pct"] is not None
            ],
            dtype=float,
        )
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
            out[feat] = {"pearson_r": None, "spearman_r": None, "n": int(ok.sum())}
            continue
        out[feat] = {
            "pearson_r": float(stats.pearsonr(x[ok], y[ok])[0]),
            "spearman_r": float(stats.spearmanr(x[ok], y[ok])[0]),
            "n": int(ok.sum()),
        }
    return out


# ---------------------------------------------------------------------------
# Per-presser pipeline + reporting
# ---------------------------------------------------------------------------


def analyze_presser(
    meeting: str,
    data_dir: Path,
    minutes: float,
    whisper_model: str,
    policy_key: str,
    force: bool = False,
) -> dict[str, Any]:
    import soundfile as sf  # noqa: PLC0415

    seconds = minutes * 60.0
    mp4_path = data_dir / f"FOMCpresconf{meeting}.head.mp4"
    wav_path = data_dir / f"FOMCpresconf{meeting}.{minutes:g}min.wav"
    txt_path = data_dir / f"FOMCpresconf{meeting}.transcript.{whisper_model}.json"

    meta_name = None
    if force or not wav_path.exists():
        video_id = fed_video_id(meeting)
        meta = fetch_video_meta(video_id, policy_key)
        meta_name = meta.get("name")
        src = lowest_mp4_source(meta)
        logger.info(
            "%s: '%s' (%.1f min total), MP4 %s kbps",
            meeting, meta_name, meta.get("duration", 0) / 60000.0,
            src.get("avg_bitrate", 0) // 1000,
        )
        if force or not mp4_path.exists():
            n_bytes = head_bytes_needed(int(src.get("avg_bitrate", 2_000_000)), seconds)
            download_media_head(str(src["src"]), mp4_path, n_bytes)
        extract_audio_wav(mp4_path, wav_path, seconds)

    if force or not txt_path.exists():
        logger.info("%s: transcribing %s with whisper '%s'", meeting, wav_path.name, whisper_model)
        transcript = transcribe_wav(wav_path, whisper_model)
        txt_path.write_text(json.dumps(transcript, indent=2))
    else:
        transcript = json.loads(txt_path.read_text())
        logger.info("%s: reusing cached transcript %s", meeting, txt_path.name)

    y, sr = sf.read(wav_path, dtype="float32")
    n_words = len(transcript["text"].split())
    logger.info("%s: extracting prosody (%.1f min audio, %d words)", meeting, minutes, n_words)
    feats = prosody_features(y, int(sr), n_words)
    feats.update(transcript_sentiment(transcript["text"]))

    market = eurusd_reaction(meeting)
    logger.info(
        "%s: EURUSD %s = %s", meeting, market.get("reaction_window"),
        f"{market['reaction_pct']:+.3f}%" if market.get("reaction_pct") is not None else "n/a",
    )

    return {
        "meeting": meeting,
        "video_name": meta_name,
        "analyzed_minutes": minutes,
        "transcript_words": n_words,
        "transcript_head": transcript["text"][:300],
        "features": feats,
        "statement_net_shift_db": statement_net_shift(meeting),
        "market": market,
    }


def write_markdown_summary(report: dict[str, Any], path: Path) -> None:
    rows = report["pressers"]
    corr = report["correlations"]
    lines = [
        "# FOMC presser audio prosody vs EUR/USD reaction (CL-68e prototype)",
        "",
        f"Generated {report['ran_at']} — whisper `{report['config']['whisper_model']}`, "
        f"first {report['config']['minutes']} min of each presser "
        "(statement reading; Q&A out of scope).",
        "",
        "## Per-presser features",
        "",
        "| meeting | words | wpm (speech) | pitch mean Hz | pitch std Hz | pauses/min "
        "| pause frac | text net | stmt net_shift (DB) | EURUSD reaction | window |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        f, m = r["features"], r["market"]
        rx = f"{m['reaction_pct']:+.3f}%" if m.get("reaction_pct") is not None else "n/a"
        shift = r["statement_net_shift_db"]
        lines.append(
            f"| {r['meeting']} | {r['transcript_words']} | {f['wpm_speech']:.0f} "
            f"| {f['pitch_mean_hz']:.1f} | {f['pitch_std_hz']:.1f} "
            f"| {f['pauses_per_min']:.1f} | {f['pause_fraction']:.2f} "
            f"| {f['text_net']:+.2f} | {shift if shift is not None else 'n/a'} "
            f"| {rx} | {m.get('reaction_window')} |"
        )
    lines += [
        "",
        "## Feature ↔ reaction correlations",
        "",
        "| feature | pearson r | spearman r | n |",
        "|---|---|---|---|",
    ]
    for feat, c in corr.items():
        pr = f"{c['pearson_r']:+.2f}" if c["pearson_r"] is not None else "n/a"
        sr_ = f"{c['spearman_r']:+.2f}" if c["spearman_r"] is not None else "n/a"
        lines.append(f"| {feat} | {pr} | {sr_} | {c['n']} |")
    lines += ["", "## Read", "", report["verdict"], ""]
    path.write_text("\n".join(lines))
    logger.info("Markdown summary written to %s", path)


def _verdict(report_rows: list[dict[str, Any]], corr: dict[str, Any]) -> str:
    n = sum(1 for r in report_rows if r["market"].get("reaction_pct") is not None)
    text_r = corr.get("text_net", {}).get("pearson_r")
    prosody = {
        k: v["pearson_r"]
        for k, v in corr.items()
        if k != "text_net" and v.get("pearson_r") is not None
    }
    if not prosody or n < 2:
        return (
            f"Not enough usable pressers (n={n}) to compare prosody against the "
            "text baseline. Pipeline is proven; add more meetings before drawing "
            "any conclusion."
        )
    best_feat, best_r = max(prosody.items(), key=lambda kv: abs(kv[1]))
    text_part = (
        f"text-sentiment baseline |r|={abs(text_r):.2f}"
        if text_r is not None
        else "text-sentiment baseline correlation undefined (zero variance across sample)"
    )
    stronger = text_r is None or abs(best_r) > abs(text_r)
    return (
        f"On this n={n} sample the strongest prosody correlate of the EUR/USD "
        f"post-statement move is `{best_feat}` (r={best_r:+.2f}) vs {text_part}. "
        + (
            "Tone features look additive to text sentiment here"
            if stronger
            else "Tone features do NOT beat the text baseline here"
        )
        + " — but n=2-3 is anecdotal: one degree of freedom, no p-values, and the "
        "reaction windows mix granularities (5m vs 60m). Treat this strictly as a "
        "pipeline proof and a hypothesis to test on 20+ pressers, not as signal."
    )


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415

    load_project_env()

    p = argparse.ArgumentParser(
        description="FOMC presser audio → whisper transcript → prosody vs EUR/USD reaction.",
    )
    p.add_argument(
        "--meetings", type=str, default="20260617,20260429,20260318",
        help="Comma-separated FOMC meeting dates (YYYYMMDD) with press conferences",
    )
    p.add_argument(
        "--minutes", type=float, default=15.0,
        help="Minutes of audio to analyze from the start (statement reading)",
    )
    p.add_argument("--whisper-model", type=str, default="base.en")
    p.add_argument("--data-dir", type=Path, default=Path("data/cb_audio"))
    p.add_argument("--out", type=Path, default=Path("reports/cb_audio_prosody.json"))
    p.add_argument(
        "--summary-md", type=Path, default=Path("reports/cb_audio_prosody.md"),
    )
    p.add_argument("--force", action="store_true", help="Redo downloads/transcripts")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    meetings = [m.strip() for m in args.meetings.split(",") if m.strip()]
    args.data_dir.mkdir(parents=True, exist_ok=True)

    policy_key = brightcove_policy_key()
    rows: list[dict[str, Any]] = []
    for meeting in meetings:
        try:
            rows.append(
                analyze_presser(
                    meeting, args.data_dir, args.minutes,
                    args.whisper_model, policy_key, force=args.force,
                )
            )
        except Exception:
            logger.exception("Presser %s failed — continuing with the rest", meeting)

    if not rows:
        logger.error("No pressers analyzed successfully")
        return 1

    corr = feature_market_correlations(rows)
    report = {
        "ran_at": datetime.now(tz=ZoneInfo("UTC")).isoformat(),
        "config": {
            "meetings": meetings,
            "minutes": args.minutes,
            "whisper_model": args.whisper_model,
        },
        "pressers": rows,
        "correlations": corr,
        "verdict": _verdict(rows, corr),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Report written to %s", args.out)
    write_markdown_summary(report, args.summary_md)

    print()
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
