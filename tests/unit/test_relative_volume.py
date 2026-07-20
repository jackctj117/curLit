"""Unit tests for the relative-volume scanner (CL-i4sr).

Covers: RVOL math (20d baseline, tail-window discipline), the unusual
threshold + thin-OTC min-volume floor, env overrides, per-ticker
failure tolerance (batch miss → per-ticker retry → skip), universe
extraction from a fixture playbook (equity_watch only, slug/junk
skipped), persistence of ALL scanned rows to volume_spikes (sqlite),
digest Watch-line annotation (spike present / absent / table missing),
scripts/event_pipeline.py --scan wiring, and a stubbed-airflow import
of the event_ingestion DAG. No live yfinance, DB, or Telegram calls.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from sqlalchemy import create_engine, text

from src.events.digest import build_digest, fetch_volume_marks, send_digest
from src.events.impact_agent import AssessmentResult
from src.research.notifications import DispatchResult
from src.scanners.relative_volume import (
    DEFAULT_MIN_AVG_VOLUME,
    DEFAULT_RVOL_THRESHOLD,
    RelativeVolumeScanner,
    equity_watch_universe,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture
def playbooks_yaml(tmp_path: Path) -> Path:
    """Fixture playbook: equity_watch across themes (with a dup), plus
    tradables, a Polymarket slug, and a junk lowercase equity entry."""

    def _inst(instrument: str, kind: str, direction: str) -> dict[str, str]:
        return {
            "instrument": instrument, "kind": kind,
            "direction": direction, "rationale": "r",
        }

    doc = {
        "themes": {
            "theme_a": {
                "name": "Theme A",
                "description": "d",
                "watch_terms": ["hormuz"],
                "instruments": [
                    _inst("BCO_USD", "oanda", "long"),
                    _inst("FRO", "equity_watch", "watch"),
                    _inst("STNG", "equity_watch", "watch"),
                    _inst("some-market-slug-2026", "polymarket", "watch"),
                ],
            },
            "theme_b": {
                "name": "Theme B",
                "description": "d",
                "watch_terms": ["opec"],
                "instruments": [
                    _inst("FRO", "equity_watch", "watch"),  # dup across themes
                    _inst("XOM", "equity_watch", "watch"),
                    _inst("BRK-B", "equity_watch", "watch"),
                    # junk that drifted into the config — must be skipped
                    _inst("lowercase-junk", "equity_watch", "watch"),
                ],
            },
        },
    }
    p = tmp_path / "playbooks.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


def _shim_pg_types_for_sqlite(sql: str) -> str:
    return (
        sql.replace("TIMESTAMPTZ", "TEXT")
        .replace("BIGSERIAL", "INTEGER")
        .replace("DOUBLE PRECISION", "REAL")
    )


@pytest.fixture
def sqlite_db_url(tmp_path: Path) -> str:
    """Sqlite DB with the volume_spikes migration applied (shimmed)."""
    from migrations.run import _strip_sql_comments

    db_url = f"sqlite:///{tmp_path / 'spikes.db'}"
    engine = create_engine(db_url)
    sql = _shim_pg_types_for_sqlite(_strip_sql_comments(
        (REPO_ROOT / "migrations/006_volume_spikes.sql").read_text(),
    ))
    with engine.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))
    return db_url


def _bars(
    volumes: list[float], closes: list[float] | None = None,
) -> pd.DataFrame:
    n = len(volumes)
    idx = pd.bdate_range(end="2026-07-17", periods=n)
    closes = closes if closes is not None else [10.0] * n
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes,
         "Close": closes, "Volume": volumes},
        index=idx,
    )


def _batch(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """(ticker, field) MultiIndex frame like yf.download(group_by='ticker')."""
    return pd.concat(frames, axis=1)


def _downloader(frames: dict[str, pd.DataFrame]) -> Any:
    def download(tickers: Any, _start: Any, _end: Any) -> pd.DataFrame:
        present = {t: frames[t] for t in tickers if t in frames}
        if not present:
            return pd.DataFrame()
        return _batch(present)
    return download


def _scanner(
    db_url: str, playbooks: Path,
    frames: dict[str, pd.DataFrame] | None = None,
    **kwargs: Any,
) -> RelativeVolumeScanner:
    return RelativeVolumeScanner(
        db_url, playbooks_path=playbooks,
        downloader=_downloader(frames or {}), **kwargs,
    )


# 20 prior sessions at 100k + today at 300k → rvol 3.0
_SPIKE = _bars([100_000.0] * 20 + [300_000.0],
               closes=[10.0] * 20 + [11.0])
_QUIET = _bars([100_000.0] * 20 + [110_000.0])


# --------------------------------------------------------------------- #
# Universe extraction
# --------------------------------------------------------------------- #


class TestUniverse:
    def test_union_across_themes_deduped_sorted(self, playbooks_yaml: Path) -> None:
        from src.events.playbooks import load_playbooks

        universe = equity_watch_universe(load_playbooks(playbooks_yaml))
        assert universe == ("BRK-B", "FRO", "STNG", "XOM")

    def test_skips_tradables_slugs_and_junk(self, playbooks_yaml: Path) -> None:
        from src.events.playbooks import load_playbooks

        universe = set(equity_watch_universe(load_playbooks(playbooks_yaml)))
        assert "BCO_USD" not in universe          # tradable, not equity
        assert "some-market-slug-2026" not in universe  # polymarket slug
        assert "lowercase-junk" not in universe   # non-ticker junk

    def test_scanner_universe_rereads_config(
        self, playbooks_yaml: Path, sqlite_db_url: str,
    ) -> None:
        scanner = _scanner(sqlite_db_url, playbooks_yaml)
        assert scanner.universe() == ("BRK-B", "FRO", "STNG", "XOM")


# --------------------------------------------------------------------- #
# RVOL math
# --------------------------------------------------------------------- #


class TestRvolMath:
    def _rows(self, scanner: RelativeVolumeScanner) -> dict[str, Any]:
        return {r.ticker: r for r in scanner.scan(persist=False)}

    def test_rvol_and_price_change(
        self, playbooks_yaml: Path, sqlite_db_url: str,
    ) -> None:
        frames = {t: _SPIKE for t in ("BRK-B", "FRO", "STNG", "XOM")}
        rows = self._rows(_scanner(sqlite_db_url, playbooks_yaml, frames))
        fro = rows["FRO"]
        assert fro.rvol == pytest.approx(3.0)
        assert fro.volume == 300_000
        assert fro.avg_volume_20d == pytest.approx(100_000.0)
        assert fro.price_change_pct == pytest.approx(10.0)

    def test_baseline_uses_only_last_20_prior_sessions(
        self, playbooks_yaml: Path, sqlite_db_url: str,
    ) -> None:
        # 9 ancient huge-volume sessions must NOT inflate the baseline.
        bars = _bars([9_000_000.0] * 9 + [100_000.0] * 20 + [300_000.0])
        rows = self._rows(_scanner(
            sqlite_db_url, playbooks_yaml, {"FRO": bars},
        ))
        assert rows["FRO"].rvol == pytest.approx(3.0)

    def test_too_little_history_skipped(
        self, playbooks_yaml: Path, sqlite_db_url: str,
    ) -> None:
        rows = self._rows(_scanner(
            sqlite_db_url, playbooks_yaml,
            {"FRO": _bars([100_000.0] * 3 + [300_000.0])},
        ))
        assert "FRO" not in rows

    def test_zero_volume_days_excluded(
        self, playbooks_yaml: Path, sqlite_db_url: str,
    ) -> None:
        # Zero-volume (halt/holiday artifact) days don't drag the mean.
        bars = _bars([0.0] * 5 + [100_000.0] * 20 + [300_000.0])
        rows = self._rows(_scanner(
            sqlite_db_url, playbooks_yaml, {"FRO": bars},
        ))
        assert rows["FRO"].rvol == pytest.approx(3.0)


# --------------------------------------------------------------------- #
# Threshold + min-volume floor
# --------------------------------------------------------------------- #


class TestThresholdAndFloor:
    def test_default_threshold_flags_at_2_5(
        self, playbooks_yaml: Path, sqlite_db_url: str,
    ) -> None:
        assert DEFAULT_RVOL_THRESHOLD == 2.5
        frames = {"FRO": _SPIKE, "XOM": _QUIET}
        rows = {
            r.ticker: r
            for r in _scanner(sqlite_db_url, playbooks_yaml, frames).scan(
                persist=False,
            )
        }
        assert rows["FRO"].is_unusual is True     # rvol 3.0
        assert rows["XOM"].is_unusual is False    # rvol 1.1

    def test_thin_otc_floor_never_flags(
        self, playbooks_yaml: Path, sqlite_db_url: str,
    ) -> None:
        # GLNCY-style tape: rvol 8.0 but avg volume 10k < 50k floor —
        # scanned and returned, never flagged.
        assert DEFAULT_MIN_AVG_VOLUME == 50_000.0
        thin = _bars([10_000.0] * 20 + [80_000.0])
        rows = {
            r.ticker: r
            for r in _scanner(
                sqlite_db_url, playbooks_yaml, {"FRO": thin},
            ).scan(persist=False)
        }
        assert rows["FRO"].rvol == pytest.approx(8.0)
        assert rows["FRO"].is_unusual is False

    def test_env_threshold_override(
        self, playbooks_yaml: Path, sqlite_db_url: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RVOL_THRESHOLD", "1.05")
        rows = {
            r.ticker: r
            for r in _scanner(
                sqlite_db_url, playbooks_yaml, {"XOM": _QUIET},
            ).scan(persist=False)
        }
        assert rows["XOM"].is_unusual is True     # rvol 1.1 >= 1.05

    def test_garbage_env_falls_back_to_default(
        self, playbooks_yaml: Path, sqlite_db_url: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RVOL_THRESHOLD", "very high")
        scanner = _scanner(sqlite_db_url, playbooks_yaml)
        assert scanner.rvol_threshold == DEFAULT_RVOL_THRESHOLD


# --------------------------------------------------------------------- #
# Failure tolerance
# --------------------------------------------------------------------- #


class TestFailureTolerance:
    def test_missing_ticker_skipped_others_survive(
        self, playbooks_yaml: Path, sqlite_db_url: str,
    ) -> None:
        # Batch has FRO only; per-ticker retries for the rest return
        # nothing usable → they are skipped, FRO still scanned.
        rows = _scanner(
            sqlite_db_url, playbooks_yaml, {"FRO": _SPIKE},
        ).scan(persist=False)
        assert [r.ticker for r in rows] == ["FRO"]

    def test_batch_failure_falls_back_to_per_ticker(
        self, playbooks_yaml: Path, sqlite_db_url: str,
    ) -> None:
        def download(tickers: Any, _s: Any, _e: Any) -> pd.DataFrame:
            if len(tickers) > 1:
                raise RuntimeError("yahoo melted")
            if tickers[0] == "FRO":
                return _batch({"FRO": _SPIKE})
            raise RuntimeError("no data")

        scanner = RelativeVolumeScanner(
            sqlite_db_url, playbooks_path=playbooks_yaml, downloader=download,
        )
        rows = scanner.scan(persist=False)
        assert [r.ticker for r in rows] == ["FRO"]

    def test_field_ticker_column_orientation_supported(
        self, playbooks_yaml: Path, sqlite_db_url: str,
    ) -> None:
        # yf.download WITHOUT group_by="ticker" → (field, ticker) columns.
        flipped = _batch({"FRO": _SPIKE}).swaplevel(axis=1)

        scanner = RelativeVolumeScanner(
            sqlite_db_url, playbooks_path=playbooks_yaml,
            downloader=lambda *_a: flipped,
        )
        rows = scanner.scan(persist=False)
        assert {r.ticker for r in rows} == {"FRO"}


# --------------------------------------------------------------------- #
# Persistence — ALL rows, not just unusual
# --------------------------------------------------------------------- #


class TestPersistence:
    def test_all_scanned_rows_persisted(
        self, playbooks_yaml: Path, sqlite_db_url: str,
    ) -> None:
        frames = {"FRO": _SPIKE, "XOM": _QUIET}
        _scanner(sqlite_db_url, playbooks_yaml, frames).scan()
        engine = create_engine(sqlite_db_url)
        with engine.connect() as conn:
            stored = conn.execute(text(
                "SELECT ticker, rvol, volume, is_unusual, source "
                "FROM volume_spikes ORDER BY ticker",
            )).fetchall()
        assert len(stored) == 2                    # quiet row persisted too
        by_ticker = {r[0]: r for r in stored}
        assert bool(by_ticker["FRO"][3]) is True
        assert bool(by_ticker["XOM"][3]) is False
        assert by_ticker["FRO"][1] == pytest.approx(3.0)
        assert by_ticker["FRO"][2] == 300_000
        assert by_ticker["FRO"][4] == "yfinance"

    def test_persist_false_writes_nothing(
        self, playbooks_yaml: Path, sqlite_db_url: str,
    ) -> None:
        _scanner(sqlite_db_url, playbooks_yaml, {"FRO": _SPIKE}).scan(
            persist=False,
        )
        engine = create_engine(sqlite_db_url)
        with engine.connect() as conn:
            n = conn.execute(text("SELECT COUNT(*) FROM volume_spikes")).scalar()
        assert n == 0


# --------------------------------------------------------------------- #
# Digest annotation
# --------------------------------------------------------------------- #


def _res(affected: list[dict[str, str]], urgency: int = 7) -> AssessmentResult:
    return AssessmentResult(
        event_id=1,
        headline="Strait of Hormuz closed",
        theme="energy_chokepoint",
        status="ASSESSED",
        assessment={
            "core_event": "x", "direction": "bearish", "urgency": urgency,
            "horizon": "hours", "confidence": 0.8,
            "affected": affected, "rationale": "r",
        },
    )


def _watch_aff(ticker: str) -> dict[str, str]:
    return {
        "instrument": ticker, "kind": "equity_watch",
        "direction": "watch", "reason": "why",
    }


class TestDigestAnnotation:
    def test_spike_present_annotates_watch_token(self) -> None:
        built = build_digest(
            [_res([_watch_aff("FRO"), _watch_aff("STNG")])],
            volume_marks={"FRO": 3.24},
        )
        assert built is not None
        _, message = built
        assert "FRO×3.2" in message
        assert "STNG×" not in message   # unmarked ticker stays bare

    def test_no_marks_renders_unannotated(self) -> None:
        for marks in (None, {}):
            built = build_digest([_res([_watch_aff("FRO")])], volume_marks=marks)
            assert built is not None
            _, message = built
            assert "FRO" in message
            assert "×" not in message

    def test_hostile_ticker_still_escaped_with_mark(self) -> None:
        built = build_digest(
            [_res([_watch_aff("<FRO>")])], volume_marks={"<FRO>": 3.0},
        )
        assert built is not None
        _, message = built
        assert "&lt;FRO&gt;×3.0" in message
        assert "<FRO>" not in message

    def test_send_digest_forwards_marks(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sent: list[str] = []

        def fake_notify(
            _title: str, message: str, priority: int = 0, *, html: bool = False,
        ) -> DispatchResult:
            sent.append(message)
            return DispatchResult(telegram_attempted=True, telegram_succeeded=True)

        monkeypatch.setattr("src.events.digest.notify_operator", fake_notify)
        send_digest([_res([_watch_aff("FRO")])], volume_marks={"FRO": 4.05})
        assert len(sent) == 1
        assert "FRO×4.0" in sent[0]


class TestFetchVolumeMarks:
    def _insert(
        self, db_url: str, ticker: str, age_hours: float,
        rvol: float, unusual: bool,
    ) -> None:
        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO volume_spikes "
                    "(ticker, scanned_at, rvol, is_unusual) "
                    "VALUES (:t, :at, :r, :u)"
                ),
                {
                    "t": ticker,
                    "at": datetime.now(UTC) - timedelta(hours=age_hours),
                    "r": rvol, "u": unusual,
                },
            )

    def test_latest_unusual_within_24h(self, sqlite_db_url: str) -> None:
        self._insert(sqlite_db_url, "FRO", 2.0, 2.8, True)
        self._insert(sqlite_db_url, "FRO", 1.0, 3.2, True)   # latest wins
        self._insert(sqlite_db_url, "STNG", 30.0, 9.9, True)  # too old
        self._insert(sqlite_db_url, "XOM", 1.0, 1.1, False)   # not unusual
        marks = fetch_volume_marks(create_engine(sqlite_db_url))
        assert marks == {"FRO": 3.2}

    def test_empty_table_returns_empty(self, sqlite_db_url: str) -> None:
        assert fetch_volume_marks(create_engine(sqlite_db_url)) == {}

    def test_missing_table_returns_empty(self, tmp_path: Path) -> None:
        # DB exists, migration never ran — digest must not die.
        engine = create_engine(f"sqlite:///{tmp_path / 'bare.db'}")
        assert fetch_volume_marks(engine) == {}

    def test_broken_engine_returns_empty(self) -> None:
        assert fetch_volume_marks(None) == {}


# --------------------------------------------------------------------- #
# Pipeline --scan wiring
# --------------------------------------------------------------------- #


class _FakeAgent:
    def __init__(self, **_kw: Any) -> None:
        pass

    def assess_new_events(self, limit: int = 20) -> list[AssessmentResult]:
        return []


class _FakeScanner:
    calls: list[dict[str, Any]] = []
    boom: bool = False

    def __init__(self, _db_url: str, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def scan(self, persist: bool = True) -> list[Any]:
        if _FakeScanner.boom:
            raise RuntimeError("yahoo melted")
        _FakeScanner.calls.append(self.kwargs)
        return []


@pytest.fixture
def pipeline_mod(monkeypatch: pytest.MonkeyPatch) -> Any:
    from scripts import event_pipeline as mod

    _FakeScanner.calls = []
    _FakeScanner.boom = False
    monkeypatch.setattr(
        "src.events.impact_agent.EventImpactAgent", _FakeAgent,
    )
    monkeypatch.setattr(
        "src.scanners.relative_volume.RelativeVolumeScanner", _FakeScanner,
    )
    monkeypatch.setattr("sqlalchemy.create_engine", lambda _url: None)
    return mod


class TestPipelineScanWiring:
    def test_scan_flag_default_on(self, pipeline_mod: Any) -> None:
        args = pipeline_mod._build_parser().parse_args(["--assess"])
        assert args.scan is True

    def test_no_scan_flag(self, pipeline_mod: Any) -> None:
        args = pipeline_mod._build_parser().parse_args(["--assess", "--no-scan"])
        assert args.scan is False

    def _cycle(self, pipeline_mod: Any, argv: list[str]) -> None:
        args = pipeline_mod._build_parser().parse_args(argv)
        args.digest_min_urgency = 5
        pipeline_mod._cycle(args)

    def test_cycle_runs_scanner_by_default(self, pipeline_mod: Any) -> None:
        self._cycle(pipeline_mod, ["--assess", "--no-digest"])
        assert len(_FakeScanner.calls) == 1
        assert _FakeScanner.calls[0]["playbooks_path"] == "configs/event_playbooks.yaml"

    def test_no_scan_skips_scanner(self, pipeline_mod: Any) -> None:
        self._cycle(pipeline_mod, ["--assess", "--no-digest", "--no-scan"])
        assert _FakeScanner.calls == []

    def test_scan_failure_does_not_crash_cycle(self, pipeline_mod: Any) -> None:
        _FakeScanner.boom = True
        self._cycle(pipeline_mod, ["--assess", "--no-digest"])  # must not raise

    def test_digest_receives_volume_marks(
        self, pipeline_mod: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: list[Any] = []

        def fake_send(_res: Any, **kw: Any) -> DispatchResult:
            captured.append(kw.get("volume_marks"))
            return DispatchResult()

        monkeypatch.setattr("src.events.digest.send_digest", fake_send)
        monkeypatch.setattr(
            "src.events.digest.fetch_volume_marks",
            lambda _engine: {"FRO": 3.2},
        )
        self._cycle(pipeline_mod, ["--assess"])
        assert captured == [{"FRO": 3.2}]


# --------------------------------------------------------------------- #
# DAG import (stubbed airflow — the real package isn't a unit-test dep)
# --------------------------------------------------------------------- #


class _FakeSkipError(Exception):
    pass


def _import_dag(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[Any]]:
    fake_airflow = types.ModuleType("airflow")
    fake_exceptions = types.ModuleType("airflow.exceptions")
    fake_operators = types.ModuleType("airflow.operators")
    fake_python = types.ModuleType("airflow.operators.python")

    class FakeDAG:
        def __init__(self, dag_id: str, **kwargs: Any) -> None:
            self.dag_id = dag_id
            self.kwargs = kwargs

        def __enter__(self) -> FakeDAG:
            return self

        def __exit__(self, *_exc: Any) -> bool:
            return False

    created: list[Any] = []

    class FakePythonOperator:
        def __init__(self, task_id: str, python_callable: Any, **_kw: Any) -> None:
            self.task_id = task_id
            self.python_callable = python_callable
            created.append(self)

    fake_airflow.DAG = FakeDAG  # type: ignore[attr-defined]
    fake_exceptions.AirflowSkipException = _FakeSkipError  # type: ignore[attr-defined]
    fake_python.PythonOperator = FakePythonOperator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "airflow", fake_airflow)
    monkeypatch.setitem(sys.modules, "airflow.exceptions", fake_exceptions)
    monkeypatch.setitem(sys.modules, "airflow.operators", fake_operators)
    monkeypatch.setitem(sys.modules, "airflow.operators.python", fake_python)

    dag_path = REPO_ROOT / "airflow" / "dags" / "event_ingestion.py"
    spec = importlib.util.spec_from_file_location(
        "_event_ingestion_dag_under_test", dag_path,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, created


class TestEventIngestionDag:
    def test_imports_cleanly_with_expected_shape(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mod, tasks = _import_dag(monkeypatch)
        assert mod.dag.dag_id == "event_ingestion"
        assert mod.dag.kwargs["schedule_interval"] == "*/10 * * * *"
        assert mod.dag.kwargs["catchup"] is False
        assert mod.dag.kwargs["max_active_runs"] == 1
        assert "curlit" in mod.dag.kwargs["tags"]
        task_ids = {t.task_id for t in tasks}
        # NO assess task on purpose — the claude CLI lives on the host.
        assert task_ids == {"ingest_gdelt", "scan_rvol"}
        for t in tasks:
            assert callable(t.python_callable)

    def test_playbooks_path_skips_when_config_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        mod, _tasks = _import_dag(monkeypatch)
        monkeypatch.delenv("EVENT_PLAYBOOKS_PATH", raising=False)
        monkeypatch.chdir(tmp_path)  # relative configs/ path won't resolve
        with pytest.raises(_FakeSkipError):
            mod._playbooks_path()

    def test_playbooks_path_env_override_wins(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        mod, _tasks = _import_dag(monkeypatch)
        cfg = tmp_path / "pb.yaml"
        cfg.write_text("themes: {}")
        monkeypatch.setenv("EVENT_PLAYBOOKS_PATH", str(cfg))
        assert mod._playbooks_path() == str(cfg)
