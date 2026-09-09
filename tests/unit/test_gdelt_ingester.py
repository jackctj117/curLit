"""Tests for the GDELT Doc 2.0 ingester (CL-6iu7).

Covers: theme query construction, URL-hash dedup keys, ArtList JSON
parsing (mocked HTTP — no live GDELT calls), per-theme failure
tolerance, transform to the geo_events schema, and an end-to-end
``run()`` against a sqlite engine (idempotent re-runs, cross-theme
URL dedup).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from sqlalchemy import create_engine, text

from src.data import gdelt as gdelt_mod
from src.data.gdelt import (
    GdeltIngester,
    build_theme_query,
    url_external_id,
)
from src.events.playbooks import load_playbooks

START = datetime(2026, 7, 14, 9, 0, tzinfo=UTC)
END = datetime(2026, 7, 14, 10, 0, tzinfo=UTC)


@pytest.fixture
def bounded_ingester(sqlite_db_url, playbooks_yaml):
    ingester = GdeltIngester(sqlite_db_url, playbooks_path=playbooks_yaml)
    from migrations.run import _strip_sql_comments

    migration = _strip_sql_comments(Path("migrations/021_gdelt_ingest_cursors.sql").read_text())
    with ingester.engine.begin() as conn:
        for stmt in migration.split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
    return ingester


class _Clock:
    def __init__(self):
        self.value = END.timestamp()

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def _cursor(ingester, theme):
    with ingester.engine.connect() as conn:
        return json.loads(
            conn.execute(
                text("SELECT payload FROM gdelt_ingest_cursors WHERE theme=:theme"),
                {"theme": theme},
            ).scalar_one()
        )


def test_bounded_429_yields_without_sleep_and_retry_survives_restart(bounded_ingester):
    from src.data.gdelt_bounded import GdeltRead, run_slice

    clock = _Clock()
    calls = []

    def read(query, start, end, timeout, limit):
        calls.append(query)
        return GdeltRead("rate_limited", retry_after=120)

    kwargs = dict(read=read, wall_clock=clock.now, monotonic=clock.now, sleep=clock.sleep)
    result = run_slice(bounded_ingester, START, END, **kwargs)
    assert result["attempted"] == 1 and clock.value == END.timestamp()
    first = _cursor(bounded_ingester, "theme_a")
    assert first["status"] == "rate_limited" and "covered_until" not in first
    assert first["window_start"] == START.isoformat()
    # A new object is subject to the persisted source-wide Retry-After too.
    again = run_slice(bounded_ingester, START, END, **kwargs)
    assert again["attempted"] == 0 and len(calls) == 1
    clock.sleep(121)
    run_slice(bounded_ingester, START, END, **kwargs)
    assert "opec" in calls[1]  # Deferred later themes are not starved.


def test_bounded_requests_share_one_total_deadline(bounded_ingester):
    from src.data.gdelt_bounded import GdeltRead, run_slice

    clock = _Clock()
    timeouts = []

    def read(query, start, end, timeout, limit):
        timeouts.append(timeout)
        clock.sleep(timeout)
        return GdeltRead("ReadTimeout")

    result = run_slice(
        bounded_ingester,
        START,
        END,
        budget_sec=35,
        read=read,
        wall_clock=clock.now,
        monotonic=clock.now,
        sleep=clock.sleep,
    )
    assert timeouts == [30, 5]
    assert clock.value == END.timestamp() + 35 and result["completed"] == 0


def test_failed_persistence_keeps_pending_window_not_success(bounded_ingester, monkeypatch):
    from src.data.gdelt_bounded import GdeltRead, run_slice

    def fail(df):
        raise RuntimeError("fixture database write failure")

    monkeypatch.setattr(bounded_ingester, "upsert", fail)
    with pytest.raises(RuntimeError, match="database write failure"):
        run_slice(
            bounded_ingester,
            START,
            END,
            read=lambda *args: GdeltRead(
                "success", [_article("https://example.org/a", "Headline")]
            ),
        )
    state = _cursor(bounded_ingester, "theme_a")
    assert state["status"] == "pending" and "covered_until" not in state
    assert _cursor(bounded_ingester, "_source")["lease_owner"] is None


def test_result_cap_cannot_advance_coverage(bounded_ingester):
    from src.data.gdelt_bounded import GdeltRead, run_slice

    clock = _Clock()
    bounded_ingester.provider.max_records = 1
    run_slice(
        bounded_ingester,
        START,
        END,
        budget_sec=1,
        read=lambda *args: GdeltRead("success", [_article("https://example.org/a", "Headline")]),
        wall_clock=clock.now,
        monotonic=clock.now,
        sleep=clock.sleep,
    )
    state = _cursor(bounded_ingester, "theme_a")
    assert state["status"] == "result_cap_reached" and "covered_until" not in state


def test_second_ingester_cannot_acquire_an_active_source_lease(bounded_ingester):
    from src.data.gdelt_bounded import GdeltRead, run_slice

    def read(*args):
        second = run_slice(
            bounded_ingester, START, END, read=lambda *args: pytest.fail("concurrent request")
        )
        assert second["attempted"] == 0
        return GdeltRead("rate_limited")

    run_slice(bounded_ingester, START, END, read=read)


def test_pipeline_still_assesses_stored_events_during_gdelt_429(
    bounded_ingester, sqlite_db_url, playbooks_yaml, monkeypatch
):
    from types import SimpleNamespace

    from scripts import event_pipeline as ep

    from src.data import gdelt_bounded
    from src.events import impact_agent

    with bounded_ingester.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO geo_events (seen_at,source,external_id,headline,status,status_updated_at) VALUES (:now,'fixture','queued','Existing event','NEW',:now)"
            ),
            {"now": END.isoformat()},
        )

    async def limited(*args):
        return gdelt_bounded.GdeltRead("rate_limited", retry_after=120)

    assessed = []

    def assess_existing(**kwargs):
        # Stub only the assessment decision boundary; use the real queued row.
        with bounded_ingester.engine.begin() as conn:
            assessed.extend(
                conn.execute(
                    text("SELECT external_id FROM geo_events WHERE status='NEW'")
                ).scalars()
            )
            conn.execute(text("UPDATE geo_events SET status='ASSESSED' WHERE external_id='queued'"))
        return []

    monkeypatch.setattr(gdelt_bounded, "fetch_once", limited)
    monkeypatch.setattr(ep, "build_db_url", lambda: sqlite_db_url)
    monkeypatch.setattr(
        impact_agent,
        "EventImpactAgent",
        lambda **kwargs: SimpleNamespace(assess_new_events=assess_existing),
    )
    monkeypatch.setattr(ep, "_enrich_and_persist", lambda *args: ({}, {}))
    args = ep._build_parser().parse_args(
        ["--ingest", "--assess", "--once", "--playbooks", str(playbooks_yaml)]
    )
    args.niche = args.digest = args.poly = args.scan = False
    ep._cycle(args)
    assert assessed == ["queued"]
    assert _cursor(bounded_ingester, "theme_a")["status"] == "rate_limited"


@pytest.mark.parametrize(
    "payload,expected",
    [
        ("upstream error", "invalid_json"),
        ("{}", "invalid_articles"),
        ('{"articles":[]}', "success"),
        ('{"articles":[{}]}', "invalid_articles"),
        (
            '{"articles":[{"url":"https://example.org","title":"News","seendate":"bad"}]}',
            "invalid_article_date",
        ),
    ],
)
def test_bounded_http_distinguishes_empty_news_from_failed_ingestion(
    monkeypatch, payload, expected
):
    import asyncio

    import httpx

    from src.data import gdelt_bounded

    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=payload))
    monkeypatch.setattr(
        gdelt_bounded.httpx, "AsyncClient", lambda **kwargs: original(transport=transport, **kwargs)
    )
    assert asyncio.run(gdelt_bounded.fetch_once("query", START, END, 1, 50)).status == expected


def test_bounded_http_honors_retry_after_without_waiting(monkeypatch):
    import asyncio

    import httpx

    from src.data import gdelt_bounded

    original = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda request: httpx.Response(429, headers={"Retry-After": "120"})
    )
    monkeypatch.setattr(
        gdelt_bounded.httpx, "AsyncClient", lambda **kwargs: original(transport=transport, **kwargs)
    )
    result = asyncio.run(gdelt_bounded.fetch_once("query", START, END, 1, 50))
    assert result.status == "rate_limited" and result.retry_after == 120


def test_bounded_http_outer_deadline_cancels_a_stalled_response(monkeypatch):
    import asyncio

    import httpx

    from src.data import gdelt_bounded

    async def stalled(request):
        await asyncio.sleep(1)
        return httpx.Response(200, json={"articles": []})

    original = httpx.AsyncClient
    transport = httpx.MockTransport(stalled)
    monkeypatch.setattr(
        gdelt_bounded.httpx, "AsyncClient", lambda **kwargs: original(transport=transport, **kwargs)
    )
    result = asyncio.run(gdelt_bounded.fetch_once("query", START, END, 0.01, 50))
    assert result.status == "TimeoutError"


# ---------------------------------------------------------------------- #
# fixtures
# ---------------------------------------------------------------------- #


@pytest.fixture
def playbooks_yaml(tmp_path: Path) -> Path:
    doc = {
        "themes": {
            "theme_a": {
                "name": "Theme A",
                "description": "d",
                "watch_terms": ["strait of hormuz", "tanker"],
                "instruments": [
                    {
                        "instrument": "BCO_USD",
                        "kind": "oanda",
                        "direction": "long",
                        "rationale": "r",
                    }
                ],
            },
            "theme_b": {
                "name": "Theme B",
                "description": "d",
                "watch_terms": ["opec emergency"],
                "instruments": [
                    {
                        "instrument": "WTICO_USD",
                        "kind": "oanda",
                        "direction": "long",
                        "rationale": "r",
                    }
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
        .replace("JSONB", "TEXT")
        .replace("BIGSERIAL", "INTEGER")
        .replace(" DEFAULT NOW()", "")
    )


@pytest.fixture
def sqlite_db_url(tmp_path: Path) -> str:
    """Sqlite DB with the geo_events migration applied (shimmed)."""
    from migrations.run import _strip_sql_comments

    db_url = f"sqlite:///{tmp_path / 'geo.db'}"
    engine = create_engine(db_url)
    sql = _shim_pg_types_for_sqlite(
        _strip_sql_comments(
            Path("migrations/005_geo_events.sql").read_text(),
        )
    )
    with engine.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))
    return db_url


class _FakeResponse:
    def __init__(
        self,
        payload: dict | None = None,
        body: str = "",
        status_code: int = 200,
    ) -> None:
        self._payload = payload
        self.text = body if payload is None else json.dumps(payload)
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _article(url: str, title: str, seendate: str = "20260714T093000Z") -> dict:
    return {
        "url": url,
        "title": title,
        "seendate": seendate,
        "domain": "example.com",
        "language": "English",
    }


# ---------------------------------------------------------------------- #
# query construction + dedup key
# ---------------------------------------------------------------------- #


class TestQueryBuilding:
    def test_phrases_quoted_single_terms_bare(self, playbooks_yaml: Path) -> None:
        pbs = load_playbooks(playbooks_yaml)
        q = build_theme_query(pbs["theme_a"])
        assert q == '("strait of hormuz" OR tanker) sourcelang:english'

    def test_external_id_is_stable_url_hash(self) -> None:
        a = url_external_id("https://x.test/a")
        assert a == url_external_id("  https://x.test/a  ")  # strip-stable
        assert a != url_external_id("https://x.test/b")
        assert len(a) == 64  # sha256 hex


class TestRealConfigQueries:
    """CL-01zt: every expanded theme must build a well-formed, bounded
    GDELT query, and the total per-run request count must stay sane for
    the ingester's one-query-per-theme pacing."""

    NEW_THEMES = (
        "russia_ukraine",
        "africa_power_shift",
        "drc_copper_cobalt",
        "sahel_gold_uranium",
        "guinea_iron_bauxite",
        "south_africa_pgm_gold",
        "red_sea_shipping",
        "taiwan_semiconductor",
        "black_sea_grain",
        "pharma_api_supply",  # CL-lu80
    )

    @pytest.fixture(scope="class")
    def real_playbooks(self) -> dict:
        return load_playbooks("configs/event_playbooks.yaml")

    @pytest.mark.parametrize("theme", NEW_THEMES)
    def test_new_theme_query_shape(self, real_playbooks: dict, theme: str) -> None:
        q = build_theme_query(real_playbooks[theme])
        assert q.startswith("(")
        assert q.endswith(") sourcelang:english")

    def test_all_theme_queries_under_gdelt_length_ceiling(
        self,
        real_playbooks: dict,
    ) -> None:
        # GDELT's Doc API rejects long queries with an HTTP-200
        # plain-text "query was too short or too long" (observed live at
        # 255 chars, CL-01zt); the longest known-good query is 191 chars
        # (cb_surprise). Guard every theme under 200.
        for key, pb in real_playbooks.items():
            q = build_theme_query(pb)
            assert len(q) <= 200, (
                f"{key}: built GDELT query is {len(q)} chars — GDELT "
                f"rejects over-long queries; trim watch_terms"
            )

    def test_multiword_terms_are_phrase_quoted(self, real_playbooks: dict) -> None:
        for theme in self.NEW_THEMES:
            pb = real_playbooks[theme]
            q = build_theme_query(pb)
            for term in pb.watch_terms:
                expected = f'"{term}"' if " " in term else term
                assert expected in q, f"{theme}: {term!r} not in query"

    def test_per_run_request_load_stays_bounded(self, real_playbooks: dict) -> None:
        # One GDELT request per theme per run; the ingester paces at
        # pause_sec=6.0 with one 20s 429 cool-off retry. ~15 themes ≈
        # 90s+fetch per cycle — fine inside the 900s producer loop, but
        # unbounded growth here would eat the loop, so pin a ceiling.
        assert len(real_playbooks) <= 20


# ---------------------------------------------------------------------- #
# fetch (mocked HTTP)
# ---------------------------------------------------------------------- #


class TestFetch:
    def test_parses_artlist_and_tags_theme(
        self,
        monkeypatch: pytest.MonkeyPatch,
        playbooks_yaml: Path,
        sqlite_db_url: str,
    ) -> None:
        def fake_get(url: str, params: dict, timeout: float) -> _FakeResponse:
            if "hormuz" in params["query"]:
                return _FakeResponse(
                    {
                        "articles": [
                            _article("https://n.test/1", "Hormuz headline"),
                        ]
                    }
                )
            return _FakeResponse(
                {
                    "articles": [
                        _article("https://n.test/2", "OPEC headline"),
                    ]
                }
            )

        monkeypatch.setattr(gdelt_mod.httpx, "get", fake_get)
        ing = GdeltIngester(sqlite_db_url, playbooks_yaml, pause_sec=0)
        raw = ing.fetch(START, END)
        assert len(raw) == 2
        assert set(raw["theme"]) == {"theme_a", "theme_b"}

    def test_one_failing_theme_does_not_kill_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        playbooks_yaml: Path,
        sqlite_db_url: str,
    ) -> None:
        def fake_get(url: str, params: dict, timeout: float) -> _FakeResponse:
            if "hormuz" in params["query"]:
                raise ConnectionError("gdelt hiccup")
            return _FakeResponse(
                {
                    "articles": [
                        _article("https://n.test/2", "OPEC headline"),
                    ]
                }
            )

        monkeypatch.setattr(gdelt_mod.httpx, "get", fake_get)
        ing = GdeltIngester(sqlite_db_url, playbooks_yaml, pause_sec=0)
        raw = ing.fetch(START, END)
        assert list(raw["theme"]) == ["theme_b"]

    def test_429_retried_once_after_cooloff(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.data.gdelt import GdeltDocProvider

        calls: list[int] = []

        def fake_get(url: str, params: dict, timeout: float) -> _FakeResponse:
            calls.append(1)
            if len(calls) == 1:
                return _FakeResponse(body="rate limited", status_code=429)
            return _FakeResponse({"articles": [_article("https://n.test/1", "h")]})

        monkeypatch.setattr(gdelt_mod.httpx, "get", fake_get)
        provider = GdeltDocProvider(rate_limit_cooloff_sec=0.01)
        articles = provider.fetch_articles("(q)", START, END)
        assert len(calls) == 2
        assert len(articles) == 1

    def test_429_twice_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.data.gdelt import GdeltDocProvider

        monkeypatch.setattr(
            gdelt_mod.httpx,
            "get",
            lambda url, params, timeout: _FakeResponse(
                body="rate limited",
                status_code=429,
            ),
        )
        provider = GdeltDocProvider(rate_limit_cooloff_sec=0.01)
        with pytest.raises(RuntimeError, match="429"):
            provider.fetch_articles("(q)", START, END)

    def test_adaptive_pause_credits_fetch_time(
        self,
        monkeypatch: pytest.MonkeyPatch,
        playbooks_yaml: Path,
        sqlite_db_url: str,
    ) -> None:
        """CL-7vn9: the inter-theme pause is a cadence, not a blind sleep.
        A fetch that already burned >= pause_sec of wall-clock must sleep
        ~nothing before the next theme; request spacing stays >= pause_sec.
        """
        clock = [0.0]
        sleeps: list[float] = []

        def fake_monotonic() -> float:
            return clock[0]

        def fake_sleep(sec: float) -> None:
            sleeps.append(sec)
            clock[0] += sec

        def fake_get(url: str, params: dict, timeout: float) -> _FakeResponse:
            clock[0] += 10.0  # each fetch "takes" 10s — longer than pause
            return _FakeResponse(
                {
                    "articles": [
                        _article("https://n.test/x", "h"),
                    ]
                }
            )

        monkeypatch.setattr(gdelt_mod.time, "monotonic", fake_monotonic)
        monkeypatch.setattr(gdelt_mod.time, "sleep", fake_sleep)
        monkeypatch.setattr(gdelt_mod.httpx, "get", fake_get)

        ing = GdeltIngester(sqlite_db_url, playbooks_yaml, pause_sec=6.0)
        ing.fetch(START, END)
        # Two themes, each fetch 10s > 6s cadence → no top-up sleep at all.
        assert sleeps == []

    def test_adaptive_pause_tops_up_when_fetch_fast(
        self,
        monkeypatch: pytest.MonkeyPatch,
        playbooks_yaml: Path,
        sqlite_db_url: str,
    ) -> None:
        """A fast (near-instant) fetch still sleeps the full cadence to
        stay polite — spacing between request starts stays >= pause_sec."""
        clock = [0.0]
        sleeps: list[float] = []

        def fake_sleep(sec: float) -> None:
            sleeps.append(sec)
            clock[0] += sec

        def fake_get(url: str, params: dict, timeout: float) -> _FakeResponse:
            clock[0] += 0.01  # near-instant fetch
            return _FakeResponse(
                {
                    "articles": [
                        _article("https://n.test/x", "h"),
                    ]
                }
            )

        monkeypatch.setattr(gdelt_mod.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(gdelt_mod.time, "sleep", fake_sleep)
        monkeypatch.setattr(gdelt_mod.httpx, "get", fake_get)

        ing = GdeltIngester(sqlite_db_url, playbooks_yaml, pause_sec=6.0)
        ing.fetch(START, END)
        # One inter-theme gap, fetch was ~instant → nearly the full 6s.
        assert len(sleeps) == 1
        assert sleeps[0] == pytest.approx(6.0 - 0.01, abs=0.02)

    def test_non_json_body_treated_as_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
        playbooks_yaml: Path,
        sqlite_db_url: str,
    ) -> None:
        monkeypatch.setattr(
            gdelt_mod.httpx,
            "get",
            lambda url, params, timeout: _FakeResponse(
                body="Your query was too short or too long.",
            ),
        )
        ing = GdeltIngester(sqlite_db_url, playbooks_yaml, pause_sec=0)
        assert ing.fetch(START, END).empty


# ---------------------------------------------------------------------- #
# transform
# ---------------------------------------------------------------------- #


class TestTransform:
    def _ingester(self, playbooks_yaml: Path, sqlite_db_url: str) -> GdeltIngester:
        return GdeltIngester(sqlite_db_url, playbooks_yaml, pause_sec=0)

    def test_maps_to_geo_events_schema(
        self,
        playbooks_yaml: Path,
        sqlite_db_url: str,
    ) -> None:
        import pandas as pd

        ing = self._ingester(playbooks_yaml, sqlite_db_url)
        raw = pd.DataFrame(
            [
                {
                    "theme": "theme_a",
                    "url": "https://n.test/1",
                    "title": "Hormuz headline",
                    "seendate": "20260714T093000Z",
                }
            ]
        )
        df = ing.transform(raw)
        row = df.iloc[0]
        assert row["source"] == "gdelt"
        assert row["status"] == "NEW"
        assert row["headline"] == "Hormuz headline"
        assert row["external_id"] == url_external_id("https://n.test/1")
        assert row["seen_at"] == pd.Timestamp("2026-07-14T09:30:00Z")
        assert row["status_updated_at"] is not None

    def test_drops_rows_missing_url_or_title(
        self,
        playbooks_yaml: Path,
        sqlite_db_url: str,
    ) -> None:
        import pandas as pd

        ing = self._ingester(playbooks_yaml, sqlite_db_url)
        raw = pd.DataFrame(
            [
                {"theme": "t", "url": None, "title": "no url", "seendate": None},
                {"theme": "t", "url": "https://n.test/x", "title": "", "seendate": None},
                {"theme": "t", "url": "https://n.test/y", "title": "ok", "seendate": None},
            ]
        )
        df = ing.transform(raw)
        assert list(df["headline"]) == ["ok"]

    def test_bad_seendate_falls_back_to_now(
        self,
        playbooks_yaml: Path,
        sqlite_db_url: str,
    ) -> None:
        import pandas as pd

        ing = self._ingester(playbooks_yaml, sqlite_db_url)
        raw = pd.DataFrame(
            [
                {
                    "theme": "t",
                    "url": "https://n.test/1",
                    "title": "h",
                    "seendate": "garbage",
                }
            ]
        )
        df = ing.transform(raw)
        assert pd.notna(df.iloc[0]["seen_at"])


# ---------------------------------------------------------------------- #
# end-to-end run() against sqlite
# ---------------------------------------------------------------------- #


class TestEndToEnd:
    def test_run_inserts_then_dedups_on_rerun(
        self,
        monkeypatch: pytest.MonkeyPatch,
        playbooks_yaml: Path,
        sqlite_db_url: str,
    ) -> None:
        def fake_get(url: str, params: dict, timeout: float) -> _FakeResponse:
            if "hormuz" in params["query"]:
                return _FakeResponse(
                    {
                        "articles": [
                            _article("https://n.test/1", "Hormuz headline"),
                            _article("https://n.test/shared", "Shared story"),
                        ]
                    }
                )
            return _FakeResponse(
                {
                    "articles": [
                        _article("https://n.test/2", "OPEC headline"),
                        # Same URL matched by both themes — must collapse to one
                        _article("https://n.test/shared", "Shared story"),
                    ]
                }
            )

        monkeypatch.setattr(gdelt_mod.httpx, "get", fake_get)
        ing = GdeltIngester(sqlite_db_url, playbooks_yaml, pause_sec=0)

        assert ing.run(START, END) == 3  # 4 articles, 1 cross-theme dup

        engine = create_engine(sqlite_db_url)
        with engine.connect() as conn:
            n = conn.execute(text("SELECT COUNT(*) FROM geo_events")).scalar()
            statuses = {r[0] for r in conn.execute(text("SELECT status FROM geo_events"))}
        assert n == 3
        assert statuses == {"NEW"}

        # Second poll over the same window: everything already in the
        # table → zero inserts (idempotent for cron overlap).
        assert ing.run(START, END) == 0
        with engine.connect() as conn:
            n = conn.execute(text("SELECT COUNT(*) FROM geo_events")).scalar()
        assert n == 3
