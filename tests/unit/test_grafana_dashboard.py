"""Smoke-test the Grafana portfolio dashboard JSON (CL-3b2).

Validates structure without actually running Grafana. Catches future
edits that break the dashboard schema (orphan datasource UIDs,
duplicate panel IDs, panel queries referencing renamed metrics).

The metric-existence check uses the same metric registry the engine
emits at runtime (src.monitoring.metrics) so a metric removal in code
is caught here before deploy time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_DASHBOARD_PATH = Path("grafana/dashboards/portfolio.json")


@pytest.fixture(scope="module")
def dashboard() -> dict:  # type: ignore[type-arg]
    return json.loads(_DASHBOARD_PATH.read_text())


class TestStructure:
    def test_required_top_level_keys(self, dashboard: dict) -> None:  # type: ignore[type-arg]
        for key in ("title", "uid", "panels", "schemaVersion"):
            assert key in dashboard, f"missing {key}"

    def test_panel_ids_are_unique(self, dashboard: dict) -> None:  # type: ignore[type-arg]
        ids = [p["id"] for p in dashboard["panels"]]
        assert len(ids) == len(set(ids)), f"duplicate panel ids: {ids}"

    def test_each_panel_has_grid_pos(self, dashboard: dict) -> None:  # type: ignore[type-arg]
        for p in dashboard["panels"]:
            assert "gridPos" in p, f"panel {p.get('title')} missing gridPos"
            for k in ("h", "w", "x", "y"):
                assert k in p["gridPos"], f"panel {p['title']} gridPos missing {k}"


class TestQueriesReferenceRealMetrics:
    """Pull every metric referenced in the dashboard and assert it's
    declared in src/monitoring/metrics.py. Catches typos + renames."""

    def test_metrics_in_dashboard_exist_in_code(self, dashboard: dict) -> None:  # type: ignore[type-arg]
        # The metric module declares Counter/Gauge/Histogram constants
        # whose first arg is the Prometheus metric name. Pull them.
        metrics_src = Path("src/monitoring/metrics.py").read_text()
        # Names look like: Counter("fx_xxx", ... or Gauge("fx_xxx", ...
        declared = set(re.findall(r'(?:Counter|Gauge|Histogram)\(\s*"(fx_[a-z_]+)"', metrics_src))

        # Pull metric names referenced by every panel target's expr.
        referenced: set[str] = set()
        for p in dashboard["panels"]:
            for t in p.get("targets", []) or []:
                expr = t.get("expr", "")
                referenced.update(re.findall(r"\bfx_[a-z_]+", expr))

        # Prometheus auto-appends _total to Counter exports unless the
        # declaration already ends with _total. Same for _created (the
        # birth-timestamp shadow series). Accept both forms.
        acceptable = set(declared)
        for d in declared:
            if not d.endswith("_total"):
                acceptable.add(d + "_total")
            acceptable.add(d + "_created")

        missing = referenced - acceptable
        assert not missing, (
            f"Dashboard references metrics not declared in metrics.py: {sorted(missing)}"
        )


class TestSectionCoverage:
    """Each curLit subsystem with metrics should have at least one panel.
    A section that goes silent on a regression is a regression."""

    def test_health_section_present(self, dashboard: dict) -> None:  # type: ignore[type-arg]
        section_titles = [p["title"] for p in dashboard["panels"] if p.get("type") == "row"]
        # Expect rows for each subsystem.
        for needle in (
            "System health",
            "Portfolio",
            "Allocation",
            "Attribution",
            "Risk",
        ):
            assert any(needle in s for s in section_titles), (
                f"no row mentions {needle!r}; got {section_titles}"
            )


class TestPrometheusDatasourceConsistency:
    def test_every_target_uses_prometheus_datasource(
        self,
        dashboard: dict,  # type: ignore[type-arg]
    ) -> None:
        for p in dashboard["panels"]:
            if p.get("type") == "row":
                continue
            ds = p.get("datasource", {})
            if isinstance(ds, dict):
                assert ds.get("type") == "prometheus", (
                    f"panel {p.get('title')!r} not on prometheus datasource"
                )
