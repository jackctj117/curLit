"""Keep the research watchdog's file I/O outside operational state."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_assessment_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSESSMENT_HEALTH_STATE", str(tmp_path / "assessment_health.json"))
