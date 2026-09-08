"""Security gate contracts and scoped exception oracles (CL-0deu.5.2).

These offline checks encode the requested fail-on-findings policy. Actual
scanner positive/negative fixtures are also exercised during implementation;
tests neither contact vulnerability services nor require installed scanners.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def security_workflow() -> dict[str, Any]:
    config = yaml.safe_load((ROOT / ".github/workflows/security-scan.yml").read_text())
    assert isinstance(config, dict)
    return config


def test_security_checks_are_independent_and_unprivileged(
    security_workflow: dict[str, Any],
) -> None:
    assert security_workflow["permissions"] == {"contents": "read"}
    assert security_workflow["on"]["pull_request"] is None
    assert "merge_group" in security_workflow["on"]
    assert "schedule" in security_workflow["on"]
    assert "pull_request_target" not in security_workflow["on"]
    jobs = security_workflow["jobs"]
    assert set(jobs) == {"bandit", "dependencies", "secrets"}
    for name, job in jobs.items():
        assert job["name"] == f"security-{name}"
        assert "needs" not in job
        assert "if" not in job
        assert "continue-on-error" not in job
        assert job["runs-on"] == "ubuntu-24.04"
        for step in job["steps"]:
            assert "continue-on-error" not in step
            assert "secrets." not in str(step)
            assert "|| true" not in step.get("run", "")
            assert "--exit-zero" not in step.get("run", "")
            if "uses" in step:
                assert re.fullmatch(r"actions/[a-z-]+@[a-f0-9]{40}", step["uses"])
            if "checkout@" in step.get("uses", ""):
                assert step["with"]["persist-credentials"] is False


def test_scanners_keep_errors_and_findings_blocking(security_workflow: dict[str, Any]) -> None:
    jobs = security_workflow["jobs"]
    bandit_steps = jobs["bandit"]["steps"]
    commands = "\n".join(step.get("run", "") for step in bandit_steps)
    assert "-r src/ -ll -f json" in commands
    assert '--ignore-nosec' not in commands  # Narrow reviewed nosec comments remain possible.
    assert 'sys.exit(bool(report["errors"]))' in commands
    assert bandit_steps[-1]["if"] == "always()"
    audit = "\n".join(step.get("run", "") for step in jobs["dependencies"]["steps"])
    assert "pip_audit --strict --skip-editable" in audit
    assert "--ignore-vuln" not in audit
    assert "--fix" not in audit
    assert "pip==26.2.1" in audit


def test_history_secret_scan_is_redacted_and_checksum_verified(
    security_workflow: dict[str, Any],
) -> None:
    steps = security_workflow["jobs"]["secrets"]["steps"]
    checkout = next(step for step in steps if "checkout@" in step.get("uses", ""))
    assert checkout["with"]["fetch-depth"] == 0
    commands = "\n".join(step.get("run", "") for step in steps)
    assert "sha256sum --check --strict" in commands
    assert commands.index("sha256sum") < commands.index("tar -xzf")
    assert '"$RUNNER_TEMP/gitleaks" git' in commands
    assert "--redact=100" in commands
    assert "--log-opts=--all" in commands
    assert "--ignore-gitleaks-allow" in commands
    assert "--config .gitleaks.toml" in commands
    assert "--baseline-path" not in commands


def test_secret_exception_cannot_allow_other_values_paths_or_rules() -> None:
    config = tomllib.loads((ROOT / ".gitleaks.toml").read_text())
    assert config["extend"] == {"useDefault": True}
    assert "allowlists" not in config
    assert len(config["rules"]) == 1
    rule = config["rules"][0]
    assert rule["id"] == "generic-api-key"
    assert len(rule["allowlists"]) == 1
    allow = rule["allowlists"][0]
    assert allow["condition"] == "AND"
    assert allow["regexTarget"] == "secret"
    assert allow["regexes"] == ["^PLACEHOLDER_ECB_Q3_YES$"]
    assert "commits" not in allow
    assert "stopwords" not in allow
    pattern, = allow["paths"]
    assert re.search(pattern, "configs/polymarket_markets.yaml")
    assert not re.search(pattern, "configs/credentials.yaml")
    assert not re.search(pattern, "configs/polymarket_markets.yaml.bak")


@pytest.mark.parametrize("has_parse_error", [False, True])
def test_actual_summary_rejects_incomplete_bandit_scans(
    security_workflow: dict[str, Any], tmp_path: Path, has_parse_error: bool,
) -> None:
    """Execute the shipped report check against independent complete/partial reports."""
    report = {
        "results": [],
        "errors": [{"filename": "broken.py", "reason": "syntax error"}] if has_parse_error else [],
    }
    (tmp_path / "bandit.json").write_text(json.dumps(report))
    command = security_workflow["jobs"]["bandit"]["steps"][-1]["run"]
    lines = command.strip().splitlines()
    assert lines[0] == "python - <<'PY'"
    assert lines[-1] == "PY"
    process = subprocess.run(
        [sys.executable, "-c", "\n".join(lines[1:-1])],
        cwd=tmp_path,
        env={"RUNNER_TEMP": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
        timeout=10,  # A tiny local JSON check should not hang a unit-test worker.
    )
    assert process.returncode == (1 if has_parse_error else 0), process.stderr
