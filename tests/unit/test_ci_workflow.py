"""Required-check configuration contracts (CL-0deu.5.1).

The operator's CI requirements are the oracle: every pull request gets all
three independent checks, failures propagate, and no trading credentials or
privileged checkout are available. These are static policy checks, not a claim
that a hosted runner or branch protection has been exercised.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def workflow() -> dict[str, Any]:
    """Parse the workflow at the YAML system boundary."""
    config = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    assert isinstance(config, dict)
    return config


def test_all_pull_requests_main_and_merge_queue_get_checks(workflow: dict[str, Any]) -> None:
    triggers = workflow["on"]
    assert triggers["pull_request"] is None
    assert triggers["push"] == {"branches": ["main"]}
    assert "merge_group" in triggers
    assert "pull_request_target" not in triggers
    assert "paths" not in str(triggers)


def test_checks_are_independent_and_failures_are_not_ignored(workflow: dict[str, Any]) -> None:
    job = workflow["jobs"]["checks"]
    assert job["name"] == "${{ matrix.check }}"
    assert job["strategy"]["fail-fast"] is False
    entries = job["strategy"]["matrix"]["include"]
    assert {entry["check"]: entry["target"] for entry in entries} == {
        "unit-tests": "test-unit",
        "lint": "lint",
        "typecheck": "typecheck",
    }
    assert len(entries) == 3
    assert "if" not in job
    assert not job.get("continue-on-error", False)
    for step in job["steps"]:
        assert "if" not in step
        assert not step.get("continue-on-error", False)
        command = step.get("run", "")
        assert "||" not in command
        assert "--exit-zero" not in command
    assert job["steps"][-1]["run"] == "make ${{ matrix.target }}"
    assert workflow["defaults"]["run"]["shell"] == "bash"


def test_hosted_checks_have_no_operational_credentials(workflow: dict[str, Any]) -> None:
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["checks"]
    assert job["runs-on"] == "ubuntu-24.04"
    assert "services" not in job
    assert "secrets" not in str(workflow)
    assert "environment" not in job
    assert workflow["env"]["CURLIT_RUN_NETWORK_TESTS"] == "0"
    checkout = next(step for step in job["steps"] if "checkout@" in step.get("uses", ""))
    assert checkout["with"]["persist-credentials"] is False
    for step in job["steps"]:
        if "uses" in step:
            assert re.fullmatch(r"actions/[a-z-]+@[a-f0-9]{40}", step["uses"])


def test_ci_installs_declared_dev_dependencies_and_supports_python_floor(
    workflow: dict[str, Any],
) -> None:
    steps = workflow["jobs"]["checks"]["steps"]
    setup = next(step for step in steps if "setup-python@" in step.get("uses", ""))
    assert setup["with"]["python-version"] == "3.11"
    commands = [step.get("run", "") for step in steps]
    assert 'python -m pip install -e ".[dev]"' in commands
    assert 'python -m pip install -e ".[audio,polymarket]"' in commands
    assert any('setuptools>=83.0.0' in command for command in commands)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dev = project["project"]["optional-dependencies"]["dev"]
    assert any(requirement.startswith("pytest-timeout>=") for requirement in dev)
    core = project["project"]["dependencies"]
    assert any(requirement.startswith("psutil>=") for requirement in core)
    assert any(requirement.startswith("cryptography>=") for requirement in core)
