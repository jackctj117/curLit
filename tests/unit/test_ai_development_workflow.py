"""Supervised development configuration contracts (CL-doqh).

Oracles are the operator's requested model, permission, hook, and handoff
requirements, not generated snapshots of these files. No installed CLI, local
settings, broker, network, or database is required by these tests.
"""

from __future__ import annotations

import json
import shlex
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_codex_defaults_to_read_only_review_without_escalation() -> None:
    config = tomllib.loads((ROOT / ".codex/config.toml").read_text())
    assert config["model"] == "gpt-6-astra"
    assert config["review_model"] == "gpt-6-astra"
    assert config["sandbox_mode"] == "read-only"
    assert config["approval_policy"] == "never"


def test_claude_template_requires_manual_permissions_and_blocks_sensitive_actions() -> None:
    config = json.loads((ROOT / ".claude/settings.local.example.json").read_text())
    assert config["model"] == "claude-opus-5"
    permissions = config["permissions"]
    assert permissions["defaultMode"] == "default"
    assert permissions["disableBypassPermissionsMode"] == "disable"
    assert permissions["disableAutoMode"] == "disable"
    assert not permissions.get("allow")
    assert {
        "Read(./.env)",
        "Read(./.env.*)",
        "Read(./secrets/**)",
        "Bash(git push)",
        "Bash(git push *)",
        "Bash(./scripts/daemons.sh *)",
    } <= set(permissions["deny"])


@pytest.mark.parametrize("event", ["SessionStart", "PreCompact"])
def test_beads_hooks_only_load_local_read_only_guidance(event: str) -> None:
    config = json.loads((ROOT / ".claude/settings.json").read_text())
    hooks = [hook for group in config["hooks"][event] for hook in group["hooks"]]
    assert hooks
    for hook in hooks:
        assert hook["type"] == "command"
        assert shlex.split(hook["command"]) == ["bd", "--sandbox", "--readonly", "prime"]
    assert (ROOT / ".beads/PRIME.md").is_file()


def test_claude_imports_shared_policy_without_circular_import() -> None:
    claude = (ROOT / "CLAUDE.md").read_text()
    assert "@AGENTS.md" in claude.split("## Build & Test")[0].splitlines()
    assert "@CLAUDE.md" not in (ROOT / "AGENTS.md").read_text().splitlines()


@pytest.mark.parametrize("path", ["AGENTS.md", "CLAUDE.md", ".beads/PRIME.md"])
def test_old_autonomous_completion_policy_is_not_reintroduced(path: str) -> None:
    content = (ROOT / path).read_text()
    for old_mandate in (
        "Work is NOT complete until",
        "NEVER stop before pushing",
        "YOU must push",
        "Clear stashes, prune remote branches",
    ):
        assert old_mandate not in content


def test_local_settings_are_ignored_and_backup_publishing_is_disabled() -> None:
    assert ".claude/settings.local.json" in (ROOT / ".gitignore").read_text().splitlines()
    config = yaml.safe_load((ROOT / ".beads/config.yaml").read_text())
    assert config["backup"]["git-push"] is False
