"""Unit tests for the PROMOTE side-effect registrar (CL-3xn1).

External actions (git, gh) are injected callbacks so the registrar
runs in tests without touching the real repo or remote. We use
tmp_path-rooted experimental + production dirs + portfolio.yaml.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.research.promote import (
    PromoteRegistrar,
    add_strategy_to_portfolio_yaml,
    extract_first_class_name,
)

# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture
def repo(tmp_path: Path) -> dict[str, Path]:
    """Build a tmp_path "repo" with experimental/production dirs +
    portfolio yaml. Returns a dict of paths for tests."""
    experimental = tmp_path / "src" / "strategies" / "_experimental"
    production = tmp_path / "src" / "strategies"
    experimental.mkdir(parents=True)
    production.mkdir(parents=True, exist_ok=True)
    yaml_path = tmp_path / "configs" / "live_portfolio.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text(yaml.safe_dump({
        "engine": {"practice": True, "starting_equity": 100000},
        "strategies": [
            {"id": "existing_strat",
             "class": "src.strategies.existing.ExistingStrategy",
             "config": {}},
        ],
        "initial_weights": {"existing_strat": 0.5},
    }, sort_keys=False))
    return {
        "experimental": experimental,
        "production": production,
        "yaml": yaml_path,
        "tmp": tmp_path,
    }


def _seed_strategy(experimental: Path, slug: str, class_name: str = "MyStrat") -> Path:
    """Drop a fake strategy file into experimental/."""
    src = experimental / f"{slug}.py"
    src.write_text(
        f"class {class_name}:\n"
        f"    id = '{slug}'\n"
        f"    def fit(self, data): pass\n"
        f"    def generate_signals(self, data): return data * 0\n",
    )
    return src


# --------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------- #


class TestExtractFirstClass:
    def test_picks_first_top_level_class(self, tmp_path: Path) -> None:
        f = tmp_path / "x.py"
        f.write_text(
            "import os\n"
            "class FirstClass:\n    pass\n"
            "class SecondClass:\n    pass\n",
        )
        assert extract_first_class_name(f) == "FirstClass"

    def test_no_class_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "x.py"
        f.write_text("def hello(): pass\n")
        with pytest.raises(ValueError, match="no top-level class"):
            extract_first_class_name(f)


class TestAddStrategyToYaml:
    def test_appends_strategy_and_weight(self, repo: dict[str, Path]) -> None:
        diff = add_strategy_to_portfolio_yaml(
            repo["yaml"],
            strategy_slug="new_strat",
            class_path="src.strategies.new_strat.NewStrat",
        )
        raw = yaml.safe_load(repo["yaml"].read_text())
        slugs = [s["id"] for s in raw["strategies"]]
        assert "new_strat" in slugs
        assert raw["initial_weights"]["new_strat"] == 0.0
        # Existing entries preserved
        assert "existing_strat" in raw["initial_weights"]
        assert "+=1" in diff.replace(" ", "")  # both incremented

    def test_idempotent_on_existing_slug(self, repo: dict[str, Path]) -> None:
        # Pre-existing weight at 0.5; we shouldn't overwrite it
        add_strategy_to_portfolio_yaml(
            repo["yaml"],
            strategy_slug="existing_strat",
            class_path="src.strategies.existing.ExistingStrategy",
        )
        raw = yaml.safe_load(repo["yaml"].read_text())
        assert raw["initial_weights"]["existing_strat"] == 0.5
        # Strategies list shouldn't have a duplicate entry
        slugs = [s["id"] for s in raw["strategies"]]
        assert slugs.count("existing_strat") == 1

    def test_creates_initial_weights_when_missing(
        self, tmp_path: Path,
    ) -> None:
        yaml_path = tmp_path / "p.yaml"
        yaml_path.write_text(yaml.safe_dump({
            "strategies": [],
        }))
        add_strategy_to_portfolio_yaml(
            yaml_path, "new", "src.strategies.new.New",
        )
        raw = yaml.safe_load(yaml_path.read_text())
        assert raw["initial_weights"] == {"new": 0.0}


# --------------------------------------------------------------------- #
# PromoteRegistrar — full register() flow
# --------------------------------------------------------------------- #


class TestRegister:
    def test_happy_path_runs_all_steps(self, repo: dict[str, Path]) -> None:
        _seed_strategy(repo["experimental"], "alpha", class_name="AlphaStrat")
        git_calls: list[list[str]] = []
        gh_calls: list[list[str]] = []

        def fake_git(args: list[str]) -> str:
            git_calls.append(args)
            return ""

        def fake_gh(args: list[str]) -> str:
            gh_calls.append(args)
            return "https://github.com/user/repo/pull/42"

        reg = PromoteRegistrar(
            experimental_dir=repo["experimental"],
            production_dir=repo["production"],
            portfolio_yaml=repo["yaml"],
            git_runner=fake_git,
            gh_runner=fake_gh,
        )
        result = reg.register(
            strategy_slug="alpha",
            candidate_report_path=repo["tmp"] / "report.json",
            debate_transcript_path=repo["tmp"] / "transcript.md",
            verdict_reason="all gates pass",
        )
        assert result.succeeded
        assert result.error == ""
        assert result.steps_completed == [
            "move_file", "portfolio_yaml", "git", "pr",
        ]
        # File moved
        assert (repo["production"] / "alpha.py").exists()
        assert not (repo["experimental"] / "alpha.py").exists()
        # YAML updated
        raw = yaml.safe_load(repo["yaml"].read_text())
        slugs = [s["id"] for s in raw["strategies"]]
        assert "alpha" in slugs
        assert raw["initial_weights"]["alpha"] == 0.0
        # Git: checkout + add + commit + push
        assert any("checkout" in args for args in git_calls)
        assert any("commit" in args for args in git_calls)
        assert any("push" in args for args in git_calls)
        # PR opened with correct title
        assert len(gh_calls) == 1
        gh_args = gh_calls[0]
        assert "pr" in gh_args and "create" in gh_args
        title_idx = gh_args.index("--title") + 1
        assert "alpha" in gh_args[title_idx]
        assert result.pr_url == "https://github.com/user/repo/pull/42"
        assert result.branch_name == "experiment/alpha"

    def test_skip_git_skips_git_and_pr(self, repo: dict[str, Path]) -> None:
        _seed_strategy(repo["experimental"], "alpha")
        reg = PromoteRegistrar(
            experimental_dir=repo["experimental"],
            production_dir=repo["production"],
            portfolio_yaml=repo["yaml"],
            skip_git=True,
        )
        result = reg.register(
            strategy_slug="alpha",
            candidate_report_path=repo["tmp"] / "r.json",
            debate_transcript_path=repo["tmp"] / "t.md",
        )
        assert result.succeeded
        assert "git_skipped" in result.steps_completed
        assert "pr_skipped" in result.steps_completed
        # File still moved + YAML still updated
        assert (repo["production"] / "alpha.py").exists()

    def test_missing_experimental_file_fails(
        self, repo: dict[str, Path],
    ) -> None:
        # No file seeded
        reg = PromoteRegistrar(
            experimental_dir=repo["experimental"],
            production_dir=repo["production"],
            portfolio_yaml=repo["yaml"],
            skip_git=True,
        )
        result = reg.register(
            strategy_slug="ghost",
            candidate_report_path=repo["tmp"] / "r.json",
            debate_transcript_path=repo["tmp"] / "t.md",
        )
        assert not result.succeeded
        assert "move_file" in result.error
        assert "FileNotFoundError" in result.error

    def test_existing_production_file_refuses(
        self, repo: dict[str, Path],
    ) -> None:
        # File in BOTH places — refuse to clobber
        _seed_strategy(repo["experimental"], "alpha")
        (repo["production"] / "alpha.py").write_text("# pre-existing\n")
        reg = PromoteRegistrar(
            experimental_dir=repo["experimental"],
            production_dir=repo["production"],
            portfolio_yaml=repo["yaml"],
            skip_git=True,
        )
        result = reg.register(
            strategy_slug="alpha",
            candidate_report_path=repo["tmp"] / "r.json",
            debate_transcript_path=repo["tmp"] / "t.md",
        )
        assert not result.succeeded
        assert "FileExistsError" in result.error

    def test_idempotent_when_already_moved(self, repo: dict[str, Path]) -> None:
        # Source missing, dest exists — treat as already-moved
        (repo["production"] / "alpha.py").write_text(
            "class AlphaStrat: pass\n",
        )
        reg = PromoteRegistrar(
            experimental_dir=repo["experimental"],
            production_dir=repo["production"],
            portfolio_yaml=repo["yaml"],
            skip_git=True,
        )
        result = reg.register(
            strategy_slug="alpha",
            candidate_report_path=repo["tmp"] / "r.json",
            debate_transcript_path=repo["tmp"] / "t.md",
        )
        assert result.succeeded
        assert "move_file" in result.steps_completed

    def test_git_failure_short_circuits_pr(self, repo: dict[str, Path]) -> None:
        _seed_strategy(repo["experimental"], "alpha")

        def boom_git(_args: list[str]) -> str:
            raise RuntimeError("git push rejected")

        gh_calls: list[list[str]] = []

        def fake_gh(args: list[str]) -> str:
            gh_calls.append(args)
            return "url"

        reg = PromoteRegistrar(
            experimental_dir=repo["experimental"],
            production_dir=repo["production"],
            portfolio_yaml=repo["yaml"],
            git_runner=boom_git,
            gh_runner=fake_gh,
        )
        result = reg.register(
            strategy_slug="alpha",
            candidate_report_path=repo["tmp"] / "r.json",
            debate_transcript_path=repo["tmp"] / "t.md",
        )
        assert not result.succeeded
        assert "git" in result.error
        # PR step NOT reached
        assert "pr" not in result.steps_completed
        assert gh_calls == []

    def test_pr_body_contains_provenance(
        self, repo: dict[str, Path],
    ) -> None:
        _seed_strategy(repo["experimental"], "alpha")
        gh_args_captured: list[list[str]] = []

        def fake_gh(args: list[str]) -> str:
            gh_args_captured.append(args)
            return "https://example.com/pr/1"

        reg = PromoteRegistrar(
            experimental_dir=repo["experimental"],
            production_dir=repo["production"],
            portfolio_yaml=repo["yaml"],
            git_runner=lambda _a: "",
            gh_runner=fake_gh,
        )
        reg.register(
            strategy_slug="alpha",
            candidate_report_path=Path("reports/candidates/alpha.json"),
            debate_transcript_path=Path("docs/research/debates/alpha/transcript.md"),
            verdict_reason="all gates pass",
        )
        body_idx = gh_args_captured[0].index("--body") + 1
        body = gh_args_captured[0][body_idx]
        assert "alpha" in body
        assert "all gates pass" in body
        assert "allocation = 0" in body
        assert "candidate report" in body.lower()
        assert "debate transcript" in body.lower()


# ---------------------------------------------------------------------- #
# CL-2uns — gh PR URL extraction (regex, not last-line)
# ---------------------------------------------------------------------- #


class TestPRUrlExtraction:
    def test_extracts_canonical_url(self) -> None:
        from src.research.promote import _extract_pr_url
        out = "https://github.com/user/repo/pull/42\n"
        assert _extract_pr_url(out, "x") == (
            "https://github.com/user/repo/pull/42"
        )

    def test_finds_url_when_gh_emits_extra_lines(self) -> None:
        # Real failure mode: gh emits a deprecation notice / login
        # prompt AFTER the URL. The old "last line" parse breaks; the
        # regex finds the URL anywhere.
        from src.research.promote import _extract_pr_url
        out = (
            "https://github.com/user/repo/pull/42\n"
            "warning: gh CLI version 2.x is deprecated; upgrade soon\n"
        )
        assert _extract_pr_url(out, "x") == (
            "https://github.com/user/repo/pull/42"
        )

    def test_falls_back_to_last_line_when_no_url(self) -> None:
        from src.research.promote import _extract_pr_url
        out = "Created draft PR\nDone\n"
        # No canonical URL → fallback to last non-empty line
        assert _extract_pr_url(out, "x") == "Done"

    def test_empty_stdout_returns_empty(self) -> None:
        from src.research.promote import _extract_pr_url
        assert _extract_pr_url("", "x") == ""
        assert _extract_pr_url("   \n\n  ", "x") == ""

    def test_picks_first_canonical_url_when_multiple(self) -> None:
        from src.research.promote import _extract_pr_url
        out = (
            "Found related PR: https://github.com/user/repo/pull/40\n"
            "https://github.com/user/repo/pull/42\n"
        )
        # Regex match returns the first URL found
        assert _extract_pr_url(out, "x") == (
            "https://github.com/user/repo/pull/40"
        )
