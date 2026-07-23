"""PROMOTE side-effect registrar (CL-3xn1) — runs the four steps that
register a PROMOTE-verdict candidate as a paper-shadow strategy:

  1. Move ``src/strategies/_experimental/{slug}.py`` →
     ``src/strategies/{slug}.py``
  2. Add the strategy to ``configs/live_portfolio.yaml`` under
     ``strategies:`` with class path resolved from the module's first
     class definition; record ``initial_weights[slug] = 0.0`` so the
     coordinator zeroes out its intents until an operator manually
     raises the allocation.
  3. Create a git branch ``experiment/{slug}``, commit, push.
  4. Open a GitHub PR via ``gh`` containing the candidate report +
     debate transcript path + verdict reason.

**Allocation = 0 is the safety**. The strategy emits intents in the
live engine but the coordinator's allocation-based scaling zeros them
out. A real-money roll-out is a separate manual step (operator edits
``initial_weights`` to a positive number after soaking the shadow).

External actions (git, gh) are injected via callbacks so the registrar
is unit-testable without actually pushing or creating PRs. Default
implementations shell out via subprocess.

The registrar is **idempotent on first failure**: if step 1 succeeds
but step 2 fails, the next call sees the already-moved file and
proceeds. If step 3 fails (e.g. branch already pushed), the caller
gets a clean error — destructive-action recovery is the operator's
call, not the registrar's.
"""

from __future__ import annotations

import ast
import logging
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# Default repository paths.
DEFAULT_EXPERIMENTAL_DIR: Path = Path("src/strategies/_experimental")
DEFAULT_PRODUCTION_DIR: Path = Path("src/strategies")
DEFAULT_PORTFOLIO_YAML: Path = Path("configs/live_portfolio.yaml")


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


@dataclass
class RegistrationResult:
    """Outcome of one register() call. Auditable from the loop's per-
    run summary."""

    succeeded: bool
    strategy_slug: str
    error: str = ""
    code_path: Path | None = None  # final destination path
    branch_name: str = ""
    pr_url: str = ""
    yaml_diff: str = ""
    steps_completed: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Callback types
# --------------------------------------------------------------------------- #


# Run a git command list (e.g. ["checkout", "-b", "x"]) in cwd. Returns
# stdout. Raise on non-zero exit. Tests inject a recorder.
GitRunner = Callable[[list[str]], str]

# Run a gh command list (e.g. ["pr", "create", "--title", ...]) in cwd.
# Returns stdout (typically the PR URL on the last line). Raise on non-
# zero exit.
GhRunner = Callable[[list[str]], str]


def _default_git_runner(args: list[str]) -> str:
    """Production git wrapper. Raises CalledProcessError on non-zero."""
    return subprocess.check_output(
        ["git", *args],
        stderr=subprocess.STDOUT,
        text=True,
    )


def _default_gh_runner(args: list[str]) -> str:
    """Production gh wrapper. Raises CalledProcessError on non-zero."""
    return subprocess.check_output(
        ["gh", *args],
        stderr=subprocess.STDOUT,
        text=True,
    )


# Match a GitHub PR URL anywhere in gh stdout (CL-2uns). The previous
# parse used "last line" which broke if gh emitted deprecation notices
# or login prompts after the URL. The canonical URL shape is the only
# stable signal.
_GH_PR_URL_PATTERN = re.compile(
    r"https://github\.com/[^/\s]+/[^/\s]+/pull/\d+",
)


def _extract_pr_url(stdout: str, strategy_slug: str) -> str:
    """Pull the GitHub PR URL out of gh's stdout. Falls back to last-
    line if no canonical URL is found, with a warning — the caller
    treats this as best-effort. Returns empty string when nothing is
    parseable."""
    match = _GH_PR_URL_PATTERN.search(stdout)
    if match:
        return match.group(0)
    # Fallback: last non-empty line, in case gh's URL format ever shifts.
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if lines:
        logger.warning(
            "gh stdout for %s had no canonical PR URL; falling back to last line: %r",
            strategy_slug,
            lines[-1][:120],
        )
        return lines[-1]
    logger.warning("gh stdout for %s was empty; PR URL unknown", strategy_slug)
    return ""


# --------------------------------------------------------------------------- #
# Helpers — class extraction + YAML mutation
# --------------------------------------------------------------------------- #


def extract_first_class_name(code_path: Path) -> str:
    """Parse the strategy file and return the first top-level class
    name. The Implementer prompt names the strategy class but we don't
    enforce a naming convention — parsing is the truth."""
    tree = ast.parse(code_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            return node.name
    msg = f"no top-level class found in {code_path}"
    raise ValueError(msg)


def add_strategy_to_portfolio_yaml(
    yaml_path: Path,
    strategy_slug: str,
    class_path: str,
) -> str:
    """Append the strategy to live_portfolio.yaml's strategies list and
    set initial_weights[slug] = 0.0. Returns a short human-readable
    description of the change for the registration result.

    Idempotent: if the slug is already in either location, leaves it as
    found (does not overwrite an existing weight).
    """
    raw = yaml.safe_load(yaml_path.read_text()) or {}

    strategies = raw.setdefault("strategies", [])
    if not any(s.get("id") == strategy_slug for s in strategies):
        strategies.append(
            {
                "id": strategy_slug,
                "class": class_path,
                "config": {},
            }
        )
        added_strategy = True
    else:
        added_strategy = False

    weights = raw.setdefault("initial_weights", {})
    if strategy_slug not in weights:
        weights[strategy_slug] = 0.0
        added_weight = True
    else:
        added_weight = False

    if added_strategy or added_weight:
        yaml_path.write_text(
            yaml.safe_dump(raw, sort_keys=False, default_flow_style=False),
        )

    return f"strategies+={int(added_strategy)} initial_weights+={int(added_weight)}"


# --------------------------------------------------------------------------- #
# PromoteRegistrar
# --------------------------------------------------------------------------- #


class PromoteRegistrar:
    """Registers a PROMOTE-verdict candidate. Reusable across runs;
    state is per-call, not per-instance."""

    def __init__(
        self,
        experimental_dir: Path | str = DEFAULT_EXPERIMENTAL_DIR,
        production_dir: Path | str = DEFAULT_PRODUCTION_DIR,
        portfolio_yaml: Path | str = DEFAULT_PORTFOLIO_YAML,
        git_runner: GitRunner | None = None,
        gh_runner: GhRunner | None = None,
        skip_git: bool = False,
        skip_pr: bool = False,
    ) -> None:
        self.experimental_dir = Path(experimental_dir)
        self.production_dir = Path(production_dir)
        self.portfolio_yaml = Path(portfolio_yaml)
        self.git_runner: GitRunner = git_runner or _default_git_runner
        self.gh_runner: GhRunner = gh_runner or _default_gh_runner
        # Escape hatches for environments without git/gh access (e.g.
        # the dev's local machine before they wire credentials). The
        # loop fails open to "registered locally; PR step skipped" so
        # the operator can finish manually.
        self.skip_git = skip_git
        self.skip_pr = skip_pr

    def register(
        self,
        strategy_slug: str,
        candidate_report_path: Path | str,
        debate_transcript_path: Path | str,
        verdict_reason: str = "",
    ) -> RegistrationResult:
        """Run all four steps in order. Returns RegistrationResult with
        steps_completed listing exactly what ran — useful for the
        operator when partial-failure recovery is needed."""
        result = RegistrationResult(
            succeeded=False,
            strategy_slug=strategy_slug,
        )

        # Step 1: move the file
        try:
            final_path = self._move_strategy_file(strategy_slug)
            result.code_path = final_path
            result.steps_completed.append("move_file")
        except Exception as exc:
            result.error = f"move_file: {type(exc).__name__}: {exc}"
            return result

        # Step 2: portfolio YAML
        try:
            class_name = extract_first_class_name(final_path)
            module_dotted = self._module_dotted_path(final_path)
            class_path = f"{module_dotted}.{class_name}"
            result.yaml_diff = add_strategy_to_portfolio_yaml(
                self.portfolio_yaml,
                strategy_slug,
                class_path,
            )
            result.steps_completed.append("portfolio_yaml")
        except Exception as exc:
            result.error = f"portfolio_yaml: {type(exc).__name__}: {exc}"
            return result

        # Step 3: git
        branch_name = f"experiment/{strategy_slug}"
        result.branch_name = branch_name
        if self.skip_git:
            result.steps_completed.append("git_skipped")
        else:
            try:
                self._run_git_steps(
                    branch_name=branch_name,
                    final_path=final_path,
                    strategy_slug=strategy_slug,
                )
                result.steps_completed.append("git")
            except Exception as exc:
                result.error = f"git: {type(exc).__name__}: {exc}"
                return result

        # Step 4: GH PR
        if self.skip_pr or self.skip_git:
            result.steps_completed.append("pr_skipped")
        else:
            try:
                pr_url = self._open_pr(
                    branch_name=branch_name,
                    strategy_slug=strategy_slug,
                    candidate_report_path=Path(candidate_report_path),
                    debate_transcript_path=Path(debate_transcript_path),
                    verdict_reason=verdict_reason,
                )
                result.pr_url = pr_url
                result.steps_completed.append("pr")
            except Exception as exc:
                result.error = f"pr: {type(exc).__name__}: {exc}"
                return result

        result.succeeded = True
        return result

    # ------------------------------------------------------------------ #
    # Step 1 — file move
    # ------------------------------------------------------------------ #

    def _move_strategy_file(self, slug: str) -> Path:
        """Move _experimental/{slug}.py → strategies/{slug}.py.
        Idempotent: if the source is missing AND the destination
        exists, treat as already-moved (a previous partial-success
        rerun)."""
        source = self.experimental_dir / f"{slug}.py"
        dest = self.production_dir / f"{slug}.py"
        if not source.exists() and dest.exists():
            logger.info(
                "strategy file already at production path %s — skipping move",
                dest,
            )
            return dest
        if not source.exists():
            msg = (
                f"experimental strategy file not found at {source}; "
                f"implementer may not have run for slug {slug!r}"
            )
            raise FileNotFoundError(msg)
        if dest.exists():
            msg = (
                f"production path {dest} already exists — refusing to clobber. "
                f"Resolve manually before retrying."
            )
            raise FileExistsError(msg)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))
        return dest

    # ------------------------------------------------------------------ #
    # Step 2 helpers — module path
    # ------------------------------------------------------------------ #

    @staticmethod
    def _module_dotted_path(file_path: Path) -> str:
        """Convert ``src/strategies/foo.py`` → ``src.strategies.foo``.
        We pick the path relative to the project root by stripping
        leading ``./`` / absolute prefix; assumes the file lives under
        ``src/`` which is already how this codebase is laid out."""
        # If the path is absolute, make it relative to cwd which is the
        # project root for our scripts; if it's already relative, just
        # use as-is.
        try:
            rel = file_path.relative_to(Path.cwd())
        except ValueError:
            rel = file_path
        if rel.suffix != ".py":
            msg = f"expected a .py file, got {rel}"
            raise ValueError(msg)
        return ".".join(rel.with_suffix("").parts)

    # ------------------------------------------------------------------ #
    # Step 3 — git
    # ------------------------------------------------------------------ #

    def _run_git_steps(
        self,
        branch_name: str,
        final_path: Path,
        strategy_slug: str,
    ) -> None:
        """Create branch, add the moved file + the YAML, commit, push.

        We don't ``git rm`` the experimental path — git's rename
        detection picks up the move from the working-tree state when
        we ``git add`` both old and new paths. Adding the experimental
        directory ensures the deletion is staged.
        """
        # Create + switch to the branch. -B handles the case where the
        # branch already exists on a partial-failure retry.
        self.git_runner(["checkout", "-B", branch_name])
        self.git_runner(
            [
                "add",
                str(self.experimental_dir),
                str(self.production_dir / f"{strategy_slug}.py"),
                str(final_path.parent),  # in case parent dir is new
                str(self.portfolio_yaml),
            ]
        )
        self.git_runner(
            [
                "commit",
                "-m",
                f"feat(promote): paper-shadow {strategy_slug} (allocation=0)",
            ]
        )
        self.git_runner(["push", "-u", "origin", branch_name])

    # ------------------------------------------------------------------ #
    # Step 4 — gh
    # ------------------------------------------------------------------ #

    def _open_pr(
        self,
        branch_name: str,
        strategy_slug: str,
        candidate_report_path: Path,
        debate_transcript_path: Path,
        verdict_reason: str,
    ) -> str:
        title = f"promote: paper-shadow {strategy_slug} (allocation=0)"
        body = self._compose_pr_body(
            strategy_slug=strategy_slug,
            candidate_report_path=candidate_report_path,
            debate_transcript_path=debate_transcript_path,
            verdict_reason=verdict_reason,
            branch_name=branch_name,
        )
        out = self.gh_runner(
            [
                "pr",
                "create",
                "--title",
                title,
                "--body",
                body,
                "--head",
                branch_name,
            ]
        )
        return _extract_pr_url(out, strategy_slug)

    @staticmethod
    def _compose_pr_body(
        strategy_slug: str,
        candidate_report_path: Path,
        debate_transcript_path: Path,
        verdict_reason: str,
        branch_name: str,
    ) -> str:
        return (
            f"## PROMOTE: {strategy_slug}\n\n"
            f"Verdict reason: {verdict_reason or '(none recorded)'}\n\n"
            f"- candidate report: `{candidate_report_path}`\n"
            f"- debate transcript: `{debate_transcript_path}`\n"
            f"- branch: `{branch_name}`\n\n"
            f"### Allocation\n\n"
            f"Registered at **allocation = 0** in "
            f"`configs/live_portfolio.yaml`. The coordinator scales "
            f"intents by allocation, so the strategy emits but does "
            f"not get capital until an operator raises this manually "
            f"(soak first, then allocation > 0 in a follow-up commit).\n"
        )


# --------------------------------------------------------------------------- #
# Default factory
# --------------------------------------------------------------------------- #


def make_registrar(
    *,
    skip_git: bool = False,
    skip_pr: bool = False,
    **kwargs: Any,
) -> PromoteRegistrar:
    """Convenience factory used by the loop CLI. Same defaults as the
    constructor but accepts skip_git/skip_pr as the most common dev-
    machine overrides."""
    return PromoteRegistrar(skip_git=skip_git, skip_pr=skip_pr, **kwargs)
