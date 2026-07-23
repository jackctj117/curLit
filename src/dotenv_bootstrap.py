"""Project-wide ``.env`` auto-loader.

Each CLI entrypoint calls ``load_project_env()`` as the first thing in
``main()`` so a developer can keep credentials in ``.env`` and run any
script directly (``python -m scripts.research_loop``) without first
sourcing the file. systemd / Docker deploys that set env vars
explicitly are unaffected — ``override=False`` means anything already
in ``os.environ`` wins over a value in ``.env``.

Why a wrapper instead of just calling ``load_dotenv()`` everywhere:

  * **Single search rule.** ``find_dotenv()`` walks up from the caller
    file looking for ``.env``; we want consistent behavior regardless
    of which entrypoint runs and what cwd the user invoked from.
  * **Safe defaults.** Industry convention is "explicit env wins";
    ``override=True`` would surprise an operator who set a value in
    a systemd unit file. We pin ``override=False`` here.
  * **Single log line.** Each entrypoint logs once at INFO level which
    file (if any) was loaded, so the per-run summary captures it.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: One-shot guard so the interpreter-health warning logs once per process.
_runtime_checked = False


def _check_interpreter_health() -> None:
    """Warn LOUDLY (never block) when critical C extensions can't load.

    CL-169t: a Homebrew python@3.14 upgrade (3.14.3_1 → 3.14.6) shipped a
    pyexpat linked against a newer /usr/lib/libexpat than this macOS has —
    every XML-touching daemon (arXiv/RSS/ENTSO-E ingest) crashed mid-run
    with a confusing dyld error. The venv is mitigated by pinning
    .venv/bin/python3.14 to the working Cellar 3.14.3_1 binary and
    `brew pin python@3.14`; this check makes any regression (e.g. a
    `brew cleanup` deleting 3.14.3_1) fail LOUD AND EARLY at boot with
    recovery instructions instead of cryptically at first XML parse.
    """
    global _runtime_checked  # noqa: PLW0603 — process-lifetime once-guard
    if _runtime_checked:
        return
    _runtime_checked = True
    try:
        import pyexpat  # noqa: F401, PLC0415 — the canary extension
    except ImportError as exc:
        logger.critical(
            "INTERPRETER HEALTH: pyexpat failed to import — XML ingest "
            "(arXiv/RSS/ENTSO-E) WILL crash. Likely cause: the venv's "
            "python symlink broke (brew cleanup removed the pinned "
            "Cellar 3.14.3_1?). Recovery: ln -sf /opt/homebrew/Cellar/"
            "python@3.14/<working-version>/bin/python3.14 .venv/bin/"
            "python3.14 — or rebuild the venv on a python whose pyexpat "
            "imports cleanly (see bd show CL-169t). Error: %s",
            exc,
        )


def load_project_env(env_path: Path | str | None = None) -> Path | None:
    """Load ``.env`` into ``os.environ`` without clobbering pre-set vars.

    Returns the path that was loaded, or ``None`` when no ``.env`` was
    found / the optional ``python-dotenv`` dep isn't importable.
    """
    _check_interpreter_health()  # CL-169t: loud early canary, never blocks
    try:
        from dotenv import find_dotenv, load_dotenv  # noqa: PLC0415
    except ImportError:
        logger.debug("python-dotenv not installed; skipping .env load")
        return None

    if env_path is not None:
        path = Path(env_path)
        if not path.exists():
            logger.debug("requested .env path %s does not exist", path)
            return None
    else:
        found = find_dotenv(usecwd=True)
        if not found:
            logger.debug("no .env found via find_dotenv()")
            return None
        path = Path(found)

    # override=False so explicit env vars (systemd, Docker, exported
    # in shell) win over the .env file. This is the safer default.
    load_dotenv(dotenv_path=path, override=False)
    logger.info("loaded environment from %s", path)
    return path
