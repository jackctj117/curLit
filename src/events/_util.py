"""Shared kernel for the events package (CL-ikz2).

Small utilities every events module kept re-implementing (code review
2026-07-21 §6.2.1):

* :func:`env_flag` — the boolean env-var parser previously copy-pasted
  verbatim into ``triage``, ``niche_agent`` and ``adversarial_critic``;
* :func:`clamp_int` / :func:`clamp_float` — defensive numeric clamps,
  reconciling the two prior calling conventions (``impact_agent`` raised
  on unparseable input; ``niche_agent`` returned a caller default). Both
  conventions work: omit ``default`` to get the raising form, pass it to
  get the fail-soft form;
* :func:`atomic_write_json` — atomic JSON persistence (tmp file +
  ``os.replace``, mirroring ``src.research.approvals.save_state_atomic``)
  so no events feature ever writes state in place.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, overload

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Sentinel: "no default supplied — re-raise on unparseable input".
_RAISE: Any = object()


def env_flag(name: str, default: bool) -> bool:
    """Boolean env toggle: unset → ``default``; otherwise True iff the
    value (stripped, case-insensitive) is one of ``1 / true / yes / on``."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


@overload
def clamp_int(value: Any, lo: int, hi: int) -> int: ...
@overload
def clamp_int(value: Any, lo: int, hi: int, default: int) -> int: ...


def clamp_int(value: Any, lo: int, hi: int, default: int = _RAISE) -> int:
    """``value`` → int (rounding via float), clamped to ``[lo, hi]``.

    Without ``default`` an unparseable value raises ``TypeError`` /
    ``ValueError`` (the caller decides what a bad field means — the
    impact-agent convention); with ``default`` it is returned instead
    (the niche-agent convention)."""
    try:
        return max(lo, min(hi, int(round(float(value)))))
    except (TypeError, ValueError):
        if default is _RAISE:
            raise
        return default


@overload
def clamp_float(value: Any, lo: float, hi: float) -> float: ...
@overload
def clamp_float(value: Any, lo: float, hi: float, default: float) -> float: ...


def clamp_float(value: Any, lo: float, hi: float, default: float = _RAISE) -> float:
    """``value`` → float clamped to ``[lo, hi]``; same default/raise
    contract as :func:`clamp_int`."""
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        if default is _RAISE:
            raise
        return default


def atomic_write_json(path: Path | str, obj: Any, *, indent: int = 2) -> None:
    """Persist ``obj`` as JSON atomically: write to a same-directory tmp
    file, then ``os.replace`` over ``path`` — a reader can never observe
    a half-written file (same idiom as
    ``src.research.approvals.save_state_atomic``; project rule: never
    write state in place). Parent directories are created; non-JSON
    values fall back to ``str`` (``default=str``). The tmp file is
    unlinked on any failure."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=indent, default=str)
    fd, tmp_name = tempfile.mkstemp(
        dir=p.parent,
        prefix=f".{p.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
        os.replace(tmp_name, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
