"""Enumerate every installed third-party package for strict advisory lookup.

Only the private curLit root is excluded. Official torch +cpu wheels use the
upstream public release's advisory identity; unknown local builds fail closed.
This is an audit input, not a deployment lockfile or artifact-integrity check.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
from collections.abc import Iterable
from pathlib import Path

from packaging.utils import canonicalize_name
from packaging.version import Version


def audit_requirements(installed: Iterable[tuple[str, str]]) -> str:
    """Retain all third-party names/versions, with one explicit build mapping."""
    rows: dict[str, str] = {}
    comments = ["# Private curLit root excluded; every installed third-party package follows."]
    for raw_name, raw_version in installed:
        name = canonicalize_name(raw_name, validate=True)
        if name == "curlit":
            continue
        if name in rows:
            raise ValueError(f"Duplicate installed distribution: {name}")
        version = Version(raw_version)
        if version.local:
            if name != "torch" or version.local != "cpu":
                raise ValueError(f"Unreviewed local build: {name}=={version}")
            # PyTorch's official CPU index publishes +cpu builds of the same
            # release. PyPI advisory URLs lack build suffixes. Never omit torch.
            comments.append(f"# Advisory mapping: {name}=={version} -> {version.public}")
            version = Version(version.public)
        rows[name] = str(version)
    if not rows:
        raise ValueError("Empty third-party audit environment")
    return "\n".join([*comments, *(f"{name}=={rows[name]}" for name in sorted(rows)), ""])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    content = audit_requirements(
        (d.metadata["Name"], d.version) for d in metadata.distributions()
    )
    args.output.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
