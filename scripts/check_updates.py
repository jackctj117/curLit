"""Automated update check — PyPI, GitHub, HuggingFace scans."""

import json
import logging
import subprocess
import sys
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


def check_pypi() -> list[dict]:
    """Run pip list --outdated and parse."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except Exception:
        logger.warning("PyPI check failed")
        return []


def check_github(repo: str) -> dict | None:
    import httpx

    try:
        resp = httpx.get(f"https://api.github.com/repos/{repo}/releases/latest", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {"repo": repo, "latest": data.get("tag_name"), "date": data.get("published_at")}
    except Exception:
        return None


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    results = {"timestamp": datetime.now(UTC).isoformat(), "pypi": [], "models": []}

    results["pypi"] = check_pypi()

    for repo in ["ProsusAI/finbert"]:
        release = check_github(repo)
        if release:
            results["models"].append(release)

    report = json.dumps(results, indent=2)
    print(report)

    if len(results["pypi"]) > 0:
        logger.info("Updates available: %d packages", len(results["pypi"]))


if __name__ == "__main__":
    main()
