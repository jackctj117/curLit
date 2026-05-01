"""Auto-discover FX/macro Polymarket markets and curate
``configs/polymarket_markets.yaml`` (CL-3t4j v2 follow-up).

Replaces the manual workflow:
  1. Find a market on https://polymarket.com — automated via Gamma /markets
  2. Get the CLOB token_id — automated; pulled from each market's
     ``clobTokenIds`` field
  3. Replace PLACEHOLDER tokens in the YAML — automated, with operator
     overrides preserved (any non-placeholder entry the operator
     already configured is kept as-is)

What this script does NOT do (stays operator-time):
  - Decide whether a market's signal is *useful* for FX trading. The
    keyword filter catches obvious matches (fed/cpi/ecb/dollar/etc.)
    but the agent debate is what determines whether a hypothesis
    using the data is tradeable.
  - Run the seed (history pull + DB upsert). That's
    ``scripts/seed_polymarket_history.py`` — runs after this on cron.

Cron-friendly. Idempotent: rerunning produces the same YAML if no
new markets matched. Operator-edited entries are preserved.

Usage:
  .venv/bin/python -m scripts.discover_polymarket_markets [--limit 20]
                                                          [--min-volume 10000]
                                                          [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

logger = logging.getLogger(__name__)


GAMMA_MARKETS_URL: str = "https://gamma-api.polymarket.com/markets"
DEFAULT_CONFIG_PATH: Path = Path("configs/polymarket_markets.yaml")
DEFAULT_HTTP_TIMEOUT_SEC: float = 30.0


# Same keyword set used by src/research/polymarket.py:PolymarketFetcher
# for research-side filtering. Kept in sync intentionally — discovery
# and research both need the same FX/macro lens.
_FX_MACRO_KEYWORDS: tuple[str, ...] = (
    "fed", "fomc", "rate cut", "rate hike", "inflation", "cpi", "ppi",
    "ecb", "boj", "bank of japan", "bank of england", "boe",
    "treasury", "yield curve", "recession", "gdp", "unemployment",
    "nonfarm", "nfp", "jobs report", "trade deficit", "tariff",
    "election", "central bank", "monetary policy", "interest rate",
    "dollar", "euro", "yen", "pound", "currency",
)


def fetch_active_markets(
    api_url: str = GAMMA_MARKETS_URL,
    limit: int = 200,
    timeout_sec: float = DEFAULT_HTTP_TIMEOUT_SEC,
) -> list[dict[str, Any]]:
    """Pull active, not-closed markets from the Gamma API. Returns
    the raw list — caller does the keyword + volume filtering."""
    resp = httpx.get(
        api_url,
        params={"active": "true", "closed": "false", "limit": str(limit)},
        timeout=timeout_sec,
        follow_redirects=True,
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return list(data["data"])
    return []


def is_fx_macro_relevant(market: dict[str, Any]) -> bool:
    """True if the market's question or description matches any of the
    FX/macro keywords."""
    haystack = (
        str(market.get("question") or market.get("title") or "")
        + " " + str(market.get("description") or "")
    ).lower()
    return any(kw in haystack for kw in _FX_MACRO_KEYWORDS)


def extract_yes_token_id(market: dict[str, Any]) -> str | None:
    """Pull the YES outcome's CLOB token ID. Polymarket returns
    ``clobTokenIds`` as either a JSON-encoded string or a native list
    of two IDs ``[yes_id, no_id]``. We always pick the first; that's
    the YES token by Polymarket convention."""
    raw = market.get("clobTokenIds")
    if not raw:
        return None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        ids = parsed if isinstance(parsed, list) else None
    elif isinstance(raw, list):
        ids = raw
    else:
        ids = None
    if not ids:
        return None
    first = ids[0]
    return str(first) if first else None


def get_volume(market: dict[str, Any]) -> float:
    """Float volume in USD, falling back across the various Gamma
    fields. Returns 0 on parse failure."""
    for key in ("volumeNum", "volume", "liquidityNum", "liquidity"):
        v = market.get(key)
        if v is None:
            continue
        try:
            return float(v)
        except (ValueError, TypeError):
            continue
    return 0.0


def filter_and_rank(
    markets: list[dict[str, Any]],
    min_volume_usd: float,
    limit: int,
) -> list[dict[str, Any]]:
    """Filter to FX/macro-relevant + above min-volume, sort by
    volume descending, cap to ``limit``."""
    relevant = []
    for m in markets:
        if not is_fx_macro_relevant(m):
            continue
        vol = get_volume(m)
        if vol < min_volume_usd:
            continue
        if not extract_yes_token_id(m):
            # No CLOB token id — can't seed price history for this market
            continue
        relevant.append(m)
    relevant.sort(key=get_volume, reverse=True)
    return relevant[:limit]


def merge_into_config(
    existing_path: Path,
    discovered: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Build a new YAML doc, preserving operator-curated entries.

    Rules:
      * Existing entries with non-placeholder token_ids are kept verbatim.
      * Existing entries with PLACEHOLDER_* token_ids: if a discovered
        market has the same slug, replace with the real entry. Otherwise
        leave the placeholder (operator may still want to fill in
        manually).
      * Discovered markets with slugs not yet in the YAML are appended.

    Returns (new_doc, stats) where stats has counts:
      preserved, replaced, added, placeholders_remaining.
    """
    if existing_path.exists():
        existing_doc = yaml.safe_load(existing_path.read_text()) or {}
    else:
        existing_doc = {}
    existing_markets = (
        existing_doc.get("markets", []) if isinstance(existing_doc, dict)
        else []
    )

    by_slug: dict[str, dict[str, Any]] = {}
    for m in existing_markets:
        if isinstance(m, dict) and m.get("slug"):
            by_slug[str(m["slug"])] = dict(m)

    discovered_by_slug = {
        str(m.get("slug")): m for m in discovered if m.get("slug")
    }
    stats = {"preserved": 0, "replaced": 0, "added": 0, "placeholders_remaining": 0}

    # Walk existing entries first to preserve order
    new_markets: list[dict[str, Any]] = []
    seen_slugs: set[str] = set()
    for m in existing_markets:
        if not isinstance(m, dict) or not m.get("slug"):
            new_markets.append(m)
            continue
        slug = str(m["slug"])
        seen_slugs.add(slug)
        token = str(m.get("token_id", ""))
        if slug in discovered_by_slug and token.startswith("PLACEHOLDER_"):
            # Replace placeholder with discovered real data
            new_markets.append(_canonical_entry(discovered_by_slug[slug], m))
            stats["replaced"] += 1
        elif token.startswith("PLACEHOLDER_"):
            new_markets.append(m)
            stats["placeholders_remaining"] += 1
        else:
            new_markets.append(m)
            stats["preserved"] += 1

    # Append discovered markets the YAML didn't already mention
    for slug, market in discovered_by_slug.items():
        if slug in seen_slugs:
            continue
        new_markets.append(_canonical_entry(market))
        stats["added"] += 1

    new_doc = dict(existing_doc) if isinstance(existing_doc, dict) else {}
    new_doc["markets"] = new_markets
    return new_doc, stats


def _canonical_entry(
    market: dict[str, Any], existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a polymarket_markets.yaml entry from a Gamma API market.
    Preserves any extra fields (description / fx_relevance) from an
    existing entry if provided."""
    entry: dict[str, Any] = {
        "slug": str(market.get("slug", "")),
        "token_id": str(extract_yes_token_id(market) or ""),
    }
    description = (
        (existing.get("description") if existing else None)
        or market.get("question")
        or market.get("title")
        or ""
    )
    if description:
        entry["description"] = str(description)
    if existing and "fx_relevance" in existing:
        entry["fx_relevance"] = existing["fx_relevance"]
    volume = get_volume(market)
    if volume:
        entry["discovered_volume_usd"] = round(volume, 0)
    end_date = market.get("endDate") or market.get("end_date_iso")
    if end_date:
        entry["end_date"] = str(end_date)[:10]
    return entry


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()

    p = argparse.ArgumentParser(
        description=(
            "Discover active FX/macro Polymarket markets via Gamma API "
            "and curate configs/polymarket_markets.yaml. Preserves "
            "operator-edited entries; only replaces PLACEHOLDER token_ids."
        ),
    )
    p.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH),
        help="Path to polymarket_markets.yaml",
    )
    p.add_argument(
        "--limit", type=int, default=20,
        help="Cap on number of markets to keep (highest volume first)",
    )
    p.add_argument(
        "--min-volume", type=float, default=10_000.0,
        help="Minimum USD volume to consider a market (filter noise)",
    )
    p.add_argument(
        "--gamma-limit", type=int, default=200,
        help="How many markets to pull from the Gamma API for filtering",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change but don't write the YAML",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        raw_markets = fetch_active_markets(limit=args.gamma_limit)
    except Exception as exc:
        print(
            f"Gamma API fetch failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    discovered = filter_and_rank(
        raw_markets,
        min_volume_usd=args.min_volume,
        limit=args.limit,
    )
    print(
        f"Gamma API returned {len(raw_markets)} active markets; "
        f"{len(discovered)} match FX/macro filter at "
        f">= ${args.min_volume:,.0f} volume.",
    )

    config_path = Path(args.config)
    new_doc, stats = merge_into_config(config_path, discovered)
    print(
        f"YAML changes: preserved={stats['preserved']} "
        f"replaced={stats['replaced']} added={stats['added']} "
        f"placeholders_remaining={stats['placeholders_remaining']}",
    )

    if args.dry_run:
        print("\nDRY RUN — would write:")
        print("---")
        print(yaml.safe_dump(new_doc, sort_keys=False, default_flow_style=False))
        return 0

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(new_doc, sort_keys=False, default_flow_style=False),
    )
    print(f"Wrote {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
