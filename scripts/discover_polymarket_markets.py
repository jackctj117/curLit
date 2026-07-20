"""Auto-discover Polymarket markets and curate a tracking config.

Two lenses share the same Gamma-API plumbing:

* **FX/macro** (CL-3t4j v2) — ``--mode fx`` (default). Keyword-matches
  fed/cpi/ecb/dollar/etc. and curates ``configs/polymarket_markets.yaml``
  (the macro DataProvider-symbol config), preserving operator overrides
  and only replacing PLACEHOLDER token_ids.

* **Geopolitical** (CL-r1ep) — ``--mode geo``. Keyword-matches every
  market's question/slug against ALL event-playbook ``watch_terms``
  (Taiwan / Hormuz / Iran / Russia / coup / sanctions …), assigns the
  single best-matching theme, and writes theme-tagged entries
  ``{slug, question, yes_token_id, theme, discovered_at}`` to a NEW
  config ``configs/polymarket_geo_markets.yaml``. This is how the
  operator populates the real geopolitical markets that
  :class:`src.events.polymarket_signal.PolymarketSignal` then polls each
  event-pipeline cycle for probability-shift Telegram alerts.

Replaces the manual workflow:
  1. Find a market on https://polymarket.com — automated via Gamma /markets
  2. Get the CLOB token_id — automated; pulled from each market's
     ``clobTokenIds`` field (a JSON-string array ``[yes_id, no_id]``)
  3. Populate the tracking config — automated, dedup vs existing entries

What this script does NOT do (stays operator-time):
  - Decide whether a market's signal is *tradeable*. The keyword filter
    catches theme matches but the agent debate / operator decides.
  - Run the seed (history pull + DB upsert). That's
    ``scripts/seed_polymarket_history.py`` — runs after this on cron.

Cron-friendly. Idempotent: rerunning produces the same YAML if no
new markets matched. Existing entries are preserved / deduped.

Usage:
  .venv/bin/python -m scripts.discover_polymarket_markets            # geo (default)
  .venv/bin/python -m scripts.discover_polymarket_markets --mode fx
  .venv/bin/python -m scripts.discover_polymarket_markets --limit 200 \\
      --min-volume 5000 --out configs/polymarket_geo_markets.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

logger = logging.getLogger(__name__)


GAMMA_MARKETS_URL: str = "https://gamma-api.polymarket.com/markets"
DEFAULT_CONFIG_PATH: Path = Path("configs/polymarket_markets.yaml")
DEFAULT_GEO_CONFIG_PATH: Path = Path("configs/polymarket_geo_markets.yaml")
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


#: The Gamma /markets endpoint caps a single response at 100 rows
#: regardless of the requested limit, so we page with ``offset``.
_GAMMA_PAGE_SIZE: int = 100


def _one_page(
    api_url: str, params: dict[str, str], timeout_sec: float,
) -> list[dict[str, Any]]:
    resp = httpx.get(
        api_url, params=params, timeout=timeout_sec, follow_redirects=True,
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return list(data["data"])
    return []


def fetch_active_markets(
    api_url: str = GAMMA_MARKETS_URL,
    limit: int = 200,
    timeout_sec: float = DEFAULT_HTTP_TIMEOUT_SEC,
    order_by_volume24hr: bool = False,
) -> list[dict[str, Any]]:
    """Pull active, not-closed markets from the Gamma API. Returns
    the raw list — caller does the keyword + volume filtering.

    ``order_by_volume24hr`` requests markets sorted by 24h volume (the
    VERIFIED-free geo endpoint variant) so the pull lands on the busiest
    markets first. Because Gamma caps one response at 100 rows, ``limit``
    above that is satisfied by paging with ``offset`` until ``limit`` is
    reached or a short page signals the end."""
    base: dict[str, str] = {"active": "true", "closed": "false"}
    if order_by_volume24hr:
        base["order"] = "volume24hr"
        base["ascending"] = "false"

    if limit <= _GAMMA_PAGE_SIZE:
        return _one_page(
            api_url, {**base, "limit": str(limit)}, timeout_sec,
        )[:limit]

    out: list[dict[str, Any]] = []
    offset = 0
    while len(out) < limit:
        page = _one_page(
            api_url,
            {**base, "limit": str(_GAMMA_PAGE_SIZE), "offset": str(offset)},
            timeout_sec,
        )
        if not page:
            break
        out.extend(page)
        if len(page) < _GAMMA_PAGE_SIZE:
            break  # last page
        offset += _GAMMA_PAGE_SIZE
    return out[:limit]


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
    existing_doc = yaml.safe_load(existing_path.read_text()) or {} if existing_path.exists() else {}
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


# ---------------------------------------------------------------------- #
# Geopolitical discovery (CL-r1ep) — keyword-match against playbook
# watch_terms, assign a theme, write configs/polymarket_geo_markets.yaml.
# ---------------------------------------------------------------------- #


def _load_theme_terms(
    playbooks_path: Path | str = "configs/event_playbooks.yaml",
) -> dict[str, tuple[str, ...]]:
    """theme key → lowercased watch_terms, straight from the playbooks.
    Reuses the same knowledge base the GDELT ingester and impact agent
    consume, so market→theme matching tracks the pipeline's own lexicon."""
    from src.events.playbooks import load_playbooks  # noqa: PLC0415

    playbooks = load_playbooks(playbooks_path)
    return {
        key: tuple(t.lower() for t in pb.watch_terms)
        for key, pb in playbooks.items()
    }


def match_theme(
    market: dict[str, Any],
    theme_terms: dict[str, tuple[str, ...]],
) -> tuple[str, str] | None:
    """Best-matching playbook theme for a market, or None if nothing
    matches. Scores each theme by the number of its watch_terms that
    appear in the market's question+slug; the highest score wins (ties
    broken by the longest single matched term, then theme name for
    determinism). Returns ``(theme, matched_term)``."""
    # Slugs are hyphenated (taiwan-blockade-q4); watch_terms are
    # space-separated ("taiwan blockade"). Normalize hyphens/underscores
    # to spaces so slug tokens match multi-word terms.
    haystack = (
        str(market.get("question") or market.get("title") or "")
        + " " + str(market.get("slug") or "")
    ).lower().replace("-", " ").replace("_", " ")
    if not haystack.strip():
        return None
    best: tuple[int, int, str, str] | None = None  # (score, term_len, theme, term)
    for theme, terms in theme_terms.items():
        hits = [t for t in terms if t and t in haystack]
        if not hits:
            continue
        longest = max(hits, key=len)
        candidate = (len(hits), len(longest), theme, longest)
        # Prefer more hits, then a longer matched term; theme name is the
        # final deterministic tie-break (reverse so 'a...' beats 'z...').
        if best is None or (
            candidate[0],
            candidate[1],
            tuple(-ord(c) for c in candidate[2]),
        ) > (best[0], best[1], tuple(-ord(c) for c in best[2])):
            best = candidate
    if best is None:
        return None
    return best[2], best[3]


def extract_yes_prob(market: dict[str, Any]) -> float | None:
    """Current YES probability from Gamma's ``outcomePrices`` — a
    JSON-string array ``"[yesProb, noProb]"`` (or a native list). None
    when absent/unparseable."""
    raw = market.get("outcomePrices")
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        prices = parsed if isinstance(parsed, list) else None
    elif isinstance(raw, list):
        prices = raw
    else:
        prices = None
    if not prices:
        return None
    try:
        return float(prices[0])
    except (ValueError, TypeError):
        return None


def discover_geo_markets(
    markets: list[dict[str, Any]],
    theme_terms: dict[str, tuple[str, ...]],
    min_volume_usd: float,
    limit: int,
) -> list[dict[str, Any]]:
    """Filter raw Gamma markets to theme-matched geopolitical ones above
    the volume floor with a usable YES token, ranked by volume desc.
    Returns theme-tagged discovery dicts."""
    out: list[tuple[float, dict[str, Any]]] = []
    for m in markets:
        vol = get_volume(m)
        if vol < min_volume_usd:
            continue
        yes_token = extract_yes_token_id(m)
        if not yes_token:
            continue  # no CLOB token id → can't poll a midpoint later
        matched = match_theme(m, theme_terms)
        if matched is None:
            continue
        theme, matched_term = matched
        slug = str(m.get("slug") or "")
        if not slug:
            continue
        entry: dict[str, Any] = {
            "slug": slug,
            "question": str(m.get("question") or m.get("title") or ""),
            "yes_token_id": yes_token,
            "theme": theme,
            "matched_term": matched_term,
            "discovered_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        prob = extract_yes_prob(m)
        if prob is not None:
            entry["yes_prob"] = round(prob, 4)
        vol_usd = get_volume(m)
        if vol_usd:
            entry["discovered_volume_usd"] = round(vol_usd, 0)
        end_date = m.get("endDate") or m.get("end_date_iso")
        if end_date:
            entry["end_date"] = str(end_date)[:10]
        out.append((vol, entry))
    out.sort(key=lambda pair: pair[0], reverse=True)
    return [entry for _, entry in out[:limit]]


def merge_geo_config(
    existing_path: Path,
    discovered: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Merge discovered geo markets into the geo config, deduped on
    ``slug``. Existing entries are preserved verbatim (operator may have
    hand-tuned the theme); genuinely new slugs are appended. Returns
    ``(new_doc, {"existing", "added", "skipped_duplicate"})``."""
    existing_doc = (
        yaml.safe_load(existing_path.read_text()) or {}
        if existing_path.exists() else {}
    )
    existing_markets = (
        existing_doc.get("markets", []) if isinstance(existing_doc, dict)
        else []
    )
    seen: set[str] = {
        str(m["slug"]) for m in existing_markets
        if isinstance(m, dict) and m.get("slug")
    }
    stats = {"existing": len(seen), "added": 0, "skipped_duplicate": 0}
    new_markets = list(existing_markets)
    for entry in discovered:
        slug = str(entry.get("slug") or "")
        if not slug or slug in seen:
            stats["skipped_duplicate"] += 1
            continue
        seen.add(slug)
        new_markets.append(entry)
        stats["added"] += 1
    new_doc = dict(existing_doc) if isinstance(existing_doc, dict) else {}
    new_doc["markets"] = new_markets
    return new_doc, stats


_GEO_CONFIG_HEADER = (
    "# Theme-tagged geopolitical Polymarket markets (CL-r1ep).\n"
    "#\n"
    "# Populated by scripts/discover_polymarket_markets.py --mode geo:\n"
    "# each entry's question/slug matched a playbook theme's watch_terms.\n"
    "# src/events/polymarket_signal.py polls yes_token_id each event-\n"
    "# pipeline cycle, persists YES prob to poly_market_probs, and alerts\n"
    "# on rapid shifts (Telegram HTML, like the trade cards). This config\n"
    "# is separate from configs/polymarket_markets.yaml (the macro\n"
    "# DataProvider-symbol config) on purpose — don't clobber that one.\n"
)


def _run_geo(args: argparse.Namespace) -> int:
    theme_terms = _load_theme_terms(args.playbooks)
    try:
        raw_markets = fetch_active_markets(
            limit=args.limit, order_by_volume24hr=True,
        )
    except Exception as exc:
        print(
            f"Gamma API fetch failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    discovered = discover_geo_markets(
        raw_markets, theme_terms,
        min_volume_usd=args.min_volume, limit=args.limit,
    )
    print(
        f"Gamma API returned {len(raw_markets)} markets; "
        f"{len(discovered)} matched a playbook theme at "
        f">= ${args.min_volume:,.0f} volume.",
    )
    for e in discovered:
        prob = e.get("yes_prob")
        prob_str = f" YES {prob * 100:.0f}%" if prob is not None else ""
        print(
            f"  [{e['theme']}] {e['slug']}{prob_str} "
            f"(match={e['matched_term']!r}) token={e['yes_token_id'][:16]}…",
        )

    out_path = Path(args.out or DEFAULT_GEO_CONFIG_PATH)
    new_doc, stats = merge_geo_config(out_path, discovered)
    print(
        f"Geo config: existing={stats['existing']} added={stats['added']} "
        f"skipped_duplicate={stats['skipped_duplicate']}",
    )

    if args.dry_run:
        print("\nDRY RUN — would write:")
        print("---")
        print(yaml.safe_dump(new_doc, sort_keys=False, default_flow_style=False))
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(new_doc, sort_keys=False, default_flow_style=False)
    out_path.write_text(_GEO_CONFIG_HEADER + body)
    print(f"Wrote {out_path}")
    return 0


def _run_fx(args: argparse.Namespace) -> int:
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


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()

    p = argparse.ArgumentParser(
        description=(
            "Discover Polymarket markets via the free Gamma API. "
            "--mode geo (default): theme-match against playbook "
            "watch_terms → configs/polymarket_geo_markets.yaml. "
            "--mode fx: FX/macro keywords → configs/polymarket_markets.yaml."
        ),
    )
    p.add_argument(
        "--mode", choices=("geo", "fx"), default="geo",
        help="geo: geopolitical theme discovery (default); fx: FX/macro",
    )
    p.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH),
        help="[fx mode] Path to polymarket_markets.yaml",
    )
    p.add_argument(
        "--out", default=None,
        help="[geo mode] Output config path "
             f"(default {DEFAULT_GEO_CONFIG_PATH})",
    )
    p.add_argument(
        "--playbooks", default="configs/event_playbooks.yaml",
        help="[geo mode] Playbook config supplying watch_terms",
    )
    p.add_argument(
        "--limit", type=int, default=200,
        help="Cap on markets pulled/kept (highest volume first)",
    )
    p.add_argument(
        "--min-volume", type=float, default=5_000.0,
        help="Minimum USD volume to consider a market (filter noise)",
    )
    p.add_argument(
        "--gamma-limit", type=int, default=200,
        help="[fx mode] How many markets to pull from Gamma for filtering",
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

    if args.mode == "geo":
        return _run_geo(args)
    return _run_fx(args)


if __name__ == "__main__":
    raise SystemExit(main())
