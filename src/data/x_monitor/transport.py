"""Post-fetch transports for the X monitor (CL-3j86): the official
(paid) API v2 backend, the operator-supplied CLI backend, and the
metered-read plumbing the budget governor meters against.

Split out of the former ``src/data/x_monitor.py`` monofile (CL-ikz2,
structural review 2026-07-21 §6.2.4). See the package ``__init__``
docstring for the full monitor overview.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shlex
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from src.data.x_monitor.watchlist import WatchAccount

if TYPE_CHECKING:
    from src.data.x_monitor.monitor import XMonitorConfig

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------- #

DEFAULT_USER_IDS_PATH = Path("data/x_user_ids.json")

DEFAULT_MONTHLY_CAP = 9_000

X_API_BASE = "https://api.twitter.com/2"

#: The one clear line logged when the API backend has no token.
IDLE_LINE_API = (
    "X monitor idle: set TWITTER_BEARER_TOKEN — X API basic tier required"
)

#: The one clear line logged when the CLI backend has no command.
IDLE_LINE_CLI = (
    "X monitor idle: set X_MONITOR_CLI_CMD — cookie-based CLI tools "
    "violate X ToS and risk banning the account whose cookies they use; "
    "use a burner account"
)


def _cap_from_env() -> int:
    raw = os.environ.get("X_MONITOR_MONTHLY_CAP", "").strip()
    if not raw:
        return DEFAULT_MONTHLY_CAP
    try:
        cap = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"X_MONITOR_MONTHLY_CAP={raw!r} is not an integer",
        ) from exc
    if cap <= 0:
        raise ValueError(f"X_MONITOR_MONTHLY_CAP must be positive, got {cap}")
    return cap


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """tmp + os.replace, mirroring telegram_approvals.save_offset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(payload, indent=0, sort_keys=True))
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


# --------------------------------------------------------------------- #
# Posts + transports
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class Post:
    """One fetched post, transport-agnostic."""

    id: str
    text: str
    created_at: str = ""


class TransportError(Exception):
    """Per-account fetch failure — triggers the failure cooldown."""


class RateLimitedError(TransportError):
    """HTTP 429 from the API — carries the reset time (epoch seconds)."""

    def __init__(self, reset_ts: float) -> None:
        super().__init__(f"rate limited until epoch {reset_ts:.0f}")
        self.reset_ts = reset_ts


class Transport(ABC):
    """How posts are fetched for one account. Two implementations:
    the official (paid) API and an external CLI tool. Both ship dark."""

    #: True when fetches consume the metered monthly read budget.
    metered: bool = False
    #: HTTP/API calls made so far (the monitor meters the delta).
    reads_made: int = 0
    #: The one clear line to log when the transport is not configured.
    idle_line: str = ""

    @abstractmethod
    def ready(self) -> bool:
        """True when the transport is configured and may be used."""

    def prepare(self, handles: Sequence[str]) -> None:  # noqa: B027
        """Optional pre-cycle setup (e.g. handle → id resolution).
        Deliberately a no-op default: only ApiTransport needs it."""

    @abstractmethod
    def fetch(self, account: WatchAccount, since_id: str | None) -> list[Post]:
        """Fetch recent posts for ``account``. Raises TransportError /
        RateLimitedError on failure. May return posts at or before
        ``since_id`` — the monitor re-filters client-side."""


@dataclass(frozen=True)
class XApiResponse:
    """Minimal HTTP response for the injectable API shim."""

    status_code: int
    body: str
    headers: Mapping[str, str] = field(default_factory=dict)


#: Injectable HTTP shim (url, headers, params) → response, in the same
#: spirit as src.research.ingest.HttpGet. Unit tests inject canned
#: responses; no live HTTP in unit tests.
XApiGet = Callable[[str, Mapping[str, str], Mapping[str, Any]], XApiResponse]


def _default_api_get(
    url: str,
    headers: Mapping[str, str],
    params: Mapping[str, Any],
) -> XApiResponse:
    resp = httpx.get(url, headers=dict(headers), params=dict(params), timeout=15.0)
    return XApiResponse(resp.status_code, resp.text, dict(resp.headers))


class ApiTransport(Transport):
    """Official X API v2 timeline fetcher (paid basic tier).

    Requires ``TWITTER_BEARER_TOKEN``; without it :meth:`ready` is
    False and the monitor idles. Every HTTP call (including 429s and
    the one-time ``GET /2/users/by`` handle resolution) increments
    ``reads_made`` so the monitor's budget governor can meter spend.
    Resolved user ids are cached to ``data/x_user_ids.json`` (atomic
    write) so resolution costs ~1 read per 100 handles, once ever.
    """

    metered = True
    idle_line = IDLE_LINE_API

    def __init__(
        self,
        user_ids_path: Path | str = DEFAULT_USER_IDS_PATH,
        api_get: XApiGet = _default_api_get,
        *,
        exclude_replies: bool = True,
        exclude_retweets: bool = True,
        max_results: int = 5,
    ) -> None:
        self.user_ids_path = Path(user_ids_path)
        self.api_get = api_get
        self.exclude_replies = exclude_replies
        self.exclude_retweets = exclude_retweets
        self.max_results = max_results
        self.reads_made = 0
        self._ids: dict[str, str] = self._load_id_cache()
        self._resolve_attempted: set[str] = set()

    # -- readiness ---------------------------------------------------- #

    @staticmethod
    def _token() -> str:
        return os.environ.get("TWITTER_BEARER_TOKEN", "").strip()

    def ready(self) -> bool:
        return bool(self._token())

    # -- id cache ----------------------------------------------------- #

    def _load_id_cache(self) -> dict[str, str]:
        if not self.user_ids_path.exists():
            return {}
        try:
            raw = json.loads(self.user_ids_path.read_text())
            return {str(k).lower(): str(v) for k, v in raw.items()}
        except (json.JSONDecodeError, AttributeError, TypeError) as exc:
            logger.warning(
                "user-id cache %s malformed (%s); will re-resolve",
                self.user_ids_path, type(exc).__name__,
            )
            return {}

    def _save_id_cache(self) -> None:
        _atomic_write_json(self.user_ids_path, self._ids)

    # -- API plumbing ------------------------------------------------- #

    def _call(self, url: str, params: Mapping[str, Any]) -> dict[str, Any]:
        self.reads_made += 1
        headers = {"Authorization": f"Bearer {self._token()}"}
        try:
            resp = self.api_get(url, headers, params)
        except Exception as exc:
            raise TransportError(
                f"transport failure for {url}: {type(exc).__name__}",
            ) from exc
        if resp.status_code == 429:
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            try:
                reset_ts = float(resp_headers.get("x-rate-limit-reset", ""))
            except ValueError:
                reset_ts = time.time() + 900.0
            raise RateLimitedError(reset_ts)
        if resp.status_code != 200:
            raise TransportError(f"HTTP {resp.status_code} for {url}")
        try:
            data = json.loads(resp.body)
        except json.JSONDecodeError as exc:
            raise TransportError(f"invalid JSON from {url}") from exc
        if not isinstance(data, dict):
            raise TransportError(f"unexpected payload shape from {url}")
        return data

    # -- Transport interface ------------------------------------------ #

    def prepare(self, handles: Sequence[str]) -> None:
        """Resolve any not-yet-cached handles to user ids, ≤100 per
        ``GET /2/users/by`` call. Unresolvable handles (suspended,
        renamed) are warned about once per process, not retried."""
        missing = [
            h for h in handles
            if h.lower() not in self._ids
            and h.lower() not in self._resolve_attempted
        ]
        if not missing:
            return
        for i in range(0, len(missing), 100):
            chunk = missing[i:i + 100]
            data = self._call(
                f"{X_API_BASE}/users/by",
                {"usernames": ",".join(chunk)},
            )
            # Mark attempted only on a successful call: a 429 / outage
            # during resolution must not permanently orphan the handle.
            for h in chunk:
                self._resolve_attempted.add(h.lower())
            for user in data.get("data", []):
                username = str(user.get("username", "")).lower()
                uid = str(user.get("id", ""))
                if username and uid:
                    self._ids[username] = uid
        self._save_id_cache()
        for h in missing:
            if h.lower() not in self._ids:
                logger.warning(
                    "could not resolve @%s to a user id (suspended or "
                    "renamed?) — skipping until restart", h,
                )

    def fetch(self, account: WatchAccount, since_id: str | None) -> list[Post]:
        uid = self._ids.get(account.handle.lower())
        if uid is None:
            raise TransportError(f"no user id for @{account.handle}")
        params: dict[str, Any] = {
            "max_results": self.max_results,
            "tweet.fields": "created_at",
        }
        excludes = []
        if self.exclude_replies:
            excludes.append("replies")
        if self.exclude_retweets:
            excludes.append("retweets")
        if excludes:
            params["exclude"] = ",".join(excludes)
        if since_id:
            params["since_id"] = since_id
        data = self._call(f"{X_API_BASE}/users/{uid}/tweets", params)
        posts: list[Post] = []
        for t in data.get("data", []):
            if not isinstance(t, dict) or "id" not in t:
                continue
            posts.append(Post(
                id=str(t["id"]),
                text=str(t.get("text", "")),
                created_at=str(t.get("created_at", "")),
            ))
        return posts


class CliTransport(Transport):
    """External-CLI fetcher — OPERATOR-SUPPLIED, cookie-based tools.

    HONESTY, READ BEFORE ENABLING: cookie-based CLI tools drive X's
    private endpoints with a logged-in session and therefore VIOLATE
    the X Terms of Service. The ban risk attaches to the account whose
    cookies the tool uses — use a burner account, never the operator's
    real account. Install the tool via an auditable package manager
    (not curl|sh), and expect breakage whenever X changes its private
    endpoints. This monitor never touches cookies or credentials
    itself: authentication is entirely the external tool's problem.
    The subprocess environment is passed through untouched.

    Configuration: ``X_MONITOR_CLI_CMD`` is a command template run once
    per account with ``{handle}`` substituted, e.g.::

        X_MONITOR_CLI_CMD='bird user-tweets @{handle} -n 5 --json'

    The command must print a JSON array of posts on stdout. Parsing is
    defensive: ``id``/``id_str``/``rest_id`` and ``text``/``full_text``
    field spellings are accepted; malformed entries are skipped. A
    top-level ``{"data": [...]}`` wrapper is also accepted. Without
    ``X_MONITOR_CLI_CMD`` this transport idles (ships dark, like the
    API path). since_id filtering happens in the monitor, client-side.
    """

    metered = False  # no metered reads; budget governor does not apply
    idle_line = IDLE_LINE_CLI

    def __init__(
        self,
        cmd_template: str | None = None,
        runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
        timeout_sec: float = 60.0,
    ) -> None:
        if cmd_template is None:
            cmd_template = os.environ.get("X_MONITOR_CLI_CMD", "").strip()
        self.cmd_template = cmd_template
        self.timeout_sec = timeout_sec
        self._runner = runner or self._default_runner
        self.reads_made = 0

    def _default_runner(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        # Operator-configured command; shell=False, argv fully tokenized.
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
            check=False,
        )

    def ready(self) -> bool:
        return bool(self.cmd_template)

    def fetch(self, account: WatchAccount, since_id: str | None) -> list[Post]:
        argv = [
            tok.replace("{handle}", account.handle)
            for tok in shlex.split(self.cmd_template)
        ]
        try:
            proc = self._runner(argv)
        except Exception as exc:
            raise TransportError(
                f"CLI command failed for @{account.handle}: "
                f"{type(exc).__name__}",
            ) from exc
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "").strip()[-200:]
            raise TransportError(
                f"CLI command exited {proc.returncode} for "
                f"@{account.handle}: {stderr_tail}",
            )
        return _parse_cli_posts(proc.stdout, account.handle)


def _parse_cli_posts(stdout: str, handle: str) -> list[Post]:
    """Defensively parse a JSON array of posts from CLI stdout."""
    try:
        payload = json.loads(stdout or "null")
    except json.JSONDecodeError as exc:
        raise TransportError(f"CLI stdout for @{handle} is not JSON") from exc
    if isinstance(payload, dict):
        payload = payload.get("data")
    if not isinstance(payload, list):
        raise TransportError(
            f"CLI stdout for @{handle} is not a JSON array of posts",
        )
    posts: list[Post] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue  # malformed entry — skip
        tid = entry.get("id") or entry.get("id_str") or entry.get("rest_id")
        if tid is None or not str(tid).strip():
            continue  # malformed entry — skip
        text = entry.get("text") or entry.get("full_text") or ""
        posts.append(Post(
            id=str(tid).strip(),
            text=str(text),
            created_at=str(entry.get("created_at") or ""),
        ))
    return posts


def build_transport(
    config: XMonitorConfig,
    *,
    api_get: XApiGet | None = None,
) -> Transport:
    """Select the transport from ``X_MONITOR_BACKEND`` (api | cli)."""
    backend = os.environ.get("X_MONITOR_BACKEND", "api").strip().lower() or "api"
    if backend == "api":
        return ApiTransport(
            user_ids_path=config.user_ids_path,
            api_get=api_get or _default_api_get,
            exclude_replies=config.exclude_replies,
            exclude_retweets=config.exclude_retweets,
            max_results=config.max_results,
        )
    if backend == "cli":
        return CliTransport()
    raise ValueError(
        f"unknown X_MONITOR_BACKEND {backend!r} (expected 'api' or 'cli')",
    )
