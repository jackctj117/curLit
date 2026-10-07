"""Durable catch-up for the OANDA transaction stream (CL-pksi, CL-vj74).

The transaction stream only carries transactions created while it is
connected. Before this module a fill that happened during a disconnect (a
network blip, a stall, an engine restart) never reached ``OrderManager.on_fill``
— only the ~300 s position poll noticed the position change, unattributed, and
an emergency-order fence waited on that poll.

Now:

* :class:`SqlTransactionCheckpointStore` keeps, per account, the id of the last
  transaction the engine has DURABLY handled (migration 028). It only moves
  forward.
* On every (re)connect :meth:`FillStreamCatchUp.catch_up` fetches
  ``/transactions/sinceid?id=<checkpoint>`` through the broker's transport and
  feeds each ORDER_FILL through the same OMS fill path (deduplicated by venue
  transaction id, and against the journal after a restart). The checkpoint is
  advanced only after a fill's ORDER_FILLED row is journaled.
* :meth:`FillStreamCatchUp.handle_live` does the same for streamed fills.

Ordering rules that keep the checkpoint honest:

* No network call runs inside a DB transaction: every page is fetched first,
  each checkpoint write is its own short transaction.
* After any failure (replay error, unjournaled fill, truncated replay) the
  checkpoint is FROZEN for the rest of the connection: advancing it past a
  newer live fill would skip the unreplayed ones. The next reconnect replays
  from the last good checkpoint and unfreezes on success. The stream itself
  stays up — a failed replay never kills it.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import text

logger = logging.getLogger(__name__)

#: Transactions processed per replay page. OANDA's ``sinceid`` endpoint takes
#: no page-size parameter, so the cap is applied client-side (lowest ids
#: first; the cursor moves to the last one processed). 500 keeps one page's
#: journal work to a few seconds on the worker thread while far exceeding the
#: practice account's transaction rate over a typical disconnect (minutes).
REPLAY_PAGE_SIZE = 500

#: Pages per catch-up (so at most 10,000 transactions per reconnect). A longer
#: backlog is left to the NEXT reconnect (the checkpoint stays frozen at the
#: last replayed id meanwhile) and to the position-poll backstop — a bounded
#: replay must not hold the stream down indefinitely.
REPLAY_MAX_PAGES = 20


@dataclass(frozen=True)
class TransactionPage:
    """One bounded ``sinceid`` page (CL-pksi catch-up)."""

    #: Normalized ORDER_FILL dicts (same shape as the stream yields), in
    #: ascending transaction-id order.
    fills: list[dict[str, Any]]
    #: Highest transaction id (any type) covered by this page; None = empty.
    max_id: int | None
    #: The account's ``lastTransactionID`` at fetch time, if reported.
    last_transaction_id: int | None
    #: True when the page was cut at the client-side cap (more remain).
    truncated: bool


class TransactionHistorySource(Protocol):
    """Broker surface the catch-up needs (implemented by OandaBroker)."""

    def transactions_since(self, since_id: int, *, limit: int) -> TransactionPage: ...

    def last_transaction_id(self) -> int: ...


class FillProcessor(Protocol):
    """OMS surface the catch-up needs (implemented by OrderManager)."""

    def process_fill(
        self,
        fill: dict[str, Any],
        *,
        check_journal: bool = False,
        defer_on_lookup_failure: bool = False,
    ) -> Any: ...

    def deferred_fills(self) -> list[dict[str, Any]]: ...


class TransactionCheckpointStore(Protocol):
    def load(self, account_id: str) -> int | None: ...

    def advance(self, account_id: str, transaction_id: int) -> None: ...

    def rewind(self, account_id: str, transaction_id: int) -> None: ...


class InMemoryTransactionCheckpointStore:
    """Process-local store for tests (survives nothing)."""

    def __init__(self) -> None:
        self._rows: dict[str, int] = {}
        self.writes: list[tuple[str, int]] = []

    def load(self, account_id: str) -> int | None:
        return self._rows.get(account_id)

    def advance(self, account_id: str, transaction_id: int) -> None:
        assert transaction_id >= 0, "transaction ids are non-negative"
        self.writes.append((account_id, transaction_id))
        if transaction_id > self._rows.get(account_id, -1):
            self._rows[account_id] = transaction_id

    def rewind(self, account_id: str, transaction_id: int) -> None:
        assert transaction_id >= 0, "transaction ids are non-negative"
        self.writes.append((account_id, -transaction_id - 1))  # marks a rewind
        current = self._rows.get(account_id)
        if current is None or current > transaction_id:
            self._rows[account_id] = transaction_id


class SqlTransactionCheckpointStore:
    """``oanda_stream_checkpoint`` (migration 028); Postgres or sqlite."""

    def __init__(self, engine: Any) -> None:
        assert engine is not None, "SqlTransactionCheckpointStore requires an engine"
        self.engine = engine

    def load(self, account_id: str) -> int | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT last_transaction_id FROM oanda_stream_checkpoint WHERE account_id = :a"
                ),
                {"a": account_id},
            ).fetchone()
        return None if row is None else int(row[0])

    def advance(self, account_id: str, transaction_id: int) -> None:
        """Move the cursor forward (never back) in one short transaction."""
        assert transaction_id >= 0, "transaction ids are non-negative"
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO oanda_stream_checkpoint "
                    "(account_id, last_transaction_id, updated_at) "
                    "VALUES (:a, :t, :u) "
                    "ON CONFLICT (account_id) DO UPDATE SET "
                    "last_transaction_id = excluded.last_transaction_id, "
                    "updated_at = excluded.updated_at "
                    "WHERE oanda_stream_checkpoint.last_transaction_id "
                    "< excluded.last_transaction_id"
                ),
                {"a": account_id, "t": int(transaction_id), "u": datetime.now(UTC)},
            )

    def rewind(self, account_id: str, transaction_id: int) -> None:
        """The ONLY backwards move (integration review r3): lower the cursor
        to ``transaction_id`` when it is above it (or create the row), so a
        replay after a restart re-fetches a deferred fill. One short
        transaction, no network inside."""
        assert transaction_id >= 0, "transaction ids are non-negative"
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO oanda_stream_checkpoint "
                    "(account_id, last_transaction_id, updated_at) "
                    "VALUES (:a, :t, :u) "
                    "ON CONFLICT (account_id) DO UPDATE SET "
                    "last_transaction_id = excluded.last_transaction_id, "
                    "updated_at = excluded.updated_at "
                    "WHERE oanda_stream_checkpoint.last_transaction_id "
                    "> excluded.last_transaction_id"
                ),
                {"a": account_id, "t": int(transaction_id), "u": datetime.now(UTC)},
            )


class ReplayIncompleteError(RuntimeError):
    """A replayed fill could not be journaled — stop before advancing."""


def _txn_id(fill: dict[str, Any]) -> int | None:
    try:
        return int(str(fill.get("transaction_id") or ""))
    except ValueError:
        return None


class FillStreamCatchUp:
    """Checkpointed replay + live fill handling for one account (CL-pksi)."""

    def __init__(
        self,
        source: TransactionHistorySource,
        oms: FillProcessor,
        store: TransactionCheckpointStore,
        account_id: str,
        *,
        page_size: int = REPLAY_PAGE_SIZE,
        max_pages: int = REPLAY_MAX_PAGES,
    ) -> None:
        assert account_id, "catch-up needs an account id"
        assert page_size > 0 and max_pages > 0, "replay bounds must be positive"
        self.source = source
        self.oms = oms
        self.store = store
        self.account_id = account_id
        self.page_size = page_size
        self.max_pages = max_pages
        # Serializes catch-up and live handling (both run on worker threads).
        self._lock = threading.Lock()
        # True = the checkpoint must not advance until a replay succeeds.
        # Starts frozen: nothing may advance before the first catch-up ran.
        self._frozen = True
        # Integration review r3: a durable rewind still owed for a deferred
        # fill (target = its transaction id - 1), retried on every live fill
        # and at every catch-up until the store accepts it. While pending the
        # checkpoint may not advance at all.
        self._rewind_to: int | None = None
        # Checkpoint freshly seeded from lastTransactionID this session and no
        # fill seen yet: the first streamed fill bounds it (seed <= id - 1).
        self._seeded_at: int | None = None

    @property
    def frozen(self) -> bool:
        return self._frozen

    def _advance(self, txn_id: int) -> None:
        """Checkpoint write — a DB transaction with no network inside. Never
        past a still-deferred fill, and not at all while a rewind is owed."""
        if self._rewind_to is not None:
            logger.warning(
                "stream checkpoint: NOT advancing %s to %d — rewind to %d still pending",
                self.account_id,
                txn_id,
                self._rewind_to,
            )
            return
        deferred = [i for i in (_txn_id(f) for f in self.oms.deferred_fills()) if i is not None]
        if deferred and txn_id >= min(deferred):
            logger.warning(
                "stream checkpoint: NOT advancing %s to %d — fill %d still deferred",
                self.account_id,
                txn_id,
                min(deferred),
            )
            return
        logger.debug("stream checkpoint: advancing %s to %d", self.account_id, txn_id)
        self.store.advance(self.account_id, txn_id)

    def _request_rewind(self, target: int, why: str) -> None:
        """Owe a durable rewind to ``target`` (kept at the lowest target) and
        try it now; a failure is logged and retried later (never raises)."""
        target = max(target, 0)
        if self._rewind_to is None or target < self._rewind_to:
            self._rewind_to = target
        logger.warning(
            "stream checkpoint: REWIND for %s requested for %s to %d (%s)",
            why,
            self.account_id,
            self._rewind_to,
            "retrying" if self._rewind_to != target else "new",
        )
        try:
            self._apply_rewind()
        except Exception:
            logger.exception(
                "stream checkpoint: rewind of %s to %d FAILED — retried on the next "
                "fill / catch-up; checkpoint will not advance until it succeeds",
                self.account_id,
                self._rewind_to,
            )

    def _apply_rewind(self) -> None:
        """Perform an owed rewind (raises on store failure)."""
        if self._rewind_to is None:
            return
        target = self._rewind_to
        self.store.rewind(self.account_id, target)
        logger.warning(
            "stream checkpoint: REWOUND %s to %d for a deferred fill (replay re-fetches it)",
            self.account_id,
            target,
        )
        self._rewind_to = None

    def catch_up(self) -> int:
        """Replay fills missed since the checkpoint. Returns the number of
        fills newly processed, or -1 when the replay failed (checkpoint left
        where the last journaled fill put it; frozen until the next try).
        Never raises (the stream must stay alive)."""
        with self._lock:
            self._frozen = True
            try:
                return self._catch_up_locked()
            except Exception:
                logger.exception(
                    "OANDA transaction catch-up FAILED for %s — checkpoint not advanced "
                    "past the last journaled fill; stream continues, retry on next "
                    "reconnect (position poll remains the backstop)",
                    self.account_id,
                )
                return -1

    def _catch_up_locked(self) -> int:
        # 1. A durable rewind owed for a deferred fill comes first: until it
        #    lands, a restart could skip that fill, so nothing else proceeds.
        self._apply_rewind()
        # 2. Drain deferred fills from memory, independent of the venue page
        #    (their ids may be at or below the checkpoint). Journal-checked:
        #    appended exactly once, listeners not re-fired.
        drained = 0
        for fill in self.oms.deferred_fills():
            outcome = self.oms.process_fill(fill, check_journal=True)
            if not outcome.durable:
                msg = f"deferred fill {fill.get('transaction_id')} was not journaled"
                raise ReplayIncompleteError(msg)
            if outcome.newly_processed:
                drained += 1  # appended now (a row that already existed is not)
        if drained:
            logger.warning(
                "OANDA transaction catch-up for %s: drained %d deferred fill(s)",
                self.account_id,
                drained,
            )
        checkpoint = self.store.load(self.account_id)
        if checkpoint is None:
            seed = int(self.source.last_transaction_id())
            logger.warning(
                "OANDA transaction catch-up: no checkpoint for %s — seeding at the "
                "account's last transaction %d (earlier fills are NOT replayed; the "
                "first streamed fill lowers it if it is not above the seed)",
                self.account_id,
                seed,
            )
            self._advance(seed)
            self._seeded_at = seed
            self._frozen = False
            return drained
        self._seeded_at = None
        logger.info(
            "OANDA transaction catch-up: replaying %s since transaction %d",
            self.account_id,
            checkpoint,
        )
        cursor = checkpoint
        replayed = drained  # deferred fills journaled above count as replayed
        duplicates = 0
        for page_no in range(1, self.max_pages + 1):
            # Network read with NO DB transaction open.
            page = self.source.transactions_since(cursor, limit=self.page_size)
            for fill in page.fills:
                fid = _txn_id(fill)
                if fid is None or fid <= cursor:
                    logger.warning("catch-up: skipping fill with bad/old id %r", fill)
                    continue
                outcome = self.oms.process_fill(fill, check_journal=True)
                if not outcome.durable:
                    msg = f"replayed fill {fid} was not journaled"
                    raise ReplayIncompleteError(msg)
                if outcome.newly_processed:
                    replayed += 1
                else:
                    duplicates += 1
                # Only AFTER the fill is journaled (or known journaled).
                self._advance(fid)
                cursor = fid
            if page.max_id is not None and page.max_id > cursor:
                # Every fill up to max_id in this page is journaled; the rest
                # are non-fill transactions — safe to skip past.
                self._advance(page.max_id)
                cursor = page.max_id
            done = page.max_id is None or not page.truncated
            if done:
                self._frozen = False
                log = logger.warning if replayed else logger.info
                log(
                    "OANDA transaction catch-up for %s: REPLAYED %d missed fill(s) "
                    "(%d duplicate(s) skipped) over %d page(s); checkpoint now %d",
                    self.account_id,
                    replayed,
                    duplicates,
                    page_no,
                    cursor,
                )
                return replayed
        logger.critical(
            "OANDA transaction catch-up for %s TRUNCATED after %d pages (%d fills "
            "replayed, checkpoint %d) — remaining backlog waits for the next "
            "reconnect / position poll; checkpoint frozen until then",
            self.account_id,
            self.max_pages,
            replayed,
            cursor,
        )
        return replayed

    def handle_live(self, fill: dict[str, Any]) -> None:
        """Process one streamed fill and advance the checkpoint past it when
        it is journaled and no earlier gap is outstanding. Never raises."""
        with self._lock:
            fid = _txn_id(fill)
            if self._rewind_to is not None:
                self._request_rewind(self._rewind_to, "a pending deferral (retry)")
            if self._seeded_at is not None and fid is not None:
                if fid <= self._seeded_at:
                    # A fill already buffered when the checkpoint was seeded:
                    # the seed may not cover it, or a deferral of it would be
                    # stranded behind the checkpoint across a restart.
                    self._request_rewind(fid - 1, f"seed above first streamed fill {fid}")
                self._seeded_at = None
            try:
                # Durable dedup on the live path too: a replay can evict a
                # newer fill id from the OMS's bounded in-memory set (a sync
                # fill recorded before a long replay), so its buffered live
                # copy must be checked against the journal. The OMS only
                # queries the journal when the id is not remembered. If that
                # lookup fails the journal state is UNKNOWN: the OMS never
                # appends it (no possible duplicate row), returns durable=False
                # (checkpoint frozen below) and the next replay — which checks
                # the journal first — journals it exactly once if missing.
                outcome = self.oms.process_fill(
                    fill, check_journal=True, defer_on_lookup_failure=True
                )
            except Exception:
                self._frozen = True
                logger.exception("stream fill %s handling failed — checkpoint frozen", fid or "?")
                return
            if getattr(outcome, "deferred", False) and fid is not None:
                # Journal state unknown: the durable boundary must sit below
                # this fill so even a restart (which loses the in-memory
                # deferred map) replays it.
                self._request_rewind(fid - 1, f"deferred fill {fid}")
            if not outcome.durable:
                if not self._frozen:
                    logger.error(
                        "stream fill %s not journaled — checkpoint frozen until the "
                        "next reconnect replays it",
                        fid or "?",
                    )
                self._frozen = True
                return
            if self._frozen or fid is None:
                return
            try:
                self._advance(fid)
            except Exception:
                # Monotonic cursor: a later successful write supersedes this one.
                logger.exception("stream checkpoint write failed at %d", fid)
