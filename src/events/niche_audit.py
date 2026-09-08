"""CL-27s0: replay-safe, append-only research outcomes, never a trading feed."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)


def persist_niche_audit(
    engine: Any,
    invocation_id: str,
    event_id: int,
    input_snapshot: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    """Commit evidence before any conditional assessment merge.

    The invocation ID is assigned before discovery, retained on retries, and
    unique per actual research run. Same-ID/different-payload writes are errors,
    not silently overwritten evidence. All errors (including commit failure)
    propagate so the caller cannot publish ideas from an unaudited invocation.
    Engine/JSON inputs are persistence boundaries. This helper never reads or
    changes geo_events, and uses no broker credentials or order tools.
    """
    if not invocation_id or event_id <= 0:
        raise ValueError("A research audit requires invocation and event identifiers")
    snapshot_json = json.dumps(input_snapshot, sort_keys=True, allow_nan=False)
    report_json = json.dumps(report, sort_keys=True, allow_nan=False)
    payload_hash = hashlib.sha256(
        json.dumps([event_id, snapshot_json, report_json]).encode("utf-8"),
    ).hexdigest()
    logger.info("niche audit: persisting event=%d invocation=%s", event_id, invocation_id)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO niche_research_audit "
                "(invocation_id, event_id, input_snapshot, report, payload_hash) "
                "VALUES (:invocation_id, :event_id, :input_snapshot, :report, :payload_hash) "
                "ON CONFLICT (invocation_id) DO NOTHING",
            ),
            {
                "invocation_id": invocation_id,
                "event_id": event_id,
                "input_snapshot": snapshot_json,
                "report": report_json,
                "payload_hash": payload_hash,
            },
        )
        stored_hash = conn.execute(
            text(
                "SELECT payload_hash FROM niche_research_audit WHERE invocation_id = :invocation_id"
            ),
            {"invocation_id": invocation_id},
        ).scalar_one()
        if stored_hash != payload_hash:
            raise ValueError("Research audit invocation ID collision")
    logger.info("niche audit: committed event=%d invocation=%s", event_id, invocation_id)
