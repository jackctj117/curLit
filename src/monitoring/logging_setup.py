"""Structured JSON logging with context propagation."""

import json
import logging
import logging.handlers
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """Format log records as newline-delimited JSON for Loki ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }

        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)

        for attr in ("trade_id", "strategy_id", "symbol", "order_id",
                      "intent_id", "doc_id", "cb", "doc_type"):
            if hasattr(record, attr):
                log_obj[attr] = getattr(record, attr)

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, default=str)


class ContextFilter(logging.Filter):
    """Thread-local filter that attaches context to log records."""

    def __init__(self) -> None:
        super().__init__()
        self._local = threading.local()

    def set_context(self, **kwargs: Any) -> None:
        if not hasattr(self._local, "context"):
            self._local.context = {}
        self._local.context.update(kwargs)

    def clear_context(self) -> None:
        if hasattr(self._local, "context"):
            self._local.context.clear()

    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(self._local, "context"):
            for k, v in self._local.context.items():
                setattr(record, k, v)
        return True


_context_filter = ContextFilter()


def get_context_filter() -> ContextFilter:
    return _context_filter


class LogContext:
    """Context manager for scoped log context propagation."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __enter__(self) -> "LogContext":
        _context_filter.set_context(**self.kwargs)
        return self

    def __exit__(self, *args: Any) -> None:
        _context_filter.clear_context()


def setup_logging(
    service_name: str,
    log_dir: Path,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> logging.Logger:
    """Configure logging for a service with console + rotating JSON file output."""
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)-20s %(message)s",
    ))
    console.addFilter(_context_filter)
    root_logger.addHandler(console)

    file_path = str(log_dir / f"{service_name}.jsonl")
    file_handler = logging.handlers.TimedRotatingFileHandler(
        file_path, when="midnight", interval=1, backupCount=30, utc=True,
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(JSONFormatter())
    file_handler.addFilter(_context_filter)
    root_logger.addHandler(file_handler)

    error_path = str(log_dir / f"{service_name}-errors.jsonl")
    error_handler = logging.handlers.TimedRotatingFileHandler(
        error_path, when="midnight", interval=1, backupCount=90, utc=True,
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(JSONFormatter())
    error_handler.addFilter(_context_filter)
    root_logger.addHandler(error_handler)

    for lib in ("urllib3", "httpx", "sqlalchemy.engine", "asyncio"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    return root_logger
