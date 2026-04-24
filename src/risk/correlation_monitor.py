"""Correlation monitor integration — exposes src.models.correlation in risk layer."""

import logging

from src.models.correlation import CorrelationMonitor as _BaseMonitor

logger = logging.getLogger(__name__)

# Re-export
CorrelationMonitor = _BaseMonitor
