"""Multi-agent research pipeline (parent epic CL-h986)."""

# Import side-effect: register additional fetcher adapters in the
# ingest._FETCHER_REGISTRY. The arxiv + rss adapters are registered
# inline in ingest.py; specialized adapters live in their own modules
# and self-register on import.
from src.research import polymarket as _polymarket  # noqa: F401
from src.research import social_ingest as _social  # noqa: F401
