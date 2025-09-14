"""API route modules."""

from .ingest import router as ingest_router
from .search import router as search_router

__all__ = [
    "ingest_router",
    "search_router",
]
