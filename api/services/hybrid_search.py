"""
Hybrid search service wrapper.

This module provides a service wrapper around the hybrid search engine
for use in the API layer. Now uses Weaviate's native hybrid search.
"""

from typing import Any, Dict, List, Optional

from index.hybrid_search import hybrid_engine


class HybridSearchService:
    """
    Service wrapper for hybrid search functionality.

    Provides an interface between the API routes and the Weaviate
    hybrid search engine.
    """

    def __init__(self):
        """Initialize the hybrid search service."""
        self.engine = hybrid_engine

    async def initialize(self) -> None:
        """Initialize the hybrid search engine."""
        await self.engine.initialize()

    async def ingest_data(
        self, jobs: List[Dict], candidates: List[Dict]
    ) -> Dict[str, int]:
        """
        Ingest jobs and candidates data.

        Args:
            jobs: List of job dictionaries
            candidates: List of candidate dictionaries

        Returns:
            Dictionary with ingestion statistics
        """
        return await self.engine.ingest_data(jobs, candidates)

    async def search_jobs(
        self, query: str, top_k: int = 10, filters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Search jobs using hybrid retrieval.

        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional search filters

        Returns:
            Dictionary with search results and metadata
        """
        return await self.engine.search_jobs(query, top_k, filters)

    async def search_candidates(
        self, query: str, top_k: int = 10, filters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Search candidates using hybrid retrieval.

        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional search filters

        Returns:
            Dictionary with search results and metadata
        """
        return await self.engine.search_candidates(query, top_k, filters)

    async def match_candidates_for_job(
        self, job_id: str, top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Find top candidates for a specific job.

        Args:
            job_id: ID of the job to find candidates for
            top_k: Number of candidates to return

        Returns:
            Dictionary with candidate matches and metadata
        """
        return await self.engine.match_candidates_for_job(job_id, top_k)


# Global hybrid search service instance
hybrid_search_service = HybridSearchService()
