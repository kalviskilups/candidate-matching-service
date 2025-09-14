"""
Hybrid search engine using Weaviate's native hybrid search capabilities.

This module uses Weaviate's built-in hybrid search.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from api.services.embedding import create_searchable_text, embedding_service

from .search_engine import WeaviateHybridSearchEngine

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class HybridSearchService:
    """
    Hybrid search service that wraps WeaviateHybridSearchEngine with additional functionality.
    """

    def __init__(self):
        """Initialize the hybrid search service."""
        fusion_method = os.getenv("FUSION_METHOD", "relative_score")
        self.engine = WeaviateHybridSearchEngine(fusion_method=fusion_method)
        self.fusion_method = fusion_method

    async def initialize(self) -> None:
        """Initialize the Weaviate hybrid search engine."""
        logger.info("Initializing Weaviate hybrid search engine")

        if not embedding_service.is_initialized():
            logger.info("Initializing embedding service...")
            await embedding_service.initialize()

        await self.engine.initialize()

        logger.info(
            f"Weaviate hybrid search engine initialized with "
            f"{self.fusion_method} fusion"
        )

    async def ingest_data(
        self, jobs: List[Dict], candidates: List[Dict]
    ) -> Dict[str, int]:
        """
        Ingest jobs and candidates data into Weaviate.

        Args:
            jobs: List of job dictionaries
            candidates: List of candidate dictionaries

        Returns:
            Dictionary with ingestion statistics
        """
        start_time = time.time()
        logger.info(f"Ingesting {len(jobs)} jobs and {len(candidates)} candidates")

        if not embedding_service.is_initialized():
            await embedding_service.initialize()

        jobs_indexed = 0
        candidates_indexed = 0

        if jobs:
            job_texts = [create_searchable_text(job) for job in jobs]
            logger.info(f"Generating embeddings for {len(job_texts)} jobs...")
            job_embeddings = await embedding_service.encode_batch(job_texts)
            jobs_with_embeddings = list(zip(jobs, job_embeddings))
            await self.engine.index_jobs(jobs_with_embeddings)
            jobs_indexed = len(jobs)

        if candidates:
            candidate_texts = [
                create_searchable_text(candidate) for candidate in candidates
            ]
            logger.info(
                f"Generating embeddings for {len(candidate_texts)} candidates..."
            )
            candidate_embeddings = await embedding_service.encode_batch(candidate_texts)
            candidates_with_embeddings = list(zip(candidates, candidate_embeddings))
            await self.engine.index_candidates(candidates_with_embeddings)
            candidates_indexed = len(candidates)

        ingestion_time = time.time() - start_time
        logger.info(f"Data ingestion completed in {ingestion_time:.2f}s")

        return {
            "jobs_loaded": jobs_indexed,
            "candidates_loaded": candidates_indexed,
            "jobs_index_size": jobs_indexed,
            "candidates_index_size": candidates_indexed,
        }

    async def search_jobs(
        self, query: str, top_k: int = 10, filters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Search jobs using hybrid search.

        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional search filters

        Returns:
            Dictionary with search results and metadata
        """
        start_time = time.time()

        logger.info(f"Searching jobs with query: '{query}', k={top_k}")

        results = await self.engine.search_jobs(query, top_k, filters)

        formatted_results = []
        for job_data, scores, snippet in results:
            formatted_results.append(
                {
                    "item": job_data,
                    "scores": scores,
                    "matched_snippet": snippet,
                }
            )

        query_time = (time.time() - start_time) * 1000

        logger.info(
            f"Job search completed in {query_time:.2f}ms, "
            f"found {len(formatted_results)} results"
        )

        return {
            "results": formatted_results,
            "total_results": len(formatted_results),
            "fusion_method": self.fusion_method,
            "query_time_ms": query_time,
        }

    async def search_candidates(
        self, query: str, top_k: int = 10, filters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Search candidates using hybrid search.

        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional search filters

        Returns:
            Dictionary with search results and metadata
        """
        start_time = time.time()

        logger.info(f"Searching candidates with query: '{query}', k={top_k}")

        results = await self.engine.search_candidates(query, top_k, filters)

        formatted_results = []
        for candidate_data, scores, snippet in results:
            formatted_results.append(
                {
                    "item": candidate_data,
                    "scores": scores,
                    "matched_snippet": snippet,
                }
            )

        query_time = (time.time() - start_time) * 1000

        logger.info(
            f"Candidate search completed in {query_time:.2f}ms, "
            f"found {len(formatted_results)} results"
        )

        return {
            "results": formatted_results,
            "total_results": len(formatted_results),
            "fusion_method": self.fusion_method,
            "query_time_ms": query_time,
        }

    async def match_candidates_for_job(
        self, job_id: str, top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Find top candidates for a specific job using hybrid search.

        Args:
            job_id: ID of the job to find candidates for
            top_k: Number of candidates to return

        Returns:
            Dictionary with candidate matches and metadata
        """
        start_time = time.time()

        logger.info(f"Matching candidates for job {job_id}, k={top_k}")

        try:
            results = await self.engine.match_candidates_for_job(job_id, top_k)

            formatted_results = []
            for candidate_data, scores, snippet in results:
                formatted_results.append(
                    {
                        "item": candidate_data,
                        "scores": scores,
                        "matched_snippet": snippet,
                    }
                )

            query_time = (time.time() - start_time) * 1000

            logger.info(
                f"Job matching completed in {query_time:.2f}ms, "
                f"found {len(formatted_results)} candidates for job {job_id}"
            )

            return {
                "results": formatted_results,
                "total_results": len(formatted_results),
                "fusion_method": self.fusion_method,
                "query_time_ms": query_time,
            }

        except Exception as e:
            logger.error(f"Job matching failed for job {job_id}: {e}")

            return {
                "results": [],
                "total_results": 0,
                "fusion_method": self.fusion_method,
                "query_time_ms": (time.time() - start_time) * 1000,
                "error": f"Job matching failed: {str(e)}",
            }


hybrid_engine = HybridSearchService()
