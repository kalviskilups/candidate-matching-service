"""
Search endpoints for jobs and candidates.

This module provides endpoints for searching jobs, searching candidates,
and matching candidates to specific jobs using hybrid retrieval.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.models import (
    Candidate,
    CandidateSearchResponse,
    CandidateSearchResult,
    Job,
    JobSearchResponse,
    JobSearchResult,
    SearchScores,
)
from api.services.hybrid_search import hybrid_search_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])


def get_search_filters(
    location: Optional[str] = Query(
        None, description="Filter by location (city or country)"
    ),
    remote: Optional[bool] = Query(
        None, description="Filter by remote work availability"
    ),
    skills: Optional[str] = Query(
        None, description="Comma-separated list of required skills"
    ),
    language: Optional[str] = Query(None, description="Filter by language requirement"),
) -> dict:
    """
    Create search filters dictionary from query parameters.

    Args:
        location: Location filter string
        remote: Remote work filter
        skills: Comma-separated skills string
        language: Language filter

    Returns:
        Dictionary of filters for search engines
    """
    filters = {}

    if location:
        filters["location"] = location.strip()

    if remote is not None:
        filters["remote"] = remote

    if skills:
        filters["skills"] = [
            skill.strip().lower() for skill in skills.split(",") if skill.strip()
        ]

    if language:
        filters["language"] = language.strip()

    return filters if filters else None


@router.get("/jobs", response_model=JobSearchResponse)
async def search_jobs(
    q: str = Query(..., description="Free-text search query"),
    k: int = Query(10, ge=1, le=100, description="Number of results to return"),
    filters: dict = Depends(get_search_filters),
):
    """
    Search jobs using hybrid retrieval with candidate-style queries.

    Args:
        q: Search query string
        k: Number of results to return
        filters: Search filters from query parameters

    Returns:
        JobSearchResponse with matching jobs and metadata

    Raises:
        HTTPException: If search fails
    """
    try:
        logger.info(f"Searching jobs with query: '{q}', k={k}, filters={filters}")

        results = await hybrid_search_service.search_jobs(q, k, filters)

        job_results = []
        for result in results["results"]:
            job_data = Job(**result["item"])
            scores = SearchScores(**result["scores"])

            job_result = JobSearchResult(
                job=job_data,
                scores=scores,
                matched_snippet=result.get("matched_snippet"),
            )
            job_results.append(job_result)

        return JobSearchResponse(
            results=job_results,
            total_results=results["total_results"],
            fusion_method=results["fusion_method"],
            query_time_ms=results["query_time_ms"],
        )

    except Exception as e:
        logger.error(f"Job search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Job search failed: {str(e)}")


@router.get("/candidates", response_model=CandidateSearchResponse)
async def search_candidates(
    q: str = Query(..., description="Free-text search query"),
    k: int = Query(10, ge=1, le=100, description="Number of results to return"),
    filters: dict = Depends(get_search_filters),
):
    """
    Search candidates using hybrid retrieval with job-style queries.

    Args:
        q: Search query string
        k: Number of results to return
        filters: Search filters from query parameters

    Returns:
        CandidateSearchResponse with matching candidates and metadata

    Raises:
        HTTPException: If search fails
    """
    try:
        logger.info(f"Searching candidates with query: '{q}', k={k}, filters={filters}")

        results = await hybrid_search_service.search_candidates(q, k, filters)

        candidate_results = []
        for result in results["results"]:
            candidate_data = Candidate(**result["item"])
            scores = SearchScores(**result["scores"])

            candidate_result = CandidateSearchResult(
                candidate=candidate_data,
                scores=scores,
                matched_snippet=result.get("matched_snippet"),
            )
            candidate_results.append(candidate_result)

        return CandidateSearchResponse(
            results=candidate_results,
            total_results=results["total_results"],
            fusion_method=results["fusion_method"],
            query_time_ms=results["query_time_ms"],
        )

    except Exception as e:
        logger.error(f"Candidate search failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Candidate search failed: {str(e)}"
        )


match_router = APIRouter(tags=["Search"])


@match_router.get("/match", response_model=CandidateSearchResponse)
async def match_candidates_for_job(
    job_id: str = Query(..., description="Job ID to find candidates for"),
    k: int = Query(10, ge=1, le=100, description="Number of candidates to return"),
):
    """
    Find top candidates for a specific job using the job description as query.

    Args:
        job_id: ID of the job to match candidates for
        k: Number of candidates to return

    Returns:
        CandidateSearchResponse with matching candidates

    Raises:
        HTTPException: If matching fails or job not found
    """
    try:
        logger.info(f"Matching candidates for job: {job_id}, k={k}")

        results = await hybrid_search_service.match_candidates_for_job(job_id, k)

        if "error" in results:
            raise HTTPException(status_code=404, detail=results["error"])

        candidate_results = []
        for result in results["results"]:
            candidate_data = Candidate(**result["item"])
            scores = SearchScores(**result["scores"])

            candidate_result = CandidateSearchResult(
                candidate=candidate_data,
                scores=scores,
                matched_snippet=result.get("matched_snippet"),
            )
            candidate_results.append(candidate_result)

        return CandidateSearchResponse(
            results=candidate_results,
            total_results=results["total_results"],
            fusion_method=results["fusion_method"],
            query_time_ms=results["query_time_ms"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job matching failed: {e}")
        raise HTTPException(status_code=500, detail=f"Job matching failed: {str(e)}")
