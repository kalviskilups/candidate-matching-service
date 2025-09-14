"""
Data ingestion endpoints.

This module provides endpoints for loading jobs and candidates data
into the search indexes.
"""

import json
import logging

from fastapi import APIRouter, HTTPException, Query

from api.models import Candidate, IngestRequest, IngestResponse, Job
from api.services.hybrid_search import hybrid_search_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Data Ingestion"])


@router.post("/", response_model=IngestResponse)
async def ingest_data(request: IngestRequest):
    """
    Ingest jobs and candidates data into the search indexes.

    Args:
        request: IngestRequest containing jobs and candidates data

    Returns:
        IngestResponse with ingestion statistics

    Raises:
        HTTPException: If ingestion fails
    """
    try:
        logger.info(
            f"Ingesting {len(request.jobs)} jobs and "
            f"{len(request.candidates)} candidates"
        )

        jobs_dict = [job.model_dump() for job in request.jobs]
        candidates_dict = [candidate.model_dump() for candidate in request.candidates]

        stats = await hybrid_search_service.ingest_data(jobs_dict, candidates_dict)

        return IngestResponse(**stats)

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Data ingestion failed: {str(e)}")


@router.post("/from-json", response_model=IngestResponse)
async def ingest_from_files(
    jobs_file: str = Query("data/jobs.json", description="Path to jobs JSON file"),
    candidates_file: str = Query(
        "data/candidates.json", description="Path to candidates JSON file"
    ),
):
    """
    Load data from JSON files (utility endpoint for testing).

    Args:
        jobs_file: Path to jobs JSON file
        candidates_file: Path to candidates JSON file

    Returns:
        IngestResponse with ingestion statistics

    Raises:
        HTTPException: If file loading or ingestion fails
    """
    try:
        logger.info(f"Loading data from files: {jobs_file}, {candidates_file}")

        jobs_data = []
        try:
            with open(jobs_file, "r", encoding="utf-8") as f:
                jobs_json = json.load(f)
                jobs_data = [Job(**job) for job in jobs_json]
        except FileNotFoundError:
            logger.warning(f"Jobs file {jobs_file} not found, using empty list")
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error loading jobs file: {str(e)}"
            )

        candidates_data = []
        try:
            with open(candidates_file, "r", encoding="utf-8") as f:
                candidates_json = json.load(f)
                candidates_data = [
                    Candidate(**candidate) for candidate in candidates_json
                ]
        except FileNotFoundError:
            logger.warning(
                f"Candidates file {candidates_file} not found, using empty list"
            )
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error loading candidates file: {str(e)}"
            )

        request = IngestRequest(jobs=jobs_data, candidates=candidates_data)
        return await ingest_data(request)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")
