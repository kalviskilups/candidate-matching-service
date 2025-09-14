"""API models for the job-candidate matching service."""

from .schemas import (
    Candidate,
    CandidateSearchResponse,
    CandidateSearchResult,
    IngestRequest,
    IngestResponse,
    Job,
    JobSearchResponse,
    JobSearchResult,
    Location,
    SearchResponse,
    SearchScores,
)

__all__ = [
    "Location",
    "Job",
    "Candidate",
    "IngestRequest",
    "IngestResponse",
    "SearchScores",
    "JobSearchResult",
    "CandidateSearchResult",
    "SearchResponse",
    "JobSearchResponse",
    "CandidateSearchResponse",
]
