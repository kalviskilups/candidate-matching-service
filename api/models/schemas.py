"""
Data models for the job-candidate matching service.

This module contains Pydantic models for jobs, candidates, search requests,
and search responses with proper typing and validation.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Location(BaseModel):
    """Geographic location model."""

    city: str = Field(..., description="City name")
    country: str = Field(..., description="Country name")


class Job(BaseModel):
    """Job posting model with all required fields."""

    id: str = Field(..., description="Unique job identifier")
    title: str = Field(..., description="Job title")
    description: str = Field(..., description="Detailed job description")
    skills: List[str] = Field(..., description="Required skills")
    location: Location = Field(..., description="Job location")
    employment_type: str = Field(
        ..., description="Employment type (full-time, part-time, contract)"
    )
    seniority: str = Field(..., description="Seniority level (junior, mid, senior)")
    language: str = Field(..., description="Primary language requirement")
    salary_min: int = Field(..., description="Minimum salary in local currency")
    salary_max: int = Field(..., description="Maximum salary in local currency")
    remote: bool = Field(..., description="Whether remote work is allowed")
    created_at: datetime = Field(..., description="Job posting creation timestamp")


class Candidate(BaseModel):
    """Candidate profile model with all suggested fields."""

    id: str = Field(..., description="Unique candidate identifier")
    name: str = Field(..., description="Candidate full name")
    title: str = Field(..., description="Current or desired job title")
    summary: str = Field(..., description="Professional summary")
    skills: List[str] = Field(..., description="Technical and professional skills")
    years_experience: int = Field(..., description="Years of professional experience")
    location: Location = Field(..., description="Current location")
    languages: List[str] = Field(..., description="Spoken languages")
    open_to_remote: bool = Field(..., description="Willingness to work remotely")
    desired_salary: int = Field(..., description="Desired salary in local currency")
    right_to_work_eu: bool = Field(..., description="EU work authorization status")
    last_updated: datetime = Field(..., description="Profile last update timestamp")


class IngestRequest(BaseModel):
    """Request model for data ingestion endpoint."""

    jobs: List[Job] = Field(
        default_factory=list, description="List of job postings to ingest"
    )
    candidates: List[Candidate] = Field(
        default_factory=list, description="List of candidates to ingest"
    )

    model_config = {"extra": "forbid"}


class IngestResponse(BaseModel):
    """Response model for data ingestion endpoint."""

    jobs_loaded: int = Field(..., description="Number of jobs successfully loaded")
    candidates_loaded: int = Field(
        ..., description="Number of candidates successfully loaded"
    )
    jobs_index_size: int = Field(..., description="Total jobs in index after loading")
    candidates_index_size: int = Field(
        ..., description="Total candidates in index after loading"
    )


class SearchScores(BaseModel):
    """Score breakdown for hybrid search results."""

    bm25: float = Field(..., description="BM25 sparse search score")
    vector: float = Field(..., description="Dense vector similarity score")
    fused: float = Field(..., description="Final fused score")


class JobSearchResult(BaseModel):
    """Single job search result with scores and metadata."""

    job: Job = Field(..., description="Job posting data")
    scores: SearchScores = Field(..., description="Search score breakdown")
    matched_snippet: Optional[str] = Field(
        None, description="Relevant text snippet for explainability"
    )


class CandidateSearchResult(BaseModel):
    """Single candidate search result with scores and metadata."""

    candidate: Candidate = Field(..., description="Candidate profile data")
    scores: SearchScores = Field(..., description="Search score breakdown")
    matched_snippet: Optional[str] = Field(
        None, description="Relevant text snippet for explainability"
    )


class SearchResponse(BaseModel):
    """Base search response with metadata."""

    total_results: int = Field(..., description="Total number of matching results")
    fusion_method: str = Field(..., description="Fusion method used")
    query_time_ms: float = Field(
        ..., description="Query execution time in milliseconds"
    )


class JobSearchResponse(SearchResponse):
    """Response model for job search endpoint."""

    results: List[JobSearchResult] = Field(..., description="List of matching jobs")


class CandidateSearchResponse(SearchResponse):
    """Response model for candidate search endpoint."""

    results: List[CandidateSearchResult] = Field(
        ..., description="List of matching candidates"
    )
