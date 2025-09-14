"""
Basic tests for the FastAPI endpoints.

These tests verify the basic functionality of the API endpoints
without requiring a full Weaviate setup.
"""

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "name" in data
    assert "endpoints" in data
    assert "Job-Candidate Matching Service" in data["name"]


def test_ingest_endpoint_validation():
    """Test the ingest endpoint with invalid data."""
    # Test with invalid JSON structure - fields missing
    response = client.post("/ingest", json={"invalid": "data"})
    assert response.status_code == 422

    # Test with invalid job data structure
    response = client.post(
        "/ingest", json={"jobs": [{"invalid": "job_data"}], "candidates": []}
    )
    assert response.status_code == 422


def test_search_jobs_validation():
    """Test job search endpoint parameter validation."""
    # Missing required query parameter
    response = client.get("/search/jobs")
    assert response.status_code == 422

    # Invalid k parameter (too large)
    response = client.get("/search/jobs?q=test&k=1000")
    assert response.status_code == 422

    # Invalid k parameter (too small)
    response = client.get("/search/jobs?q=test&k=0")
    assert response.status_code == 422


def test_search_candidates_validation():
    """Test candidate search endpoint parameter validation."""
    # Missing required query parameter
    response = client.get("/search/candidates")
    assert response.status_code == 422

    # Invalid k parameter
    response = client.get("/search/candidates?q=test&k=-1")
    assert response.status_code == 422


def test_match_endpoint_validation():
    """Test job matching endpoint parameter validation."""
    # Missing required job_id parameter
    response = client.get("/match")
    assert response.status_code == 422

    # Invalid k parameter
    response = client.get("/match?job_id=test&k=0")
    assert response.status_code == 422


def test_ingest_valid_data_structure():
    """Test ingest endpoint with valid data structure."""
    valid_job = {
        "id": "test_job_1",
        "title": "Test Developer",
        "description": "A test job description",
        "skills": ["python", "testing"],
        "location": {"city": "Test City", "country": "Test Country"},
        "employment_type": "full-time",
        "seniority": "mid",
        "language": "en",
        "salary_min": 50000,
        "salary_max": 70000,
        "remote": True,
        "created_at": "2025-09-12T00:00:00Z",
    }

    valid_candidate = {
        "id": "test_candidate_1",
        "name": "Test Candidate",
        "title": "Test Developer",
        "summary": "A test candidate summary",
        "skills": ["python", "testing"],
        "years_experience": 5,
        "location": {"city": "Test City", "country": "Test Country"},
        "languages": ["en"],
        "open_to_remote": True,
        "desired_salary": 60000,
        "right_to_work_eu": True,
        "last_updated": "2025-09-12T00:00:00Z",
    }

    # Test with valid structure (will likely fail at runtime due to dependencies)
    # but should pass validation
    response = client.post(
        "/ingest", json={"jobs": [valid_job], "candidates": [valid_candidate]}
    )

    # Expecting either 200 (success) or 500 (runtime error)
    # but not 422 (validation error)
    assert response.status_code in [200, 500]


if __name__ == "__main__":
    pytest.main([__file__])
