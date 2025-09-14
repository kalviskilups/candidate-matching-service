"""
FastAPI application factory.

This module creates and configures the FastAPI application with all
routes, middleware, and lifecycle management.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import ingest_router, search_router
from .routes.search import match_router
from .services.hybrid_search import hybrid_search_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown tasks.

    Initializes the hybrid search service on startup and cleans up on shutdown.
    """
    try:
        logger.info("Starting up application...")
        await hybrid_search_service.initialize()
        logger.info("Application startup complete")

        yield

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        logger.info("Shutting down application...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Job-Candidate Matching Service",
        description="Service for matching jobs and candidates using hybrid search",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_model=dict)
    async def root():
        return {
            "name": "Job-Candidate Matching Service",
            "description": (
                "Service for matching jobs and candidates using hybrid search"
            ),
            "endpoints": {
                "POST /ingest": "Load jobs and candidates data",
                "GET /search/jobs": "Search jobs with candidate-style queries",
                "GET /search/candidates": "Search candidates with job-style queries",
                "GET /match": "Find candidates for a specific job",
            },
        }

    app.include_router(ingest_router)
    app.include_router(search_router)
    app.include_router(match_router)

    return app
