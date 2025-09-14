"""
Main entry point for the Job-Candidate Matching Service.

This module provides the main entry point for running the FastAPI application
using uvicorn server.
"""

import os

import uvicorn
from dotenv import load_dotenv

from api.app import create_app

# Load environment variables from .env file
load_dotenv()

app = create_app()

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    uvicorn.run("main:app", host=host, port=port, reload=True, log_level=log_level)
