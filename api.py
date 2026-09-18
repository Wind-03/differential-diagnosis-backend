"""Backward-compatible FastAPI entry point.

Run with: uvicorn api:app --reload --port 8000
"""
from app.api.routes import app

__all__ = ["app"]
