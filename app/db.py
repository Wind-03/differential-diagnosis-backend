"""Neon/PostgreSQL database setup and small persistence helpers."""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


_engine = None
SessionLocal = None


def _database_url() -> str:
    if not settings.database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured. Set DATABASE_URL to your Neon PostgreSQL connection string."
        )
    return settings.database_url


def get_engine():
    global _engine
    if _engine is None:
        url = _database_url()
        if url.startswith(("postgres://", "postgresql://")) and "sslmode=" not in url:
            url += "&sslmode=require" if "?" in url else "?sslmode=require"
        _engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
        )
    return _engine


def get_session_factory():
    global SessionLocal
    if SessionLocal is None:
        SessionLocal = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            expire_on_commit=False,
        )
    return SessionLocal


def get_db() -> Iterator[Session]:
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Iterator[Session]:
    db = get_session_factory()()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create missing tables.

    This is convenient for the project deployment. For an existing production
    database with schema changes, apply the SQL in migrations/ explicitly.
    """
    # Import models only after Base exists; this also avoids a circular import.
    from app import models  # noqa: F401

    Base.metadata.create_all(get_engine())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def save_diagnosis(
    db: Session,
    *,
    user_id: str,
    session_id: str | None,
    patient_data: dict[str, Any],
    result: dict[str, Any],
) -> int:
    from app.models import DiagnosisRecord

    row = DiagnosisRecord(
        id=None,
        user_id=user_id,
        session_id=session_id,
        patient_data=_json_safe(patient_data),
        result=_json_safe(result),
        created_at=utcnow(),
    )
    db.add(row)
    db.flush()
    return row.id


def save_feedback(
    db: Session,
    *,
    user_id: str,
    session_id: str,
    diagnosis_id: int | None = None,
    doctor_correction: str | None = None,
    rating: int | None = None,
    comment: str | None = None,
) -> int:
    from app.models import FeedbackRecord

    row = FeedbackRecord(
        user_id=user_id,
        session_id=session_id,
        diagnosis_id=diagnosis_id,
        doctor_correction=doctor_correction,
        rating=rating,
        comment=comment,
        created_at=utcnow(),
    )
    db.add(row)
    db.flush()
    return row.id


def get_idempotent_response(db: Session, *, user_id: str, key: str) -> dict[str, Any] | None:
    from app.models import IdempotencyRecord

    row = db.query(IdempotencyRecord).filter(
        IdempotencyRecord.user_id == user_id,
        IdempotencyRecord.key == key,
    ).first()
    return json.loads(row.response_json) if row else None


def save_idempotent_response(db: Session, *, user_id: str, key: str, response: dict[str, Any]) -> None:
    from app.models import IdempotencyRecord

    row = IdempotencyRecord(
        id=None,
        user_id=user_id,
        key=key,
        response_json=json.dumps(_json_safe(response), separators=(",", ":")),
        created_at=utcnow(),
    )
    db.add(row)
    db.flush()
