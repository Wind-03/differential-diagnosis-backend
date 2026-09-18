"""FastAPI routes with authentication, ownership checks and Neon persistence."""
from __future__ import annotations

import json
import uuid
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.agent.factory import create_conversational_agent
from app.agent.history import reset_current_user, serialize_history, set_current_user
from app.auth import create_access_token, create_user, get_current_user, require_practitioner, verify_password
from app.config import settings
from app.db import get_db, get_idempotent_response, init_db, save_diagnosis, save_feedback, save_idempotent_response
from app.models import AuditEvent, ChatSession, DiagnosisRecord, IdempotencyRecord, User
from app.diagnosis.service import DifferentialDiagnosis
from app.nlp.extractor import text_to_features
from app.rate_limit import enforce_rate_limit
from app.schemas import *


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_for_json(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def audit(
    db: Session,
    request: Request,
    user_id: str | None,
    action: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    event = AuditEvent(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        event_metadata=metadata,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    db.add(event)
    db.commit()


def require_idempotency_key(request: Request) -> str | None:
    key = request.headers.get("Idempotency-Key")
    if key is None:
        return None
    key = key.strip()
    if not key or len(key) > 128:
        raise HTTPException(status_code=400, detail="Idempotency-Key must be between 1 and 128 characters.")
    return key


def create_app(diagnosis_system: DifferentialDiagnosis | None = None) -> FastAPI:
    app = FastAPI(
        title="Malaria/COVID-19 Differential Diagnosis API",
        description="Clinical decision-support tool — not a substitute for clinical judgement or diagnostic testing.",
        version="1.2.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
    )

    init_db()
    system = diagnosis_system or DifferentialDiagnosis()
    app.state.diagnosis_system = system
    try:
        app.state.conversational_agent = create_conversational_agent(system)
    except RuntimeError:
        app.state.conversational_agent = None

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cache-Control"] = "no-store" if request.url.path.startswith(("/auth", "/chat", "/diagnose")) else "no-cache"
        return response

    @app.get("/health", response_model=HealthResponse)
    def health_check(db: Session = Depends(get_db)):
        try:
            db.execute(text("SELECT 1"))
            db_ok = True
        except Exception:
            db_ok = False
        models_ok = system.malaria_model is not None and system.covid_model is not None
        return HealthResponse(
            status="ok" if db_ok and models_ok else "degraded",
            database_connected=db_ok,
            malaria_model_loaded=system.malaria_model is not None,
            covid_model_loaded=system.covid_model is not None,
        )

    @app.post("/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
    def register(request: AuthRegisterRequest, http_request: Request, db: Session = Depends(get_db)):
        enforce_rate_limit(http_request, 5, 900, "register")
        if not request.practitioner_attestation:
            raise HTTPException(status_code=400, detail="Practitioner attestation is required.")
        email = request.email.lower()
        if db.scalar(select(User).where(User.email == email)):
            raise HTTPException(status_code=409, detail="An account with that email already exists.")
        try:
            user = create_user(db, email, request.password, request.name)
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(status_code=409, detail="An account with that email already exists.") from exc
        audit(db, http_request, user.id, "auth.register")
        return AuthResponse(access_token=create_access_token(user), user=UserResponse.model_validate(user))

    @app.post("/auth/login", response_model=AuthResponse)
    def login(request: AuthLoginRequest, http_request: Request, db: Session = Depends(get_db)):
        enforce_rate_limit(http_request, 10, 300, "login")
        user = db.scalar(select(User).where(User.email == request.email.lower()))
        if user is None or not verify_password(request.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        audit(db, http_request, user.id, "auth.login")
        return AuthResponse(access_token=create_access_token(user), user=UserResponse.model_validate(user))

    @app.get("/auth/me", response_model=UserResponse)
    def me(user: User = Depends(get_current_user)):
        return UserResponse.model_validate(user)

    def owned_session(session_id: str, user: User, db: Session) -> ChatSession:
        session = db.scalar(select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user.id))
        if session is None:
            raise HTTPException(status_code=404, detail="Chat session not found.")
        return session

    def create_session(session_id: str, user: User, db: Session) -> None:
        db.add(ChatSession(id=session_id, user_id=user.id))
        try:
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(status_code=409, detail="Chat session already exists.") from exc

    @app.get("/chat", response_model=list[ChatSessionSummary])
    def list_chat_sessions(http_request: Request, user: User = Depends(require_practitioner), db: Session = Depends(get_db)):
        enforce_rate_limit(http_request, 60, 60, "chat-list")
        rows = db.scalars(
            select(ChatSession)
            .where(ChatSession.user_id == user.id)
            .order_by(ChatSession.updated_at.desc())
            .limit(100)
        ).all()
        return [
            ChatSessionSummary(
                session_id=row.id,
                title=row.title,
                created_at=row.created_at.isoformat(),
                updated_at=row.updated_at.isoformat(),
            )
            for row in rows
        ]

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest, http_request: Request, user: User = Depends(require_practitioner), db: Session = Depends(get_db)):
        enforce_rate_limit(http_request, 30, 60, "chat")
        if app.state.conversational_agent is None:
            raise HTTPException(status_code=503, detail="Conversational agent is not configured.")
        key = require_idempotency_key(http_request)
        if key:
            cached = get_idempotent_response(db, user_id=user.id, key=key)
            if cached is not None:
                return cached
        session_id = request.session_id or str(uuid.uuid4())

        session = db.scalar(
            select(ChatSession).where(
                ChatSession.id == session_id
            )
        )

        if session is None:
            session = ChatSession(
                id=session_id,
                user_id=user.id,
                title=None,
            )
            db.add(session)
            db.commit()
            db.refresh(session)

        elif session.user_id != user.id:
            raise HTTPException(
                status_code=404,
                detail="Chat session not found",
            )
        token = set_current_user(user.id)
        try:
            response = app.state.conversational_agent.invoke(
                {"input": request.message}, config={"configurable": {"session_id": session_id}}
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Chat failed — check server logs.") from exc
        finally:
            reset_current_user(token)
        reply = response.get("output", str(response)) if isinstance(response, dict) else str(response)
        result = ChatResponse(session_id=session_id, reply=reply)
        if key:
            try:
                save_idempotent_response(
                    db, user_id=user.id, key=key,
                    response=result.model_dump(),
                )
                db.commit()
            except IntegrityError:
                db.rollback()
                cached = get_idempotent_response(db, user_id=user.id, key=key)
                if cached is not None:
                    return cached
                raise
        audit(db, http_request, user.id, "chat.message", session_id)
        return result

    @app.get("/chat/{session_id}", response_model=ChatHistoryResponse)
    def get_chat_history(session_id: str, http_request: Request, user: User = Depends(require_practitioner), db: Session = Depends(get_db)):
        enforce_rate_limit(http_request, 60, 60, "chat-history")
        owned_session(session_id, user, db)
        audit(db, http_request, user.id, "chat.history.read", session_id)
        return ChatHistoryResponse(session_id=session_id, messages=serialize_history(session_id, user.id))

    @app.post("/chat/{session_id}/feedback")
    def chat_feedback(session_id: str, request: FeedbackRequest, http_request: Request, user: User = Depends(require_practitioner), db: Session = Depends(get_db)):
        enforce_rate_limit(http_request, 20, 60, "feedback")
        owned_session(session_id, user, db)
        if request.diagnosis_id is not None:
            diagnosis = db.scalar(select(DiagnosisRecord).where(
                DiagnosisRecord.id == request.diagnosis_id,
                DiagnosisRecord.user_id == user.id,
                DiagnosisRecord.session_id == session_id,
            ))
            if diagnosis is None:
                raise HTTPException(status_code=404, detail="Diagnosis record not found.")
        else:
            diagnosis = db.scalar(select(DiagnosisRecord).where(
                DiagnosisRecord.user_id == user.id,
                DiagnosisRecord.session_id == session_id,
            ).order_by(DiagnosisRecord.created_at.desc()))
        if request.correction is None and request.rating is None and request.comment is None:
            raise HTTPException(status_code=400, detail="At least one feedback field is required.")
        feedback_id = save_feedback(
            db, user_id=user.id, session_id=session_id,
            diagnosis_id=diagnosis.id if diagnosis else None,
            doctor_correction=request.correction,
            rating=request.rating,
            comment=request.comment,
        )
        audit(
            db,
            http_request,
            user.id,
            "chat.message",
            resource_type="chat_session",
            resource_id=session_id,
        )
        return {"status": "recorded", "feedback_id": feedback_id}

    def run_diagnosis_and_persist(*, patient_data: dict[str, Any], session_id: str, user: User, db: Session) -> dict[str, Any]:
        result = clean_for_json(system.diagnose(patient_data, session_id=session_id))
        diagnosis_id = save_diagnosis(
            db, user_id=user.id, session_id=session_id,
            patient_data=patient_data, result=result,
        )
        result["diagnosis_id"] = diagnosis_id
        return result

    @app.post("/diagnose/text", response_model=DiagnosisResponse)
    def diagnose_from_text(request: TextDiagnosisRequest, http_request: Request, user: User = Depends(require_practitioner), db: Session = Depends(get_db)):
        enforce_rate_limit(http_request, 20, 60, "diagnose-text")
        key = require_idempotency_key(http_request)
        if key:
            cached = get_idempotent_response(db, user_id=user.id, key=key)
            if cached is not None:
                return cached
        session_id = request.session_id or str(uuid.uuid4())
        if request.session_id:
            owned_session(session_id, user, db)
        else:
            create_session(session_id, user, db)
        try:
            features = text_to_features(request.text)
            result = run_diagnosis_and_persist(patient_data=features, session_id=session_id, user=user, db=db)
            result["session_id"] = session_id
            result["extracted_features"] = clean_for_json(features)
            if key:
                try:
                    save_idempotent_response(db, user_id=user.id, key=key, response=result)
                    db.commit()
                except IntegrityError:
                    db.rollback()
                    cached = get_idempotent_response(db, user_id=user.id, key=key)
                    if cached is not None:
                        return cached
                    raise
            audit(db, http_request, user.id, "diagnosis.text", session_id)
            return result
        except HTTPException:
            raise
        except Exception as exc:
            db.rollback()
            raise HTTPException(status_code=500, detail="Diagnosis failed — check server logs.") from exc

    @app.post("/diagnose/structured", response_model=DiagnosisResponse)
    def diagnose_from_structured_data(request: StructuredDiagnosisRequest, http_request: Request, user: User = Depends(require_practitioner), db: Session = Depends(get_db)):
        enforce_rate_limit(http_request, 20, 60, "diagnose-structured")
        key = require_idempotency_key(http_request)
        if key:
            cached = get_idempotent_response(db, user_id=user.id, key=key)
            if cached is not None:
                return cached
        session_id = request.session_id or str(uuid.uuid4())
        if request.session_id:
            owned_session(session_id, user, db)
        else:
            create_session(session_id, user, db)
        try:
            result = run_diagnosis_and_persist(patient_data=request.patient_data, session_id=session_id, user=user, db=db)
            result["session_id"] = session_id
            if key:
                try:
                    save_idempotent_response(db, user_id=user.id, key=key, response=result)
                    db.commit()
                except IntegrityError:
                    db.rollback()
                    cached = get_idempotent_response(db, user_id=user.id, key=key)
                    if cached is not None:
                        return cached
                    raise
            audit(db, http_request, user.id, "diagnosis.structured", session_id)
            return result
        except HTTPException:
            raise
        except Exception as exc:
            db.rollback()
            raise HTTPException(status_code=500, detail="Diagnosis failed — check server logs.") from exc

    return app


app = create_app()

print("\n=== REGISTERED ROUTES ===")
for route in app.routes:
    print(
        getattr(route, "methods", None),
        getattr(route, "path", None)
    )
print("=========================\n")
