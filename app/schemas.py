"""Pydantic request/response schemas for the API."""
from typing import Any
from datetime import datetime
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class AuthRegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=12, max_length=128)
    practitioner_attestation: bool = Field(..., description="Acknowledges self-attested practitioner access; no registry verification is performed.")

class AuditEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: str | None = None
    action: str
    resource_type: str | None = None
    resource_id: str | None = None
    metadata: dict[str, Any] | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    created_at: datetime

class AuthLoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    email: EmailStr
    role: str
    verification_status: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class TextDiagnosisRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=12000)
    session_id: str | None = Field(None, max_length=36)


class StructuredDiagnosisRequest(BaseModel):
    patient_data: dict[str, Any] = Field(...)
    session_id: str | None = Field(None, max_length=36)


class TopFactor(BaseModel):
    feature: str
    value: float
    direction: str
    magnitude: float


class ModelExplanation(BaseModel):
    base_rate_probability: float
    raw_model_probability: float
    top_factors: list[TopFactor]


class DiagnosisResponse(BaseModel):
    diagnosis_id: int | None = None
    session_id: str
    primary_diagnosis: str
    full_diagnosis: str
    malaria_probability: float
    covid_probability: float
    confidence: float
    malaria_model_prediction: str
    covid_model_prediction: str
    recommendations: list[str]
    malaria_explanation: ModelExplanation
    covid_explanation: ModelExplanation
    covid_model_status: str
    extracted_features: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    malaria_model_loaded: bool
    covid_model_loaded: bool


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=12000)
    session_id: str | None = Field(None, max_length=36)


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]


class ChatSessionSummary(BaseModel):
    session_id: str
    title: str | None = None
    created_at: str
    updated_at: str


class FeedbackRequest(BaseModel):
    correction: str | None = Field(None, min_length=1, max_length=4000)
    rating: int | None = Field(None, ge=1, le=5)
    comment: str | None = Field(None, min_length=1, max_length=4000)
    diagnosis_id: int | None = Field(None, ge=1)
