# AI Differential Diagnosis Backend

FastAPI backend for a Malaria vs COVID-19 differential decision-support application. It uses the supplied XGBoost model artifacts, SHAP explanations, conservative NLP feature extraction, optional DeepSeek/LangChain conversation, and Neon PostgreSQL for authenticated application data.

> **Clinical safety:** This is a research/educational decision-support system, not a clinically validated diagnostic device. Model outputs are not confirmed diagnoses. The COVID-19 model is explicitly marked provisional in API responses.

## Architecture

```text
React frontend
    │ HTTPS + Bearer JWT
    ▼
FastAPI
    ├── authentication / authorization
    ├── chat ──► LangChain / DeepSeek
    │              └── PostgreSQL message history
    ├── diagnosis ──► NLP ──► XGBoost + SHAP
    │                    └── diagnosis_records
    └── feedback ───────────► diagnosis_feedback
                                  │
                                  ▼
                           Neon PostgreSQL
```

## Authentication

Accounts are application accounts. Practitioner status is **self-attested** and is stored as `verification_status=self_attested`; no licensing registry or professional identity verification service is claimed or performed.

- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`

Protected endpoints require `Authorization: Bearer <access-token>`.

## Run

```bash
uvicorn api:app --reload --port 8000
```

API documentation is available at `/docs`.


