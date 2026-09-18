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

## Neon setup

Copy `.env.example` to `.env` and put your Neon connection string in the backend only:

```env
DATABASE_URL=postgresql://USER:PASSWORD@HOST/DATABASE?sslmode=require
```

Never put `DATABASE_URL` in a React `VITE_*` variable and never commit `.env`.

Also set a strong JWT secret of at least 32 characters:

```env
JWT_SECRET=replace_with_a_long_random_secret_at_least_32_characters
JWT_ALGORITHM=HS256
ACCESS_TOKEN_MINUTES=60
```

The backend uses Psycopg 3 through SQLAlchemy. Neon TLS is enforced when `sslmode` is absent from a PostgreSQL URL.

## Setup

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env
```

Set `DEEPSEEK_API_KEY` if conversational chat is required. Structured diagnosis endpoints do not require DeepSeek.

Missing tables are created on application startup. For an existing database whose schema came from an older version of this project, inspect/migrate it before deployment rather than assuming `create_all()` can safely transform old tables.

## Run

```bash
uvicorn api:app --reload --port 8000
```

API documentation is available at `/docs`.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Database/model health |
| POST | `/auth/register` | Create self-attested practitioner account |
| POST | `/auth/login` | Login and receive JWT |
| GET | `/auth/me` | Current authenticated user |
| GET | `/chat` | List the current user's sessions |
| POST | `/chat` | Send a conversational message |
| GET | `/chat/{session_id}` | Read the current user's conversation history |
| POST | `/chat/{session_id}/feedback` | Persist practitioner feedback |
| POST | `/diagnose/text` | Run diagnosis from free text |
| POST | `/diagnose/structured` | Run diagnosis from structured features |

`/chat`, `/diagnose/text`, and `/diagnose/structured` accept an optional `Idempotency-Key` header. Reusing the same key for the same authenticated user returns the previously stored response instead of executing the operation again.

### Example diagnosis request

```json
{
  "text": "35-year-old male with fever, headache and vomiting",
  "session_id": "optional-case-id"
}
```

The response includes a database `diagnosis_id` when persistence succeeds.

### Example structured request

```json
{
  "patient_data": {
    "Age": 35,
    "Sex": "Male",
    "Fever": 1,
    "Headache": 1,
    "Vomiting": 1
  }
}
```

### Example feedback

```json
{
  "diagnosis_id": 123,
  "correction": "Confirmed malaria by RDT",
  "rating": 4,
  "comment": "Useful symptom collection"
}
```

## Data handling

The database is the authoritative persistence layer for authenticated users, chat sessions/messages, diagnosis records, feedback, and idempotency responses. The old raw-patient CSV diagnosis log is no longer written by the diagnosis service.

The NLP extractor records only information explicitly stated in the supplied patient text. Missing age, sex, vitals, contact history, and comorbidities are not fabricated into the extracted feature payload. The model-preparation layer may use training-derived medians for missing model inputs; these should not be interpreted as observed patient facts.

Because `diagnosis_records` can contain clinical information, use appropriate database access controls, retention, backups, encryption/TLS, and legal/organizational compliance controls before handling real patient information.

## Production notes

- The current rate limiter is process-local. Use Redis or an API gateway for multi-instance deployments.
- JWT access tokens are bearer tokens. The frontend currently stores them client-side; an httpOnly Secure SameSite cookie design is a stronger production option against token theft through XSS.
- `create_all()` is convenient for initial deployment. Use Alembic or an equivalent migration process for controlled schema changes.
- No professional licensing verification is performed.

## Tests

```bash
python -m compileall app src api.py main.py differential_diagnosis.py langchain_agent.py schemas.py
pytest -q
```
