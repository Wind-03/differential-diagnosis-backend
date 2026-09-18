"""Application configuration and project paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    logs_dir: Path = PROJECT_ROOT / "logs"
    malaria_data: Path = PROJECT_ROOT / "data" / "Malaria_Dataset.csv"
    covid_data: Path = PROJECT_ROOT / "data" / "covid19_patient_symptoms_diagnosis.csv"
    malaria_model: Path = PROJECT_ROOT / "models" / "malaria_model.pkl"
    malaria_scaler: Path = PROJECT_ROOT / "models" / "malaria_scaler.pkl"
    malaria_features: Path = PROJECT_ROOT / "models" / "malaria_features.pkl"
    malaria_medians: Path = PROJECT_ROOT / "models" / "malaria_medians.pkl"
    malaria_label_encoders: Path = PROJECT_ROOT / "models" / "malaria_label_encoders.pkl"
    covid_model: Path = PROJECT_ROOT / "models" / "covid_model.pkl"
    covid_scaler: Path = PROJECT_ROOT / "models" / "covid_scaler.pkl"
    covid_features: Path = PROJECT_ROOT / "models" / "covid_features.pkl"
    covid_medians: Path = PROJECT_ROOT / "models" / "covid_medians.pkl"
    covid_label_encoders: Path = PROJECT_ROOT / "models" / "covid_label_encoders.pkl"

    database_url: str | None = os.getenv("DATABASE_URL")
    jwt_secret: str = os.getenv("JWT_SECRET", "")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    access_token_minutes: int = int(os.getenv("ACCESS_TOKEN_MINUTES", "60"))
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", "3"))
    db_max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "2"))

    deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    cors_origins: tuple[str, ...] = tuple(
        origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if origin.strip()
    )
    diagnosis_threshold: float = float(os.getenv("DIAGNOSIS_THRESHOLD", "0.6"))


settings = Settings()
if not settings.jwt_secret or len(settings.jwt_secret) < 32:
    raise RuntimeError("JWT_SECRET must be set and contain at least 32 characters.")
