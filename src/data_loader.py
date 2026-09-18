"""Dataset loading utilities."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config import settings


def _load_csv(path: str | Path, name: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} dataset not found: {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded {len(df)} {name} records")
    return df


def load_malaria_data(data_path: str | Path | None = None) -> pd.DataFrame:
    return _load_csv(data_path or settings.malaria_data, "malaria")


def load_covid_data(data_path: str | Path | None = None) -> pd.DataFrame:
    return _load_csv(data_path or settings.covid_data, "COVID")
