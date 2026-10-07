"""Shared pytest fixtures and path setup."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import SAMPLE_DATASET_PATH, Settings  # noqa: E402


@pytest.fixture
def settings() -> Settings:
    """Settings independent of the developer's local `.env`."""
    return Settings(
        openai_api_key="",
        openai_model="gpt-4.1-mini",
        max_upload_mb=50,
        max_rows=100_000,
        preview_rows=25,
        log_level="WARNING",
    )


@pytest.fixture
def messy_frame() -> pd.DataFrame:
    """A small frame that exercises most of the profiler's branches."""
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(12)],
            "region": ["North", "South", "North", "East", "West"] * 2 + ["North", "South"],
            "signup_date": [
                "2023-01-15", "2023-02-20", "2023-03-05", "2023-04-11",
                "2023-05-30", "2023-06-14", "2023-07-19", "2023-08-23",
                "2023-09-02", "2023-10-27", "2023-11-08", "2023-12-31",
            ],
            "monthly_charge": [50.0, 60.0, 55.0, 70.0, 65.0, 58.0,
                               62.0, 57.0, 61.0, 59.0, 63.0, 9_000.0],
            "support_calls": [1, 2, 0, 3, 1, 2, 1, 0, 2, 1, 3, 2],
            "is_active": [True, False, True, True, False, True,
                          True, False, True, True, False, True],
            "notes": [None, "ok", None, None, "late payment", None,
                      None, None, "upgraded", None, None, None],
            "plan": ["basic"] * 12,
            "empty_col": [None] * 12,
        }
    )


@pytest.fixture
def csv_file(tmp_path: Path, messy_frame: pd.DataFrame) -> Path:
    path = tmp_path / "data.csv"
    messy_frame.to_csv(path, index=False)
    return path


@pytest.fixture
def xlsx_file(tmp_path: Path, messy_frame: pd.DataFrame) -> Path:
    path = tmp_path / "data.xlsx"
    messy_frame.to_excel(path, index=False, sheet_name="customers")
    return path


@pytest.fixture
def sample_dataset_path() -> Path:
    if not SAMPLE_DATASET_PATH.exists():
        pytest.skip(
            "sample dataset missing; run `python datasets/generate_sample_data.py`"
        )
    return SAMPLE_DATASET_PATH
