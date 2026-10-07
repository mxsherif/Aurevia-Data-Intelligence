"""Tests for the synthetic telecom dataset generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from datasets.generate_sample_data import (
    DEFAULT_ROWS,
    MISSING_RATES,
    RANDOM_SEED,
    build_dataframe,
    generate,
)

EXPECTED_COLUMNS = [
    "customer_id",
    "region",
    "city",
    "signup_date",
    "contract_type",
    "monthly_charge",
    "data_usage_gb",
    "support_calls",
    "payment_method",
    "tenure_months",
    "satisfaction_score",
    "revenue",
    "churn",
    "product_type",
    "network_type",
]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return build_dataframe(rows=1_200, seed=RANDOM_SEED)


def test_schema_and_shape(frame: pd.DataFrame):
    assert list(frame.columns) == EXPECTED_COLUMNS
    assert len(frame) == 1_200


def test_default_row_count_is_5000():
    assert DEFAULT_ROWS == 5_000


def test_generation_is_deterministic():
    first = build_dataframe(rows=300, seed=RANDOM_SEED)
    second = build_dataframe(rows=300, seed=RANDOM_SEED)

    pd.testing.assert_frame_equal(first, second)


def test_a_different_seed_changes_the_data():
    first = build_dataframe(rows=300, seed=RANDOM_SEED)
    second = build_dataframe(rows=300, seed=RANDOM_SEED + 1)

    assert not first["churn"].equals(second["churn"])


def test_customer_ids_are_unique(frame: pd.DataFrame):
    assert frame["customer_id"].is_unique


def test_controlled_missing_values(frame: pd.DataFrame):
    for column, rate in MISSING_RATES.items():
        missing = int(frame[column].isna().sum())
        assert missing == pytest.approx(rate * len(frame), abs=2), column

    # Columns outside MISSING_RATES must be complete.
    complete = set(EXPECTED_COLUMNS) - set(MISSING_RATES)
    assert frame[sorted(complete)].notna().all().all()


def test_value_domains(frame: pd.DataFrame):
    assert set(frame["churn"].unique()) == {0, 1}
    assert set(frame["contract_type"].unique()) <= {"Month-to-month", "One year", "Two year"}
    assert set(frame["network_type"].unique()) <= {"3G", "4G", "5G"}

    satisfaction = frame["satisfaction_score"].dropna()
    assert satisfaction.between(1, 5).all()

    assert frame["tenure_months"].min() >= 0
    assert frame["revenue"].min() >= 0


def test_city_belongs_to_its_region(frame: pd.DataFrame):
    from datasets.generate_sample_data import REGIONS

    paired = frame.dropna(subset=["city"])
    for region, city in zip(paired["region"], paired["city"]):
        assert city in REGIONS[region]["cities"]


def test_churn_rate_is_realistic(frame: pd.DataFrame):
    assert 0.05 < frame["churn"].mean() < 0.45


def test_relationships_are_present(frame: pd.DataFrame):
    # Monthly charge should drive both usage and revenue.
    assert frame["monthly_charge"].corr(frame["data_usage_gb"]) > 0.3

    # Month-to-month customers must churn more than two-year ones.
    by_contract = frame.groupby("contract_type")["churn"].mean()
    assert by_contract["Month-to-month"] > by_contract["Two year"]

    # More support calls should mean lower satisfaction.
    assert frame["support_calls"].corr(frame["satisfaction_score"]) < -0.2


def test_seasonality_in_signups(frame: pd.DataFrame):
    monthly = pd.to_datetime(frame["signup_date"]).dt.month.value_counts()
    # A seasonal pattern means months are clearly uneven.
    assert monthly.max() > 1.4 * monthly.min()


def test_anomalies_are_injected(frame: pd.DataFrame):
    charge = frame["monthly_charge"]

    # Whale accounts far above the bulk of the distribution.
    assert charge.max() > 5 * charge.median()
    # Billing glitches: a few negative charges.
    assert (charge < 0).sum() > 0
    # Dormant subscribers with zero usage.
    assert (frame["data_usage_gb"] == 0).sum() > 0
    # Support-call storms.
    assert frame["support_calls"].max() >= 20


def test_date_range_spans_multiple_years(frame: pd.DataFrame):
    dates = pd.to_datetime(frame["signup_date"])
    assert dates.dt.year.nunique() >= 3
    assert dates.is_monotonic_increasing  # sorted, so time series work out of the box


def test_generate_writes_a_readable_csv(tmp_path: Path):
    path = generate(rows=200, seed=1, output=tmp_path / "out.csv")

    assert path.exists()
    reloaded = pd.read_csv(path)
    assert list(reloaded.columns) == EXPECTED_COLUMNS
    assert len(reloaded) == 200
