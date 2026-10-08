"""Generate `sample_telecom_customers.csv` -- Aurevia's bundled demo dataset.

The dataset is synthetic but deliberately *structured*: churn, revenue and
satisfaction are driven by the other fields, signups follow a seasonal pattern,
and a controlled amount of missingness plus a handful of injected anomalies give
the profiler and (later) the anomaly/forecasting pages something real to find.

Run it from the project root::

    python datasets/generate_sample_data.py
    python datasets/generate_sample_data.py --rows 10000 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
DEFAULT_ROWS = 5_000
OUTPUT_PATH = Path(__file__).resolve().parent / "sample_telecom_customers.csv"

SIGNUP_START = pd.Timestamp("2021-01-01")
SIGNUP_END = pd.Timestamp("2024-12-31")
SNAPSHOT_DATE = pd.Timestamp("2025-01-31")

# --------------------------------------------------------------------------- #
# Reference data: regions carry both a city list and a spending multiplier.
# --------------------------------------------------------------------------- #

REGIONS: dict[str, dict] = {
    "Cairo": {
        "weight": 0.30,
        "cities": ["Nasr City", "Maadi", "Heliopolis", "Zamalek", "New Cairo"],
        "spend_multiplier": 1.20,
    },
    "Alexandria": {
        "weight": 0.18,
        "cities": ["Smouha", "Sidi Gaber", "Montazah", "Borg El Arab"],
        "spend_multiplier": 1.05,
    },
    "Delta": {
        "weight": 0.20,
        "cities": ["Mansoura", "Tanta", "Zagazig", "Damanhour"],
        "spend_multiplier": 0.92,
    },
    "Upper Egypt": {
        "weight": 0.17,
        "cities": ["Assiut", "Sohag", "Minya", "Qena"],
        "spend_multiplier": 0.82,
    },
    "Canal": {
        "weight": 0.10,
        "cities": ["Suez", "Ismailia", "Port Said"],
        "spend_multiplier": 0.98,
    },
    "Red Sea": {
        "weight": 0.05,
        "cities": ["Hurghada", "Sharm El Sheikh", "Marsa Alam"],
        "spend_multiplier": 1.15,
    },
}

CONTRACT_TYPES = {
    "Month-to-month": {"weight": 0.48, "churn_bias": 0.22, "price_bias": 0.95},
    "One year": {"weight": 0.30, "churn_bias": -0.04, "price_bias": 1.05},
    "Two year": {"weight": 0.22, "churn_bias": -0.12, "price_bias": 1.12},
}

PAYMENT_METHODS = {
    "Credit card": {"weight": 0.26, "churn_bias": -0.04},
    "Bank transfer": {"weight": 0.22, "churn_bias": -0.05},
    "Mobile wallet": {"weight": 0.30, "churn_bias": 0.01},
    "Cash": {"weight": 0.22, "churn_bias": 0.09},
}

PRODUCT_TYPES = {
    "Mobile": {"weight": 0.42, "base_charge": 180.0, "base_usage": 9.0},
    "Fiber Internet": {"weight": 0.26, "base_charge": 420.0, "base_usage": 180.0},
    "DSL Internet": {"weight": 0.14, "base_charge": 260.0, "base_usage": 95.0},
    "Bundle": {"weight": 0.18, "base_charge": 560.0, "base_usage": 220.0},
}

NETWORK_TYPES = {
    "3G": {"weight": 0.12, "usage_multiplier": 0.55, "churn_bias": 0.10},
    "4G": {"weight": 0.56, "usage_multiplier": 1.00, "churn_bias": 0.00},
    "5G": {"weight": 0.32, "usage_multiplier": 1.45, "churn_bias": -0.05},
}

# Fraction of values blanked out per column, to exercise missing-value handling.
MISSING_RATES = {
    "satisfaction_score": 0.075,
    "data_usage_gb": 0.030,
    "payment_method": 0.018,
    "city": 0.012,
    "support_calls": 0.009,
}


def _weights(table: dict[str, dict]) -> tuple[list[str], np.ndarray]:
    keys = list(table)
    weights = np.array([table[k]["weight"] for k in keys], dtype=float)
    return keys, weights / weights.sum()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- #
# Field builders
# --------------------------------------------------------------------------- #

def _signup_dates(rng: np.random.Generator, n: int) -> pd.Series:
    """Signup dates with a yearly seasonal peak (summer) and a growth trend."""
    span_days = (SIGNUP_END - SIGNUP_START).days
    day_index = np.arange(span_days + 1)

    # Growth trend plus a sine wave peaking around mid-year, plus a Ramadan-ish
    # secondary bump so the series is not a clean sinusoid.
    trend = 1.0 + 0.6 * (day_index / span_days)
    seasonal = 1.0 + 0.45 * np.sin(2 * np.pi * (day_index - 40) / 365.25)
    secondary = 1.0 + 0.18 * np.sin(4 * np.pi * day_index / 365.25)
    weekday_dip = np.where(((SIGNUP_START + pd.to_timedelta(day_index, "D")).dayofweek >= 4), 0.85, 1.0)

    intensity = trend * seasonal * secondary * weekday_dip
    intensity = intensity / intensity.sum()

    chosen = rng.choice(day_index, size=n, p=intensity)
    return pd.Series(SIGNUP_START + pd.to_timedelta(np.sort(chosen), unit="D"))


def build_dataframe(rows: int = DEFAULT_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build the synthetic telecom customer dataset."""
    rng = np.random.default_rng(seed)

    # -- identity ---------------------------------------------------------- #
    customer_id = [f"CUST-{100000 + i}" for i in range(rows)]

    region_keys, region_p = _weights(REGIONS)
    region = rng.choice(region_keys, size=rows, p=region_p)
    city = np.array([rng.choice(REGIONS[r]["cities"]) for r in region])

    # -- contract / product ------------------------------------------------ #
    contract_keys, contract_p = _weights(CONTRACT_TYPES)
    contract_type = rng.choice(contract_keys, size=rows, p=contract_p)

    payment_keys, payment_p = _weights(PAYMENT_METHODS)
    payment_method = rng.choice(payment_keys, size=rows, p=payment_p)

    product_keys, product_p = _weights(PRODUCT_TYPES)
    product_type = rng.choice(product_keys, size=rows, p=product_p)

    network_keys, network_p = _weights(NETWORK_TYPES)
    network_type = rng.choice(network_keys, size=rows, p=network_p)

    # -- tenure ------------------------------------------------------------ #
    signup_date = _signup_dates(rng, rows)
    tenure_months = (
        ((SNAPSHOT_DATE - signup_date).dt.days / 30.44).round().astype(int).clip(lower=0)
    )

    # -- monthly charge ---------------------------------------------------- #
    base_charge = np.array([PRODUCT_TYPES[p]["base_charge"] for p in product_type])
    price_bias = np.array([CONTRACT_TYPES[c]["price_bias"] for c in contract_type])
    spend_mult = np.array([REGIONS[r]["spend_multiplier"] for r in region])

    monthly_charge = (
        base_charge
        * price_bias
        * spend_mult
        * rng.normal(1.0, 0.13, rows)          # customer-level variation
        * (1 + 0.004 * np.minimum(tenure_months, 36))  # long-timers upsold
    )
    monthly_charge = np.clip(monthly_charge, 60, None).round(2)

    # -- data usage (driven by product, network and charge) ---------------- #
    base_usage = np.array([PRODUCT_TYPES[p]["base_usage"] for p in product_type])
    usage_mult = np.array([NETWORK_TYPES[n]["usage_multiplier"] for n in network_type])
    data_usage_gb = (
        base_usage
        * usage_mult
        * (monthly_charge / base_charge) ** 0.7
        * rng.lognormal(0.0, 0.30, rows)
    )
    data_usage_gb = np.clip(data_usage_gb, 0.1, None).round(2)

    # -- support calls (Poisson, worse on 3G and for heavy users) ---------- #
    call_rate = (
        0.8
        + 1.4 * (network_type == "3G")
        + 0.5 * (product_type == "Fiber Internet")
        + 0.6 * (data_usage_gb > np.quantile(data_usage_gb, 0.85))
    )
    support_calls = rng.poisson(call_rate).astype(int)

    # -- satisfaction (1-5, hurt by calls, helped by 5G and tenure) -------- #
    satisfaction_raw = (
        4.2
        - 0.42 * support_calls
        + 0.30 * (network_type == "5G")
        - 0.35 * (network_type == "3G")
        + 0.012 * np.minimum(tenure_months, 40)
        + rng.normal(0, 0.55, rows)
    )
    satisfaction_score = np.clip(np.rint(satisfaction_raw), 1, 5).astype(int)

    # -- churn (logistic in the drivers above) ----------------------------- #
    contract_bias = np.array([CONTRACT_TYPES[c]["churn_bias"] for c in contract_type])
    payment_bias = np.array([PAYMENT_METHODS[p]["churn_bias"] for p in payment_method])
    network_bias = np.array([NETWORK_TYPES[n]["churn_bias"] for n in network_type])

    churn_logit = (
        -1.55
        + 4.0 * contract_bias
        + 3.0 * payment_bias
        + 2.5 * network_bias
        + 0.26 * support_calls
        - 0.42 * (satisfaction_score - 3)
        - 0.022 * np.minimum(tenure_months, 48)
        + 0.0011 * (monthly_charge - monthly_charge.mean())
        + rng.normal(0, 0.45, rows)
    )
    churn = (rng.random(rows) < _sigmoid(churn_logit)).astype(int)

    # -- revenue (lifetime-to-date; churners billed for a partial month) --- #
    active_months = np.where(
        churn == 1, np.maximum(tenure_months.to_numpy() - 1, 0), tenure_months.to_numpy()
    )
    revenue = np.round(
        np.clip(monthly_charge * active_months * rng.normal(1.0, 0.04, rows), 0, None), 2
    )

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "region": region,
            "city": city,
            "signup_date": signup_date.dt.strftime("%Y-%m-%d"),
            "contract_type": contract_type,
            "monthly_charge": monthly_charge,
            "data_usage_gb": data_usage_gb,
            "support_calls": support_calls,
            "payment_method": payment_method,
            "tenure_months": tenure_months,
            "satisfaction_score": satisfaction_score,
            "revenue": revenue,
            "churn": churn,
            "product_type": product_type,
            "network_type": network_type,
        }
    )

    df = _inject_anomalies(df, rng)
    df = _inject_missing(df, rng)
    return df


# --------------------------------------------------------------------------- #
# Imperfections
# --------------------------------------------------------------------------- #

def _inject_anomalies(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Plant a small, known set of outliers and dirty values."""
    n = len(df)
    df = df.copy()

    # Whale accounts: implausibly high charge and usage.
    whales = rng.choice(n, size=max(6, n // 600), replace=False)
    df.loc[whales, "monthly_charge"] = (df.loc[whales, "monthly_charge"] * 9.5).round(2)
    df.loc[whales, "data_usage_gb"] = (df.loc[whales, "data_usage_gb"] * 12.0).round(2)
    df.loc[whales, "revenue"] = (
        df.loc[whales, "monthly_charge"] * df.loc[whales, "tenure_months"]
    ).round(2)

    # Support-call storms.
    storms = rng.choice(n, size=max(4, n // 1200), replace=False)
    df["support_calls"] = df["support_calls"].astype("int64")
    df.loc[storms, "support_calls"] = rng.integers(28, 60, size=len(storms)).astype("int64")

    # Billing glitch: a few negative charges that should be flagged, not fixed.
    glitches = rng.choice(n, size=max(3, n // 1600), replace=False)
    df.loc[glitches, "monthly_charge"] = -df.loc[glitches, "monthly_charge"].abs().round(2)

    # Zero-usage subscribers who are still being billed.
    dormant = rng.choice(n, size=max(5, n // 700), replace=False)
    df.loc[dormant, "data_usage_gb"] = 0.0

    return df


def _inject_missing(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Blank out a controlled share of values in selected columns."""
    df = df.copy()
    n = len(df)
    for column, rate in MISSING_RATES.items():
        if column not in df.columns:
            continue
        count = int(round(rate * n))
        if count <= 0:
            continue
        idx = rng.choice(n, size=count, replace=False)
        if pd.api.types.is_integer_dtype(df[column]):
            # Integers cannot hold NaN; widen to a nullable/float column.
            df[column] = df[column].astype("float64")
        df.loc[idx, column] = np.nan
    return df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def generate(
    rows: int = DEFAULT_ROWS,
    seed: int = RANDOM_SEED,
    output: Path | str = OUTPUT_PATH,
) -> Path:
    """Generate the dataset and write it to `output`, returning the path."""
    df = build_dataframe(rows=rows, seed=seed)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _summarise(df: pd.DataFrame) -> str:
    lines = [
        f"rows x columns      : {df.shape[0]:,} x {df.shape[1]}",
        f"churn rate          : {df['churn'].mean():.2%}",
        f"mean monthly charge : {df['monthly_charge'].mean():,.2f}",
        f"total revenue       : {df['revenue'].sum():,.0f}",
        f"signup range        : {df['signup_date'].min()} -> {df['signup_date'].max()}",
        f"missing cells       : {int(df.isna().sum().sum()):,}",
    ]
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        lines.append("missing by column   : " + ", ".join(f"{k}={v}" for k, v in missing.items()))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Aurevia's sample telecom dataset.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="number of customers")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="random seed")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="output CSV path")
    args = parser.parse_args()

    path = generate(rows=args.rows, seed=args.seed, output=args.output)
    print(f"Wrote {path}")
    print(_summarise(pd.read_csv(path)))


if __name__ == "__main__":
    main()
