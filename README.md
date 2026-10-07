# DataPilot

An agentic AI data analyst. Upload a CSV or Excel file and DataPilot works out
what the dataset *is* — its shape, field semantics, and quality problems —
before you ask it anything.

> **Status: Phase 1 (foundation).** Data loading, dataset profiling, the sample
> dataset, and the Overview page are implemented. LLM agents, natural-language
> analysis, visualisations, anomaly detection, forecasting, and the ML Lab are
> not built yet.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (optional) Configure the environment
cp .env.example .env        # Windows: copy .env.example .env

# 3. (optional) Regenerate the sample dataset
python datasets/generate_sample_data.py

# 4. Run the app
python run.py               # http://localhost:8501
```

`run.py` accepts `--port`, `--host`, and `--headless`. You can also start the
app directly with `streamlit run app/main.py`.

Run the test suite with:

```bash
python -m pytest
```

---

## Configuration

Settings are read from the environment, seeded from a `.env` file at the project
root. **No API key is required for Phase 1** — profiling is entirely local, and
a missing, blank, or placeholder key is treated as "not configured" rather than
an error.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | _(unset)_ | Reserved for the Phase 2 agents. |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Reserved for the Phase 2 agents. |
| `DATAPILOT_MAX_UPLOAD_MB` | `200` | Rejects files larger than this. |
| `DATAPILOT_MAX_ROWS` | `500000` | Rows loaded before truncation kicks in. |
| `DATAPILOT_PREVIEW_ROWS` | `100` | Rows shown in the UI preview table. |
| `DATAPILOT_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR`. |

Malformed numeric values fall back to the defaults with a logged warning, so a
typo in `.env` cannot stop the app from starting.

---

## Project layout

```text
app/
  main.py          Streamlit entry point (page config, state, wiring)
  config.py        Environment-driven settings; never raises
  agents/          (Phase 2) LLM agents
  tools/           (Phase 2) agent-callable tools
  services/
    data_loader.py CSV / XLSX loading and error handling
    profiler.py    Type inference, statistics, quality warnings
  models/
    profile.py     DatasetProfile / ColumnProfile / DataQualityWarning
  ui/
    overview.py    The Overview page
    components.py  Shared metric, table, and warning renderers

datasets/
  generate_sample_data.py      Synthetic telecom dataset generator
  sample_telecom_customers.csv 5,000-row bundled demo dataset

tests/                         pytest suite
screenshots/                   UI screenshots
run.py                         Launcher
```

### Data flow

```text
file / upload  ->  data_loader.load_dataframe()  ->  LoadResult
                                                        |
                                                        v
                        profiler.profile_dataframe()  ->  DatasetProfile
                                                        |
                                                        v
                                             ui.overview / ui.components
```

The services layer is pure: it takes bytes or a path and returns dataclasses.
Streamlit is confined to `app/ui` and `app/main.py`, which keeps the loader and
profiler straightforward to test and to reuse from the Phase 2 agents.

---

## What the loader handles

`load_dataframe()` raises exactly one exception type — `DataLoadError`, with a
message safe to show a user — for:

- empty and whitespace-only files
- header-only files with no data rows
- malformed / ragged CSVs (over-long rows are skipped, not fatal)
- unknown delimiters (`,`, `;`, `\t` are sniffed) and non-UTF-8 encodings
- `.xlsx` files that are not really workbooks, and missing sheet names
- unsupported extensions, and files with no extension at all
- files above the configured size limit

It also normalises the frame on the way in: duplicate column names are made
unique (`name`, `name_2`, `name_3`), blank and `Unnamed:` headers get positional
names, fully empty rows and columns are dropped, and oversized files are
truncated to `DATAPILOT_MAX_ROWS` — each of which is reported in
`LoadResult.notes` and surfaced in the UI.

## What the profiler detects

Dataset level: row count, column count, memory usage, duplicate rows (count and
percentage), total/missing cells and the missing percentage, and strongly
correlated numeric pairs (`|r| >= 0.75`).

Per column: pandas dtype plus an inferred **semantic** type — `numeric`,
`integer`, `categorical`, `boolean`, `datetime`, `text`, `identifier`, `empty` —
non-null count, missing count and percentage, unique count and percentage,
memory, and the constant / low-cardinality / likely-ID flags. Numeric columns
additionally get min, Q1, median, mean, Q3, max, std, skew, zero and negative
counts, and IQR-based outliers (falling back to a 3σ rule when the IQR is zero).
Date columns get their range and span in days; categorical columns get their
most frequent values.

Type inference is deliberately conservative: a `report_year` column of bare
integers is **not** claimed as a date, short codes like `A1` are categorical
rather than dates, and a near-unique numeric column is only an identifier if its
name says so — `revenue` stays a measure, `customer_id` becomes an ID.

Findings are returned as `DataQualityWarning`s sorted most-severe-first, covering
empty and constant columns, columns that are mostly missing, duplicate rows,
unexpected negative values in amount-like fields, outliers, high-cardinality free
text, the absence of a date column, and correlated pairs.

## The sample dataset

`datasets/sample_telecom_customers.csv` — 5,000 synthetic telecom customers,
seeded at `42` so it regenerates byte-identically. It is structured, not random:

- **Relationships.** `monthly_charge` is driven by product, contract, region and
  tenure; `data_usage_gb` follows charge and network generation; `support_calls`
  is Poisson and worse on 3G; `satisfaction_score` falls with support calls;
  `churn` is logistic in contract type, payment method, network, calls,
  satisfaction and tenure (~17% churn rate); `revenue` is charge × active months.
- **Seasonality.** Signups follow a yearly sine pattern on a growth trend, with a
  weekend dip — enough signal for forecasting later.
- **Controlled missingness.** ~7.5% of `satisfaction_score`, 3% of
  `data_usage_gb`, 1.8% of `payment_method`, 1.2% of `city`, 0.9% of
  `support_calls` (720 cells in total).
- **Planted anomalies.** A handful of whale accounts, support-call storms,
  negative "billing glitch" charges, and dormant zero-usage subscribers.

```bash
python datasets/generate_sample_data.py --rows 10000 --seed 7 --output my.csv
```

---

## Roadmap

| Phase | Scope |
| --- | --- |
| **1 — Foundation** | ✅ Config, loading, profiling, sample data, Overview page |
| 2 — Agents | LLM agents and tools for natural-language analysis |
| 3 — Visualisation | Automatic chart selection and a chart gallery |
| 4 — Detection | Anomaly detection and forecasting pages |
| 5 — ML Lab | Guided model training and evaluation |

## Limitations (Phase 1)

- No LLM or agent functionality — `app/agents/` and `app/tools/` are empty
  packages, and `OPENAI_API_KEY` is read but never used.
- Everything is loaded into memory; very large files are truncated to
  `DATAPILOT_MAX_ROWS` rather than streamed or chunked.
- Excel loading reads one sheet at a time (the first unless a sheet is chosen).
  Legacy `.xls` support depends on `xlrd` and is best effort.
- Date sniffing is day-first/ISO tolerant but ambiguous formats like `03/04/2023`
  are resolved by pandas, not by DataPilot.
- Outlier detection is univariate (IQR). Multivariate anomalies arrive in Phase 4.
- Correlations are Pearson on up to 40 numeric columns, so non-linear
  relationships are not reported.
