# Aurevia

An agentic data-intelligence platform. Upload a CSV or Excel file and Aurevia
works out what the dataset *is* — its shape, field semantics, and quality
problems — then lets you interrogate and chart it.

> **Status: Phase 2 (analytical foundation).** Phase 1's loading and profiling
> plus a deterministic tool layer, a Plotly chart engine, and the Explore and
> Visualize pages. There is **no LLM integration yet**: everything on screen is
> computed deterministically. Agents, natural-language analysis, anomaly
> detection, forecasting, and the ML Lab come later.

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
python -m pytest            # 377 tests
```

---

## The three pages

| Page | What it does |
| --- | --- |
| **Overview** | Load a file, see shape, memory, field types, and data-quality warnings. |
| **Explore** | Guided exploration in five tabs: summary, data quality, distributions, categories, relationships. |
| **Visualize** | A manual chart builder whose controls adapt to the chart type. |

All three share one loaded dataset and one sidebar (`app/ui/state.py`), so
switching pages never reloads the file.

---

## Configuration

Settings are read from the environment, seeded from a `.env` file at the project
root. **No API key is required**: Phase 2 is entirely local, and a missing,
blank, or placeholder key is treated as "not configured" rather than an error.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | _(unset)_ | Reserved for the agent phase. |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Reserved for the agent phase. |
| `DATAPILOT_MAX_UPLOAD_MB` | `200` | Rejects files larger than this. |
| `DATAPILOT_MAX_ROWS` | `500000` | Rows loaded before truncation kicks in. |
| `DATAPILOT_PREVIEW_ROWS` | `100` | Rows shown in the UI preview table. |
| `DATAPILOT_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR`. |

Malformed numeric values fall back to the defaults with a logged warning, so a
typo in `.env` cannot stop the app from starting. (The `DATAPILOT_` prefix
predates the rename to Aurevia and is kept for compatibility.)

---

## Project layout

```text
app/
  main.py          Streamlit entry point and page navigation
  config.py        Environment-driven settings; never raises
  agents/          (later) LLM agents
  tools/           The deterministic analysis layer
    exceptions.py  ToolError and its subclasses
    validation.py  Shared column / type / parameter checks
    aggregations.py The aggregation registry
    filters.py     Filter operators and filter_rows()
    timeseries.py  Frequencies and calculate_time_trend()
    analysis.py    Schema, summaries, grouping, stats, correlation,
                   segments, ranking, outliers
    charts.py      generate_chart_data() and the chart-type registry
  services/
    data_loader.py CSV / XLSX loading and error handling
    profiler.py    Type inference, statistics, quality warnings
    visualization.py The Plotly rendering engine
  models/
    profile.py     DatasetProfile / ColumnProfile / DataQualityWarning
  ui/
    state.py       Shared session state and the data-source sidebar
    overview.py    The Overview page
    explore.py     The Explore page
    visualize.py   The Visualize page
    components.py  Shared metric, table, warning and chart renderers

datasets/
  generate_sample_data.py      Synthetic telecom dataset generator
  sample_telecom_customers.csv 5,000-row bundled demo dataset

tests/                         pytest suite
screenshots/                   UI screenshots
run.py                         Launcher
```

### Layering

```text
file / upload
   -> services.data_loader      LoadResult
   -> services.profiler         DatasetProfile
   -> tools.*                   ChartData, DatasetSchema, TimeTrendResult, ...
   -> services.visualization    plotly Figure
   -> ui.*                      Streamlit pages
```

`app/tools` imports pandas and numpy and nothing else of ours except the
profiler and the models — **no Streamlit, no Plotly, no LLM**. That is what
makes it testable in isolation and reusable by the agents in a later phase.

One ordering constraint falls out of this: the tools import
`services.profiler`, and `services.visualization` imports the tools. So the
renderer is deliberately **not** re-exported from `app/services/__init__.py`;
import it by module (`from app.services.visualization import build_chart`).

---

## The tool layer

Eleven functions, each validating its own inputs, never mutating the caller's
dataframe, and failing with a `ToolError` subclass whose message names the
offending column and lists the valid alternatives.

| Tool | Returns |
| --- | --- |
| `get_dataset_schema(df)` | `DatasetSchema` — dtypes, inferred roles, nullability, samples |
| `get_column_summary(df, column)` | `ColumnProfile` (delegates to the Phase 1 profiler) |
| `filter_rows(df, conditions)` | A **new** `DataFrame` |
| `group_and_aggregate(df, group_by, aggs)` | A **new** tidy `DataFrame` |
| `calculate_statistics(df, columns, stats)` | `StatisticsResult` |
| `calculate_correlation(df, ...)` | `CorrelationResult` — matrix plus ranked pairs |
| `compare_segments(df, segment, metric)` | `SegmentComparison` — value, share, delta vs baseline |
| `calculate_time_trend(df, date, value)` | `TimeTrendResult` — ordered periods with % change |
| `rank_values(df, column, metric)` | `RankingResult` |
| `detect_outliers(df, column)` | `OutlierResult` — bounds, indices, extreme rows |
| `generate_chart_data(df, chart_type, ...)` | `ChartData` — plot-ready frame plus axis metadata |

Tools that *reshape* data return a DataFrame; tools that *compute an insight*
return a dataclass with `to_dict()` for serialisation.

### Exceptions

`ToolError` is the base. `ColumnNotFoundError` suggests close matches
(`"Column 'revenu' does not exist. Did you mean 'revenue'?"`),
`InvalidColumnTypeError` names the expected and actual type,
`InvalidParameterError` covers malformed arguments, `UnsupportedOperationError`
lists what *is* supported, and `InvalidDataError` covers a non-dataframe input.

Empty results are **not** errors: an empty frame, a filter that matches
nothing, a constant column, or a dataset with no date column all produce a
valid result carrying an explanatory `notes` entry.

### Aggregations

`sum`, `mean`, `median`, `count`, `min`, `max`, `std`, `percentage_change` —
each an `Aggregation` entry in the `AGGREGATIONS` registry, with aliases
(`average`, `total`, `stdev`, …). Adding one is a single call:

```python
register_aggregation(Aggregation(
    "p95", "95th percentile",
    lambda s: float(s.quantile(0.95)),
    measure_template="95th percentile of {measure}",
))
```

Every tool resolves names through the registry, so a new aggregation
immediately works in grouping, statistics, ranking, time trends and charts —
including its axis labels.

`percentage_change` has two readings and both exist: as a scalar aggregation it
is the change from a group's first value to its last; `add_percentage_change()`
gives period-over-period change along an ordered series, and that is what the
time-trend tool puts in its `pct_change` column.

### Filtering

Conditions are `{"column": ..., "operator": ..., "value": ...}` dicts
(`(column, operator, value)` tuples and `FilterCondition` objects also work),
combined with `logic="and"` or `"or"`:

```python
filter_rows(df, [
    {"column": "region",         "operator": "in",           "value": ["Cairo", "Delta"]},
    {"column": "monthly_charge", "operator": "between",      "value": [200, 500]},
    {"column": "signup_date",    "operator": "date_between", "value": ["2023-01-01", "2023-12-31"]},
    {"column": "satisfaction_score", "operator": "not_null"},
])
```

Operators: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `between`, `not_between`,
`date_between`, `in`, `not_in`, `is_null`, `not_null`, plus symbol aliases
(`>=`, `!=`, …). Values are coerced to the column's own type, so `"250"`
compares numerically against a numeric column and `"yes"` against a boolean
one; comparing a numeric column with unparseable text raises rather than
silently matching nothing.

### Time series

`daily`, `weekly`, `monthly`, `quarterly`, `yearly` (with aliases). Output is
always sorted oldest-first and carries `period`, `period_label`, `value`,
`row_count` and `pct_change`. Rows whose date is missing or unparseable are
excluded and *counted* (`rows_missing_date`, `rows_unparseable_date`) with a
note, and `fill_gaps=True` inserts absent periods so the axis is evenly spaced.

---

## The chart engine

`generate_chart_data()` prepares and validates; `app/services/visualization.py`
renders. Six types — line, bar, scatter, histogram, box, correlation heatmap —
dispatched through an **explicit** `RENDERERS` map. There is no "pick something
sensible" fallback: an unusable combination (a scatter plot of two text
columns, a histogram of a category) raises a message naming the problem.

Every figure goes through `apply_theme()`, so all charts share one palette,
font, margin set, grid treatment and hover style. Beyond that:

- **Titles and axes** are generated from the columns and the aggregation:
  `data_usage_gb` + `mean` becomes *"Average data usage (GB) by region"*, with
  axis labels to match.
- **Ordering is explicit.** Time axes are chronological; bar categories are
  ordered by the measure; box categories by median. The renderer is told the
  order and must not re-sort.
- **Hover** is unified along time axes and closest-point elsewhere, and line
  charts carry row counts and period-over-period change in the tooltip.
- **Formatting** switches to SI tick labels (1.2k, 3.4M) once values get large,
  rotates crowded category labels, and fixes the heatmap's colour domain to
  −1…1 so a colour always means the same correlation.
- **Large inputs** are handled: scatter plots sample above 5,000 points and bar
  charts cap categories, both with a note on the chart.

---

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
| **2 — Analytical foundation** | ✅ Tool layer, chart engine, Explore and Visualize pages |
| 3 — Agents | LLM agents and tool calling for natural-language analysis |
| 4 — Detection | Anomaly detection and forecasting |
| 5 — ML Lab | Guided model training and evaluation |

## Limitations

- **No LLM or agent functionality.** `app/agents/` is an empty package and
  `OPENAI_API_KEY` is read but never used. The tool layer is shaped for tool
  calling (plain arguments, serialisable results, corrective error messages)
  but nothing calls it automatically yet.
- Everything is in memory; large files are truncated to `DATAPILOT_MAX_ROWS`
  rather than streamed, and `group_and_aggregate` refuses above 5,000 groups.
- Outlier detection is univariate (IQR or z-score). Multivariate anomalies are
  Phase 4.
- Correlations are Pearson/Spearman/Kendall on up to 40 numeric columns, so
  non-linear relationships are not reported, and correlation is not causation —
  nothing here claims otherwise.
- `filter_rows` is not yet wired into the UI pages; it exists and is tested for
  the agent phase. Explore and Visualize operate on the whole dataset.
- Charts are rendered server-side by Plotly on every rerun; no chart-level
  caching yet, so a very wide heatmap on a large dataset is noticeably slower
  than the rest of the app.
- Ambiguous dates like `03/04/2023` are resolved by pandas, not by Aurevia.
- Excel loading reads one sheet at a time; legacy `.xls` support depends on
  `xlrd` and is best effort.
