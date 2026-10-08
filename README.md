# Aurevia

An agentic data-intelligence platform. Upload a CSV or Excel file and Aurevia
works out what the dataset *is* — its shape, field semantics, and quality
problems — then lets you interrogate and chart it.

> **Status: Phase 4 (reliability and investigation).** Ask a question in plain
> language, follow it up ("what about last quarter?"), or ask *why* something
> changed and have Aurevia investigate it step by step. Every answer is
> validated before it is shown, every figure is computed by Python, and the
> evidence behind each claim is on the page. Forecasting, the ML Lab and the
> visual redesign come later.

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
python -m pytest            # 785 tests
```

---

## The pages

| Page | What it does | Needs an API key |
| --- | --- | --- |
| **Overview** | Load a file, see shape, memory, field types, and data-quality warnings. | no |
| **Explore** | Guided exploration in five tabs: summary, data quality, distributions, categories, relationships. | no |
| **Visualize** | A manual chart builder whose controls adapt to the chart type. | no |
| **Ask Aurevia** | Ask in plain language; get a plan, computed figures, a chart, a table, an explanation and the evidence. Follow-up questions use session context. | **yes** |
| **Investigate** | Ask *why* something changed. Aurevia verifies the premise, decomposes the change across several dimensions, checks the timing and ranks the contributors. | **yes** |

All five share one loaded dataset, one sidebar and one analytical session
(`app/services/session_manager.py`), so switching pages never reloads the file
and never loses the conversation. Loading a *different* dataset clears the
conversation, history and results — context from one dataset must never be
applied to another. Without an API key the first three pages work exactly as
before, and Ask Aurevia and Investigate explain what they need.

---

## Configuration

Settings are read from the environment, seeded from a `.env` file at the project
root. **No API key is required to start**: a missing, blank, or placeholder key is
treated as "not configured" rather than an error, and only Ask Aurevia needs
one.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | _(unset)_ | Required for Ask Aurevia and Investigate. |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Must support structured output. |
| `AUREVIA_LLM_TIMEOUT` | `45` | Seconds before an AI request is abandoned. |
| `AUREVIA_LLM_MAX_RETRIES` | `2` | SDK retries for transient failures. |
| `AUREVIA_LLM_TEMPERATURE` | `0.1` | Planning is structured, not creative. |
| `AUREVIA_LLM_MAX_OUTPUT_TOKENS` | `1200` | Caps response size, and cost. |
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
  agents/          The LLM agents
    planner.py     Question -> AnalysisPlan, via structured output
    insights.py    Computed result -> written answer
    investigation_planner.py  Investigation plan, and its write-up
    suggestions.py Dataset-aware suggested questions (no LLM)
  tools/           The deterministic analysis layer
    exceptions.py  ToolError and its subclasses
    validation.py  Shared column / type / parameter checks
    aggregations.py The aggregation registry
    filters.py     Filter operators and filter_rows()
    timeseries.py  Frequencies and calculate_time_trend()
    analysis.py    Schema, summaries, grouping, stats, correlation,
                   segments, ranking, outliers
    charts.py      generate_chart_data() and the chart-type registry
    resolution.py  Safe natural-language -> column mapping
  services/
    data_loader.py CSV / XLSX loading and error handling
    profiler.py    Type inference, statistics, quality warnings
    visualization.py The Plotly rendering engine
    llm_service.py  The OpenAI client: timeouts, typed errors, availability
    dataset_context.py The compact schema sent to the LLM (never data)
    plan_validator.py  Checks planner output against the real dataset
    analysis_executor.py Intent router -> deterministic tools
    grounding.py    Verifies the LLM quoted only computed figures
    result_validator.py  Did we answer the question that was asked?
    context_manager.py   When to carry conversational context, and when not
    session_manager.py   All session state, scoped to one dataset
    investigation_engine.py  Premise check and contribution analysis
    investigation_validator.py  Checks an investigation plan
    ask_pipeline.py      The Ask orchestrator, with targeted retries
    investigation_pipeline.py  The Investigate orchestrator
  models/
    profile.py     DatasetProfile / ColumnProfile / DataQualityWarning
    plans.py       AnalysisPlan / Intent / PlanFilter
    results.py     AnalysisResult / Insight
    validation.py  ValidationResult / RetryStage / EvidenceStrength
    context.py     AnalyticalContext / AnalysisHistoryItem
    investigation.py  InvestigationPlan / Finding / Result
  ui/
    state.py       Shared session state and the data-source sidebar
    overview.py    The Overview page
    explore.py     The Explore page
    visualize.py   The Visualize page
    ask.py         The Ask Aurevia page
    investigate.py The Investigate page
    evidence.py    The shared Evidence view and strength labels
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
makes it testable in isolation, and it is what the Phase 3 agents sit on top
of: the executor calls these tools, and the LLM calls nothing at all.

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

## The AI layer

```text
question
   -> planner (LLM)        AnalysisPlan        intent + field references
   -> plan_validator       repaired plan       or a clarification / refusal
   -> analysis_executor    AnalysisResult      every number computed here
   -> chart selection      chart spec          matched to the result shape
   -> insights (LLM)       Insight             grounding-checked prose
```

Two LLM calls per question, and no more: one to plan, one to explain. The model
never computes, never picks a Python function, and never sees a data row.

### What each side does

| The LLM does | Python does |
| --- | --- |
| Read intent; name the intent, metric, dimensions, filters and time field | Resolve every reference against the real columns |
| Write the user-facing plan steps | Filter, aggregate, correlate, rank, resample |
| Interpret the computed figures in prose | Compute every figure, percentage and coefficient |
| Suggest follow-up questions | Choose the chart from the result shape |

### The plan

`AnalysisPlan` (`app/models/plans.py`) is filled by the model as OpenAI
structured output, so there is no prose to parse and no path from model text
into control flow:

| Field | Meaning |
| --- | --- |
| `intent` | One of twelve: `summary`, `ranking`, `comparison`, `segmentation`, `trend`, `distribution`, `correlation`, `time_comparison`, `percentage_change`, `count`, `anomaly`, `dataset_question` |
| `metric` | The numeric column to measure, or null for row counts |
| `dimensions` | Categorical columns to group by |
| `filters` | `PlanFilter` triples; values arrive as strings and are coerced against the real column |
| `aggregation` | A name from the Phase 2 registry |
| `time_column`, `time_granularity`, `periods` | Date field, calendar frequency, and how many recent periods to scope |
| `comparison`, `sort_direction`, `limit` | How to contrast and order the result |
| `visualization` | A chart *suggestion*, re-checked against the result |
| `steps` | The user-facing execution summary |
| `requires_clarification`, `clarification_question` | Set when the question is genuinely ambiguous |

### Validation: nothing the model says is trusted

`plan_validator.py` checks every field against the loaded dataframe and either
repairs it (recording a note the UI shows), asks the user a question, or
refuses. It will:

- resolve column references through the ladder in `tools/resolution.py` —
  exact, normalised, curated alias, token overlap, then high-threshold fuzzy —
  and **ask rather than guess** below that;
- refuse a non-numeric metric, naming usable alternatives;
- fall back on an unsupported aggregation, granularity or chart type;
- supply a date column for a time question, or refuse when none exists;
- dry-run every filter and refuse one that matches nothing, listing values that
  *do* exist;
- drop a grouping column that is near-unique, since grouping by an ID
  summarises nothing;
- clamp `limit`, and widen a one-row ranking so there is something to compare;
- catch a **metric substitution** — the planner quietly answering about
  `monthly_charge` when the user asked for "profit margin".

### Numerical grounding

The insight agent is given the computed result and nothing else, and told to
quote only figures it contains. Because a prompt is not a control,
`grounding.py` then extracts every number from the written answer and matches it
against the values Python produced — tolerant of formatting (`1.24M`,
`1,240,000`, `36.4%` all match), strict about value. Anything unmatched is
listed on the answer as unverified, and the user is pointed at the table.

Rates are computed as percentages by Python for the same reason: asking "which
contract type has the highest churn rate?" yields both `0.2386` and `23.86`, so
the natural phrasing needs no arithmetic from the model.

### Token discipline

The model receives a bounded schema summary — column names, inferred types,
ranges, up to eight category examples, a date span — and **no data rows**. For
the 5,000-row sample dataset that is roughly 460 tokens, and it barely grows
with the data: 500× the rows produces the same prompt.

### Failure handling

Every failure arrives as a sentence, with the technical detail logged and no
stack trace on screen: a missing key, a timeout, a rate limit, a rejected key, a
model that does not support structured output, malformed planner output, an
unsupported operation, a nonexistent field, an empty filtered dataset, an
unparseable date column, or an impossible comparison. When the explanation call
fails but the computation succeeded, the figures and chart are still shown.

---

## Reliability: validating an answer

Phase 3 trusted the executor's success flag. Phase 4 adds a stage that asks a
harder question — *did we answer what was asked?* — because an analysis can run
cleanly and still be grouped by the wrong field, miss a requested filter, rank
the wrong way round, or be explained with a figure nobody computed.

`result_validator.py` runs **14 deterministic checks** comparing the plan
against the result it produced: columns exist, the result is non-empty, the
metric is present, the grouping was applied, the filters were applied and kept
rows, the shape matches the intent, the ranking is in the stated order, the
requested time window was used, every figure is finite, the chart actually
builds, and the explanation quotes only computed numbers.

One optional LLM check covers the single judgement code cannot make: *does the
prose address the question?* It is **gated**, not routine — it runs only when a
deterministic check already raised a warning, the answer quotes no figure at all
(how a model dodges a question), or the answer is long enough for drift to
accumulate. A clean, figure-quoting answer costs two API calls, not three.

### Targeted retries

Each finding names the stage that could fix it, so a failure retries the
*cheapest* stage rather than the whole pipeline:

| Failure | Retries | Cost |
| --- | --- | --- |
| Invented figure, off-topic prose | the explanation, with a stricter prompt | one call |
| Unbuildable or empty chart | the chart, replaced from the result shape | **no call** |
| Wrong columns or analysis | the plan, with the validation failure as feedback | one call |
| Broken computation | nothing — reported as a failure | none |

Retries are capped at two, and a retry genuinely changes the request: the
regenerated explanation carries a reminder naming what was rejected. When
retries are exhausted Aurevia says so, and says why —

> Aurevia could not confidently complete this analysis from the available data.
> The explanation contains 2 figures that were not computed: …

— and the computed figures are still shown, marked as unverified. An answer is
never fabricated to fill the gap.

### Evidence strength

Three qualitative labels, assigned from concrete facts rather than an invented
score: **Strong** (every check passed, sample large enough), **Moderate** (a
warning fired or the sample is small), **Limited** (a check failed or there are
too few records). The reason is printed beside the label, and the Evidence panel
lists the fields, filters, aggregation, row counts and computed values behind
the answer.

---

## Conversational context

"What about last quarter?" is meaningless without the previous question. "Show
the distribution of satisfaction scores" is meaningless *with* it — and that
direction is the dangerous one, because inheriting `revenue by region` into an
unrelated question produces a confidently wrong answer rather than a confused
one.

So context is **opt-in**. `AnalyticalContext` holds the analytical state of the
last answered question — metric, dimensions, filters, time range, referenced
values — and it reaches the planner only when the question shows a
**continuation signal**:

| Signal | Example |
| --- | --- |
| Linking phrase | "what about…", "and for…", "only for…", "break that down by…" |
| Back-reference | "compare **that** with…", "show **it** monthly" |
| Time scope alone | "last quarter?" — a period with nothing to apply it to |
| Entity fragment | "Delta?" — a category value with nothing to measure |

Shortness alone is deliberately *not* a signal: "What does this dataset
contain?" is brief and names no field, yet it is a complete question.

Even under continuation, inheritance only fills **gaps**. Anything the new
question names explicitly wins; naming a different measure drops the old one; a
correlation or dataset question resets the measure entirely; and a filter is
carried only when the new question adds none of its own. Every carry-over is
reported on the page ("Continued with the measure `revenue`").

The context is stamped with a **dataset key** — a hash of the shape, column
names and dtypes. Loading a different file clears the context, the history, the
last answer and any investigation, in one place (`SessionManager.set_dataset`).
Reloading the *same* file keeps the conversation.

---

## Investigate

Where Ask answers a defined question, Investigate takes a *why* and works
through it:

```text
question
   -> investigation planner (LLM)   metric, periods, candidate dimensions
   -> validate (Python)             repaired plan, or a refusal
   -> premise check (Python)        did the change actually happen?
   -> contribution analysis         decompose across 3-5 dimensions
   -> timing (Python)               when inside the window did it move?
   -> rank evidence (Python)
   -> write-up (LLM)                grounding-checked
```

**Exactly two LLM calls**, however many dimensions get decomposed. The planner
is not consulted per dimension, and the engine never calls out.

### The premise comes first

"Why did revenue decline last quarter?" asserts a decline. If revenue rose,
answering the question as asked would be fabrication. So the direction is
computed first and checked against the claim — read both from the question's
wording and from the planner — and a mismatch stops the investigation:

> `revenue` did not increase in 2024Q4: it decreased by 73.62% compared with
> 2024Q3. I can investigate what contributed to the decrease instead.

There is no contributor analysis of a change that did not happen.

### Contribution: two readings, because one of them lies

For an additive metric, each category gets its baseline value, its comparison
value and its absolute change. Two shares are then reported, because a single
number cannot honestly carry both meanings:

- **Share of the net change** — `category_change / overall_change`. The
  intuitive reading, and correct when categories move together. It is
  *withheld* when they do not: if one category falls 500 while another rises
  400, the net is −100 and the faller "contributed 500% of the decline". When
  the net change is under a quarter of total movement, this column is dropped
  and the breakdown is marked as offsetting.
- **Share of total movement** — `|category_change| / Σ|category_change|`.
  Always defined, always 0–100%, and the honest answer to "where did the
  movement happen".

A non-additive aggregation (a mean, a rate) does not decompose additively at
all, so only each category's own change is shown, with a note saying why.
Categories that moved *against* the overall direction are listed separately
rather than averaged away.

### Bounded, and transparent about its limits

An investigation decomposes at most five dimensions and eight categories each,
and inspects at most twelve sub-periods. Dimension choice is screened by
deterministic rules *before* the planner's preference is honoured: identifiers,
free text, dates, numeric measures, single-valued and near-unique columns are
rejected with a reason the page shows. Deterministic cautions cover thin
periods, uneven coverage, excluded rows, missing values and small category
samples:

> The region finding for "Red Sea" is based on only 3 records and should be
> interpreted cautiously.

The write-up may describe concentration and association; it is instructed not to
claim cause. "The decline was concentrated in the Western region" is allowed;
"customers left because…" is not.

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
| **3 — AI intelligence layer** | ✅ Planner, validator, executor, grounding, Ask Aurevia |
| **4 — Reliability & investigation** | ✅ Result validator, targeted retries, context, Investigate, Evidence |
| 5 — Detection | Anomaly detection and forecasting |
| 6 — ML Lab & UI | Guided model training, and the premium Aurevia redesign |

## Limitations

- **Context is rule-based, not understood.** Continuation is detected by
  phrase patterns, back-references, time phrases and named values. A follow-up
  phrased unusually ("same thing but for the other one") will be read as a new
  question and answered from the schema alone — safe, but not clever.
- **Two turns of depth, not a dialogue.** Context carries the *last* answered
  question's state, not a history of reasoning. "Compare that with the first
  one I asked about" is not supported.
- **Retries are capped at two, and one failure is not retryable.** A broken
  computation is reported rather than re-run, because re-running it would fail
  identically.
- **Validation is structural, not statistical.** The checks confirm that the
  analysis matches the plan and the figures are quoted faithfully. They do not
  assess whether the analysis was the *right* one to run, beyond intent/shape
  agreement, and there is no significance testing.
- **Investigations compare two adjacent periods.** Not three, not a baseline
  against a seasonal average, and not year-on-year unless the granularity makes
  the previous period a year. Contribution analysis is single-dimension at a
  time: it will not find that the decline sits in *Fiber Internet in Cairo
  specifically*.
- **"Why" is answered as "where".** The engine measures concentration and
  association. Nothing in Aurevia establishes cause, and the write-up is
  instructed not to claim it.
- **Twelve intents, not arbitrary analysis.** Multi-dimensional pivots,
  cohort analysis, statistical testing, custom formulas and multi-step
  reasoning are out of scope. A question outside the set is answered with the
  closest supported intent or a clarification.
- **Filters are single-column conditions** combined with AND. No nested
  boolean logic.
- **`percentage_change` compares the first and last period** of the window, not
  a fitted trend.
- Ask Aurevia needs a model that supports OpenAI structured output; a model
  without it fails with a clear message rather than degrading.
- Everything is in memory; large files are truncated to `DATAPILOT_MAX_ROWS`
  rather than streamed, and `group_and_aggregate` refuses above 5,000 groups.
- Outlier detection is univariate (IQR or z-score). Multivariate anomalies are
  Phase 5.
- Correlations are Pearson/Spearman/Kendall on up to 40 numeric columns, so
  non-linear relationships are not reported, and correlation is not causation —
  nothing here claims otherwise.
- `filter_rows` is reached through Ask Aurevia (the planner emits filters), but
  Explore and Visualize still operate on the whole dataset; neither page has a
  filter control.
- Charts are rendered server-side by Plotly on every rerun; no chart-level
  caching yet, so a very wide heatmap on a large dataset is noticeably slower
  than the rest of the app.
- Ambiguous dates like `03/04/2023` are resolved by pandas, not by Aurevia.
- Excel loading reads one sheet at a time; legacy `.xls` support depends on
  `xlrd` and is best effort.
