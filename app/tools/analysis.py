"""The deterministic analysis tools.

This is the layer future agents will call: every function takes a dataframe plus
plain-Python arguments, validates them, and returns either a new dataframe or a
structured dataclass. Nothing here mutates its input, reads global state, or
depends on Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from app.models.profile import ColumnProfile, FieldType
from app.services.profiler import infer_field_type, profile_column
from app.tools.aggregations import (
    Aggregation,
    resolve_aggregation,
    supported_aggregations,
)
from app.tools.exceptions import (
    InvalidParameterError,
    UnsupportedOperationError,
)
from app.tools.validation import (
    categorical_columns,
    datetime_columns,
    numeric_columns,
    require_column,
    require_columns,
    require_dataframe,
    require_numeric_column,
    require_positive_int,
)

#: Correlation methods pandas supports.
CORRELATION_METHODS = ("pearson", "spearman", "kendall")

#: Outlier detection strategies.
OUTLIER_METHODS = ("iqr", "zscore")

#: Default statistics reported by :func:`calculate_statistics`.
DEFAULT_STATISTICS = ("count", "mean", "median", "std", "min", "max", "sum")

#: Hard cap on grouping cardinality, to keep results renderable.
MAX_GROUPS = 5_000


# --------------------------------------------------------------------------- #
# 1. Schema
# --------------------------------------------------------------------------- #

@dataclass
class ColumnSchema:
    """One column's structural description (cheap -- no heavy statistics)."""

    name: str
    dtype: str
    inferred_type: str
    nullable: bool
    missing_count: int
    missing_pct: float
    unique_count: int
    sample_values: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "inferred_type": self.inferred_type,
            "nullable": self.nullable,
            "missing_count": self.missing_count,
            "missing_pct": self.missing_pct,
            "unique_count": self.unique_count,
            "sample_values": list(self.sample_values),
        }


@dataclass
class DatasetSchema:
    """The shape and field roles of a dataset."""

    name: str
    row_count: int
    column_count: int
    columns: list[ColumnSchema] = field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def _names_where(self, *types: FieldType) -> list[str]:
        wanted = {t.value for t in types}
        return [c.name for c in self.columns if c.inferred_type in wanted]

    @property
    def numeric_columns(self) -> list[str]:
        return self._names_where(FieldType.NUMERIC, FieldType.INTEGER)

    @property
    def categorical_columns(self) -> list[str]:
        return self._names_where(FieldType.CATEGORICAL, FieldType.TEXT)

    @property
    def boolean_columns(self) -> list[str]:
        return self._names_where(FieldType.BOOLEAN)

    @property
    def date_columns(self) -> list[str]:
        return self._names_where(FieldType.DATETIME)

    @property
    def identifier_columns(self) -> list[str]:
        return self._names_where(FieldType.IDENTIFIER)

    def column(self, name: str) -> ColumnSchema | None:
        return next((c for c in self.columns if c.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": [c.to_dict() for c in self.columns],
            "roles": {
                "numeric": self.numeric_columns,
                "categorical": self.categorical_columns,
                "boolean": self.boolean_columns,
                "date": self.date_columns,
                "identifier": self.identifier_columns,
            },
        }


def get_dataset_schema(
    df: pd.DataFrame,
    *,
    name: str = "dataset",
    sample_size: int = 3,
) -> DatasetSchema:
    """Describe the dataset's columns, dtypes, inferred roles, and nullability.

    Deliberately cheap: this is the orientation call an agent makes first, so it
    skips the expensive statistics that :func:`get_column_summary` provides.
    """
    require_dataframe(df)
    samples = require_positive_int(sample_size, "sample_size", maximum=20)

    rows = len(df)
    columns: list[ColumnSchema] = []

    for raw_name in df.columns:
        column = str(raw_name)
        series = df[raw_name]
        missing = int(series.isna().sum())
        try:
            unique = int(series.nunique(dropna=True))
        except TypeError:  # unhashable values
            unique = int(series.astype(str).nunique(dropna=True))

        columns.append(
            ColumnSchema(
                name=column,
                dtype=str(series.dtype),
                inferred_type=infer_field_type(series, column).value,
                nullable=missing > 0,
                missing_count=missing,
                missing_pct=round(100.0 * missing / rows, 4) if rows else 0.0,
                unique_count=unique,
                sample_values=[_jsonable(v) for v in series.dropna().head(samples)],
            )
        )

    return DatasetSchema(
        name=name,
        row_count=rows,
        column_count=int(df.shape[1]),
        columns=columns,
    )


def _jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars into plain Python for serialisation."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NaT or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


# --------------------------------------------------------------------------- #
# 2. Column summary
# --------------------------------------------------------------------------- #

def get_column_summary(df: pd.DataFrame, column: str) -> ColumnProfile:
    """Full statistics for one column.

    Delegates to the Phase 1 profiler so the Overview page and the tools layer
    can never disagree about what a column is.
    """
    require_dataframe(df)
    name = require_column(df, column)
    return profile_column(df[name], name)


# --------------------------------------------------------------------------- #
# 3. Grouping
# --------------------------------------------------------------------------- #

def _normalize_agg_spec(
    df: pd.DataFrame,
    aggregations: Any,
) -> list[tuple[str, Aggregation]]:
    """Normalise the many shapes an aggregation spec can take.

    Accepts ``{"revenue": "sum"}``, ``{"revenue": ["sum", "mean"]}``,
    ``[("revenue", "sum")]``, or a bare ``"count"`` (which counts rows).
    """
    if aggregations is None:
        raise InvalidParameterError(
            "'aggregations' is required, e.g. {'revenue': 'sum'} or 'count'."
        )

    pairs: list[tuple[str, Any]] = []

    if isinstance(aggregations, str):
        aggregator = resolve_aggregation(aggregations)
        if aggregator.name != "count":
            raise InvalidParameterError(
                f"A bare aggregation name is only supported for 'count'; "
                f"for '{aggregator.name}' pass a column, e.g. "
                f"{{'<column>': '{aggregator.name}'}}."
            )
        return [("__rows__", aggregator)]

    if isinstance(aggregations, dict):
        for column, spec in aggregations.items():
            for name in spec if isinstance(spec, (list, tuple, set)) else [spec]:
                pairs.append((column, name))
    elif isinstance(aggregations, (list, tuple)):
        for entry in aggregations:
            if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
                raise InvalidParameterError(
                    "Each aggregation entry must be a (column, aggregation) pair, "
                    f"got {entry!r}."
                )
            pairs.append((entry[0], entry[1]))
    else:
        raise InvalidParameterError(
            f"'aggregations' must be a dict, a list of pairs, or 'count', "
            f"got {type(aggregations).__name__}."
        )

    if not pairs:
        raise InvalidParameterError("'aggregations' must name at least one measure.")

    resolved: list[tuple[str, Aggregation]] = []
    for column, name in pairs:
        aggregator = resolve_aggregation(name)
        target = require_column(df, column)
        if aggregator.numeric_only:
            require_numeric_column(df, target)  # raises with a clear message
        resolved.append((target, aggregator))
    return resolved


def group_and_aggregate(
    df: pd.DataFrame,
    group_by: str | list[str],
    aggregations: Any = "count",
    *,
    sort_by: str | None = None,
    ascending: bool = False,
    top_n: int | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Group rows and aggregate measures, returning a **new** tidy dataframe.

    Output columns are the grouping keys followed by ``<column>_<aggregation>``
    (or ``row_count`` when counting rows). An empty input yields an empty frame
    with the correct columns rather than an error.
    """
    require_dataframe(df)
    keys = require_columns(df, group_by if isinstance(group_by, list) else [group_by])
    if len(set(keys)) != len(keys):
        raise InvalidParameterError(f"Duplicate grouping columns: {keys}.")

    specs = _normalize_agg_spec(df, aggregations)
    output_names = [
        "row_count" if column == "__rows__" else f"{column}_{agg.column_suffix}"
        for column, agg in specs
    ]

    if df.empty:
        return pd.DataFrame(columns=[*keys, *output_names])

    working = df.loc[:, list(dict.fromkeys([*keys, *(c for c, _ in specs if c != "__rows__")]))]
    grouped = working.groupby(keys, dropna=dropna, observed=True, sort=True)

    group_count = grouped.ngroups
    if group_count > MAX_GROUPS:
        raise InvalidParameterError(
            f"Grouping by {keys} produces {group_count:,} groups, which exceeds the "
            f"{MAX_GROUPS:,} limit. Filter the data or group by a coarser column."
        )

    columns: dict[str, pd.Series] = {}
    for (column, aggregator), output in zip(specs, output_names):
        if column == "__rows__":
            columns[output] = grouped.size()
        else:
            columns[output] = grouped[column].apply(aggregator.apply)

    result = pd.DataFrame(columns).reset_index()
    # Grouping keys come back with their original names; make sure of the order.
    result = result[[*keys, *output_names]]

    order_column = sort_by or output_names[0]
    if order_column not in result.columns:
        raise InvalidParameterError(
            f"Cannot sort by '{order_column}'; available: {list(result.columns)}."
        )
    result = result.sort_values(
        order_column, ascending=ascending, kind="stable"
    ).reset_index(drop=True)

    if top_n is not None:
        result = result.head(require_positive_int(top_n, "top_n")).reset_index(drop=True)

    return result


# --------------------------------------------------------------------------- #
# 4. Statistics
# --------------------------------------------------------------------------- #

@dataclass
class StatisticsResult:
    """Requested statistics for one or more numeric columns."""

    statistics: list[str]
    columns: list[str] = field(default_factory=list)
    values: dict[str, dict[str, float]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    row_count: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.values

    def to_frame(self) -> pd.DataFrame:
        """One row per column, one column per statistic."""
        if not self.values:
            return pd.DataFrame(columns=["column", *self.statistics])
        frame = pd.DataFrame(self.values).T.reset_index(names="column")
        return frame[["column", *[s for s in self.statistics if s in frame.columns]]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "statistics": list(self.statistics),
            "columns": list(self.columns),
            "values": self.values,
            "skipped": self.skipped,
        }


def calculate_statistics(
    df: pd.DataFrame,
    columns: str | list[str] | None = None,
    statistics: list[str] | None = None,
) -> StatisticsResult:
    """Compute `statistics` for `columns` (all numeric columns by default).

    Non-numeric columns are reported in ``skipped`` with a reason instead of
    raising, so a caller can ask for "everything" without pre-filtering.
    """
    require_dataframe(df)

    # `None` means "the defaults"; an explicit empty list is a caller mistake.
    requested = list(DEFAULT_STATISTICS if statistics is None else statistics)
    if not requested:
        raise InvalidParameterError("'statistics' must name at least one statistic.")
    aggregators = [resolve_aggregation(name) for name in requested]
    names = [a.name for a in aggregators]

    if columns is None:
        targets = numeric_columns(df)
    else:
        targets = require_columns(df, columns if isinstance(columns, list) else [columns])

    result = StatisticsResult(statistics=names, row_count=int(len(df)))

    for column in targets:
        series = df[column]
        if pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series):
            numeric = series
        else:
            try:
                numeric = require_numeric_column(df, column)
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                result.skipped[column] = str(exc)
                continue

        clean = pd.to_numeric(numeric, errors="coerce").dropna()
        if clean.empty:
            result.skipped[column] = "No numeric values to summarise."
            continue

        result.columns.append(column)
        result.values[column] = {
            aggregator.name: _as_float(aggregator.apply(clean))
            for aggregator in aggregators
        }

    return result


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


# --------------------------------------------------------------------------- #
# 5. Correlation
# --------------------------------------------------------------------------- #

@dataclass
class CorrelationResult:
    """A correlation matrix plus the notable pairs extracted from it."""

    method: str
    columns: list[str] = field(default_factory=list)
    matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    pairs: list[dict[str, Any]] = field(default_factory=list)
    threshold: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.matrix.empty

    @property
    def strongest(self) -> dict[str, Any] | None:
        return self.pairs[0] if self.pairs else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "columns": list(self.columns),
            "threshold": self.threshold,
            "matrix": {} if self.matrix.empty else self.matrix.round(4).to_dict(),
            "pairs": list(self.pairs),
            "notes": list(self.notes),
        }


def calculate_correlation(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    method: str = "pearson",
    threshold: float = 0.0,
    min_periods: int = 3,
) -> CorrelationResult:
    """Correlate the numeric columns and list the pairs above `threshold`.

    Fewer than two usable numeric columns yields an empty result with an
    explanatory note, not an error.
    """
    require_dataframe(df)

    if not isinstance(method, str) or method.strip().lower() not in CORRELATION_METHODS:
        raise UnsupportedOperationError("correlation method", method, CORRELATION_METHODS)
    method_name = method.strip().lower()

    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise InvalidParameterError(f"'threshold' must be a number, got {threshold!r}.")
    if not 0.0 <= float(threshold) <= 1.0:
        raise InvalidParameterError(
            f"'threshold' must be between 0 and 1, got {threshold}."
        )

    if columns is None:
        targets = numeric_columns(df)
    else:
        targets = require_columns(df, columns)
        for column in targets:
            require_numeric_column(df, column)

    result = CorrelationResult(method=method_name, threshold=float(threshold))

    numeric = (
        df.loc[:, targets].apply(pd.to_numeric, errors="coerce") if targets
        else pd.DataFrame()
    )
    # A constant column has no correlation with anything; drop it with a note.
    if not numeric.empty:
        constant = [c for c in numeric.columns if numeric[c].nunique(dropna=True) <= 1]
        if constant:
            numeric = numeric.drop(columns=constant)
            result.notes.append(
                "Ignored constant column(s): " + ", ".join(constant) + "."
            )

    if numeric.shape[1] < 2:
        result.notes.append(
            "At least two numeric columns with variation are needed to correlate."
        )
        return result

    if int(numeric.notna().all(axis=1).sum()) < min_periods:
        result.notes.append(
            f"Fewer than {min_periods} complete rows, so correlations are unreliable."
        )

    matrix = numeric.corr(method=method_name, min_periods=min_periods)
    result.columns = [str(c) for c in matrix.columns]
    result.matrix = matrix

    names = list(matrix.columns)
    pairs: list[dict[str, Any]] = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            value = matrix.loc[left, right]
            if pd.isna(value) or abs(float(value)) < float(threshold):
                continue
            correlation = round(float(value), 4)
            pairs.append(
                {
                    "left": str(left),
                    "right": str(right),
                    "correlation": correlation,
                    "direction": "positive" if correlation > 0 else "negative",
                    "strength": _correlation_strength(abs(correlation)),
                }
            )
    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    result.pairs = pairs
    return result


def _correlation_strength(magnitude: float) -> str:
    if magnitude >= 0.9:
        return "very strong"
    if magnitude >= 0.7:
        return "strong"
    if magnitude >= 0.4:
        return "moderate"
    if magnitude >= 0.2:
        return "weak"
    return "negligible"


# --------------------------------------------------------------------------- #
# 6. Segment comparison
# --------------------------------------------------------------------------- #

@dataclass
class SegmentComparison:
    """One metric compared across the values of a categorical column."""

    segment_column: str
    metric_column: str | None
    aggregation: str
    baseline: str
    baseline_value: float | None = None
    #: Columns: ``segment``, ``row_count``, ``value``, ``share_pct``,
    #: ``difference``, ``difference_pct``.
    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.dataframe.empty

    @property
    def best(self) -> dict[str, Any] | None:
        if self.dataframe.empty:
            return None
        return self.dataframe.iloc[0].to_dict()

    @property
    def worst(self) -> dict[str, Any] | None:
        if self.dataframe.empty:
            return None
        return self.dataframe.iloc[-1].to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_column": self.segment_column,
            "metric_column": self.metric_column,
            "aggregation": self.aggregation,
            "baseline": self.baseline,
            "baseline_value": self.baseline_value,
            "segments": self.dataframe.to_dict("records"),
            "notes": list(self.notes),
        }


def compare_segments(
    df: pd.DataFrame,
    segment_column: str,
    metric_column: str | None = None,
    *,
    aggregation: str = "mean",
    segments: list[Any] | None = None,
    baseline: str | Any = "overall",
    top_n: int | None = None,
) -> SegmentComparison:
    """Compare a metric across the values of `segment_column`.

    Each segment reports its aggregated value, its share of the total, and its
    difference from a baseline -- either the overall dataset (``"overall"``, the
    default) or a named segment.
    """
    require_dataframe(df)
    segment_name = require_column(df, segment_column)
    aggregator = resolve_aggregation("count" if metric_column is None else aggregation)

    metric_name: str | None = None
    if metric_column is not None:
        metric_name = require_column(df, metric_column)
        if aggregator.numeric_only:
            require_numeric_column(df, metric_name)

    working = df if segments is None else _restrict_segments(df, segment_name, segments)

    result = SegmentComparison(
        segment_column=segment_name,
        metric_column=metric_name,
        aggregation=aggregator.name,
        baseline=str(baseline),
    )

    if working.empty:
        result.notes.append("No rows matched the requested segments.")
        result.dataframe = pd.DataFrame(
            columns=[
                "segment", "row_count", "value", "share_pct",
                "difference", "difference_pct",
            ]
        )
        return result

    aggregated = group_and_aggregate(
        working,
        segment_name,
        "count" if metric_name is None else {metric_name: aggregator.name},
        ascending=False,
    )
    value_column = aggregated.columns[-1]
    frame = aggregated.rename(columns={segment_name: "segment", value_column: "value"})

    counts = (
        working.groupby(segment_name, dropna=True, observed=True)
        .size()
        .rename("row_count")
    )
    if "row_count" not in frame.columns:
        frame = frame.merge(
            counts.reset_index().rename(columns={segment_name: "segment"}),
            on="segment",
            how="left",
        )

    # Baseline.
    if isinstance(baseline, str) and baseline.strip().lower() == "overall":
        series = working[metric_name] if metric_name else working.iloc[:, 0]
        baseline_value = _as_float(aggregator.apply(series))
        result.baseline = "overall"
    else:
        match = frame.loc[frame["segment"].astype(str) == str(baseline), "value"]
        if match.empty:
            available = ", ".join(map(str, frame["segment"].head(10)))
            raise InvalidParameterError(
                f"Baseline segment {baseline!r} not found in '{segment_name}'. "
                f"Available: {available}."
            )
        baseline_value = _as_float(match.iloc[0])
        result.baseline = str(baseline)

    result.baseline_value = baseline_value

    total = _as_float(pd.to_numeric(frame["value"], errors="coerce").sum())
    frame["share_pct"] = (
        pd.to_numeric(frame["value"], errors="coerce") / total * 100.0
        if total not in (0.0, float("nan")) else np.nan
    )
    frame["difference"] = pd.to_numeric(frame["value"], errors="coerce") - baseline_value
    frame["difference_pct"] = (
        frame["difference"] / abs(baseline_value) * 100.0
        if baseline_value not in (0.0, None) and not pd.isna(baseline_value)
        else np.nan
    )

    frame = frame.sort_values("value", ascending=False, kind="stable").reset_index(drop=True)
    if top_n is not None:
        frame = frame.head(require_positive_int(top_n, "top_n")).reset_index(drop=True)

    result.dataframe = frame[
        ["segment", "row_count", "value", "share_pct", "difference", "difference_pct"]
    ]
    return result


def _restrict_segments(
    df: pd.DataFrame, column: str, segments: list[Any]
) -> pd.DataFrame:
    if not isinstance(segments, (list, tuple, set)):
        raise InvalidParameterError(
            f"'segments' must be a list of values, got {type(segments).__name__}."
        )
    wanted = {str(s).strip() for s in segments}
    if not wanted:
        raise InvalidParameterError("'segments' must contain at least one value.")
    mask = df[column].astype("string").str.strip().isin(wanted).fillna(False)
    return df.loc[mask].copy()


# --------------------------------------------------------------------------- #
# 7. Ranking
# --------------------------------------------------------------------------- #

@dataclass
class RankingResult:
    """An ordered leaderboard of values or groups."""

    column: str
    metric_column: str | None
    aggregation: str | None
    ascending: bool
    #: Columns: ``rank``, ``<column>``, ``value`` (and ``row_count`` when grouped).
    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_candidates: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.dataframe.empty

    @property
    def top(self) -> dict[str, Any] | None:
        return self.dataframe.iloc[0].to_dict() if not self.dataframe.empty else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "metric_column": self.metric_column,
            "aggregation": self.aggregation,
            "ascending": self.ascending,
            "total_candidates": self.total_candidates,
            "entries": self.dataframe.to_dict("records"),
            "notes": list(self.notes),
        }


def rank_values(
    df: pd.DataFrame,
    column: str,
    metric_column: str | None = None,
    *,
    aggregation: str = "sum",
    top_n: int = 10,
    ascending: bool = False,
) -> RankingResult:
    """Rank the values of `column` by an aggregated metric.

    Three modes, chosen by the arguments rather than by guessing:

    - ``rank_values(df, "region")`` -- regions by row count;
    - ``rank_values(df, "region", "revenue")`` -- regions by total revenue;
    - ``rank_values(df, "revenue", aggregation=None)`` -- the individual rows
      with the largest ``revenue``.
    """
    require_dataframe(df)
    name = require_column(df, column)
    limit = require_positive_int(top_n, "top_n")

    # Row-level ranking.
    if aggregation is None:
        if metric_column is not None:
            raise InvalidParameterError(
                "With aggregation=None the ranking is row-level, so "
                "'metric_column' must not be given."
            )
        return _rank_rows(df, name, limit=limit, ascending=ascending)

    aggregator = resolve_aggregation("count" if metric_column is None else aggregation)
    metric_name: str | None = None
    if metric_column is not None:
        metric_name = require_column(df, metric_column)
        if aggregator.numeric_only:
            require_numeric_column(df, metric_name)

    result = RankingResult(
        column=name,
        metric_column=metric_name,
        aggregation=aggregator.name,
        ascending=ascending,
    )

    if df.empty:
        result.dataframe = pd.DataFrame(columns=["rank", name, "value", "row_count"])
        result.notes.append("The dataset is empty, so there is nothing to rank.")
        return result

    aggregated = group_and_aggregate(
        df,
        name,
        "count" if metric_name is None else {metric_name: aggregator.name},
        ascending=ascending,
    )
    result.total_candidates = int(len(aggregated))

    value_column = aggregated.columns[-1]
    frame = aggregated.rename(columns={value_column: "value"}).head(limit).copy()

    if "row_count" not in frame.columns:
        counts = (
            df.groupby(name, dropna=True, observed=True).size().rename("row_count")
        )
        frame = frame.merge(counts.reset_index(), on=name, how="left")

    frame.insert(0, "rank", range(1, len(frame) + 1))
    result.dataframe = frame.reset_index(drop=True)
    if result.total_candidates > limit:
        result.notes.append(
            f"Showing the top {limit} of {result.total_candidates:,} values."
        )
    return result


def _rank_rows(
    df: pd.DataFrame, column: str, *, limit: int, ascending: bool
) -> RankingResult:
    """Rank individual rows by a numeric column."""
    series = require_numeric_column(df, column)
    result = RankingResult(
        column=column, metric_column=None, aggregation=None, ascending=ascending
    )

    clean = pd.to_numeric(series, errors="coerce").dropna()
    result.total_candidates = int(len(clean))
    if clean.empty:
        result.dataframe = pd.DataFrame(columns=["rank", column, "value"])
        result.notes.append(f"Column '{column}' has no numeric values to rank.")
        return result

    ordered = clean.sort_values(ascending=ascending, kind="stable").head(limit)
    frame = pd.DataFrame(
        {
            "rank": range(1, len(ordered) + 1),
            "source_index": ordered.index,
            column: ordered.to_numpy(),
            "value": ordered.to_numpy(),
        }
    )
    result.dataframe = frame
    return result


# --------------------------------------------------------------------------- #
# 8. Outliers
# --------------------------------------------------------------------------- #

@dataclass
class OutlierResult:
    """Outliers found in a single numeric column."""

    column: str
    method: str
    threshold: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    count: int = 0
    pct: float = 0.0
    values_considered: int = 0
    indices: list[Any] = field(default_factory=list)
    #: The outlying rows (full rows, capped by ``max_rows``).
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "method": self.method,
            "threshold": self.threshold,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "count": self.count,
            "pct": self.pct,
            "values_considered": self.values_considered,
            "indices": [_jsonable(i) for i in self.indices],
            "notes": list(self.notes),
        }


def detect_outliers(
    df: pd.DataFrame,
    column: str,
    *,
    method: str = "iqr",
    threshold: float | None = None,
    max_rows: int = 100,
) -> OutlierResult:
    """Find outliers in `column` using the IQR fence or a z-score cut-off.

    `threshold` defaults to 1.5 for ``iqr`` and 3.0 for ``zscore``. A column
    with no spread returns zero outliers and a note rather than raising.
    """
    require_dataframe(df)
    name = require_column(df, column)

    if not isinstance(method, str) or method.strip().lower() not in OUTLIER_METHODS:
        raise UnsupportedOperationError("outlier method", method, OUTLIER_METHODS)
    method_name = method.strip().lower()

    if threshold is None:
        cutoff = 1.5 if method_name == "iqr" else 3.0
    elif isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise InvalidParameterError(f"'threshold' must be a number, got {threshold!r}.")
    elif threshold <= 0:
        raise InvalidParameterError(f"'threshold' must be positive, got {threshold}.")
    else:
        cutoff = float(threshold)

    row_cap = require_positive_int(max_rows, "max_rows", maximum=10_000)
    series = require_numeric_column(df, name)
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values.fillna(np.nan))].dropna()

    result = OutlierResult(
        column=name,
        method=method_name,
        threshold=cutoff,
        values_considered=int(len(values)),
    )

    if values.empty:
        result.notes.append(f"Column '{name}' has no finite numeric values.")
        result.rows = df.head(0).copy()
        return result

    if method_name == "iqr":
        q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            result.notes.append(
                "The interquartile range is zero, so the IQR fence cannot "
                "separate outliers; try method='zscore'."
            )
            result.rows = df.head(0).copy()
            return result
        lower, upper = q1 - cutoff * iqr, q3 + cutoff * iqr
    else:
        mean, std = float(values.mean()), float(values.std(ddof=0))
        if std <= 0:
            result.notes.append(
                "The column has no variation, so no z-score outliers exist."
            )
            result.rows = df.head(0).copy()
            return result
        lower, upper = mean - cutoff * std, mean + cutoff * std

    mask = (values < lower) | (values > upper)
    outlying = values[mask]

    result.lower_bound = lower
    result.upper_bound = upper
    result.count = int(len(outlying))
    result.pct = round(100.0 * result.count / len(values), 4)
    result.indices = list(outlying.index)

    ordered = (
        outlying.sub((lower + upper) / 2).abs().sort_values(ascending=False).index[:row_cap]
    )
    result.rows = df.loc[ordered].copy()

    if result.count > row_cap:
        result.notes.append(
            f"Showing the {row_cap} most extreme of {result.count:,} outliers."
        )
    return result


__all__ = [
    "CORRELATION_METHODS",
    "ColumnSchema",
    "CorrelationResult",
    "DEFAULT_STATISTICS",
    "DatasetSchema",
    "OUTLIER_METHODS",
    "OutlierResult",
    "RankingResult",
    "SegmentComparison",
    "StatisticsResult",
    "calculate_correlation",
    "calculate_statistics",
    "categorical_columns",
    "compare_segments",
    "datetime_columns",
    "detect_outliers",
    "get_column_summary",
    "get_dataset_schema",
    "group_and_aggregate",
    "numeric_columns",
    "rank_values",
    "supported_aggregations",
]
