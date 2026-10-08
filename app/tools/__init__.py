"""Deterministic analysis tools.

Every function here is pure with respect to its input dataframe, validates its
own arguments, and fails with a :class:`~app.tools.exceptions.ToolError`
subclass carrying a message fit for a user -- or, in a later phase, for an LLM
agent to read and correct itself from.

This package deliberately has no Streamlit, Plotly, or LLM dependency: it is the
analytical substrate the UI pages and the future agents both sit on top of.
"""

from app.tools.aggregations import (
    AGGREGATIONS,
    Aggregation,
    add_percentage_change,
    register_aggregation,
    resolve_aggregation,
    supported_aggregations,
)
from app.tools.analysis import (
    ColumnSchema,
    CorrelationResult,
    DatasetSchema,
    OutlierResult,
    RankingResult,
    SegmentComparison,
    StatisticsResult,
    calculate_correlation,
    calculate_statistics,
    compare_segments,
    detect_outliers,
    get_column_summary,
    get_dataset_schema,
    group_and_aggregate,
    rank_values,
)
from app.tools.charts import (
    CHART_SPECS,
    ChartData,
    ChartSpec,
    generate_chart_data,
    humanize,
    resolve_chart_type,
    suggest_chart_options,
    supported_chart_types,
)
from app.tools.exceptions import (
    ColumnNotFoundError,
    InvalidColumnTypeError,
    InvalidDataError,
    InvalidParameterError,
    ToolError,
    UnsupportedOperationError,
)
from app.tools.filters import (
    FilterCondition,
    build_mask,
    describe_conditions,
    filter_rows,
    normalize_operator,
    supported_operators,
)
from app.tools.timeseries import (
    FREQUENCIES,
    Frequency,
    TimeTrendResult,
    calculate_time_trend,
    resolve_frequency,
    supported_frequencies,
)
from app.tools.validation import (
    categorical_columns,
    datetime_columns,
    numeric_columns,
)

__all__ = [
    # The eleven tools
    "get_dataset_schema",
    "get_column_summary",
    "filter_rows",
    "group_and_aggregate",
    "calculate_statistics",
    "calculate_correlation",
    "compare_segments",
    "calculate_time_trend",
    "rank_values",
    "detect_outliers",
    "generate_chart_data",
    # Registries and extension points
    "AGGREGATIONS",
    "Aggregation",
    "CHART_SPECS",
    "ChartSpec",
    "FREQUENCIES",
    "Frequency",
    "add_percentage_change",
    "register_aggregation",
    "resolve_aggregation",
    "resolve_chart_type",
    "resolve_frequency",
    "supported_aggregations",
    "supported_chart_types",
    "supported_frequencies",
    "supported_operators",
    "normalize_operator",
    # Result types
    "ChartData",
    "ColumnSchema",
    "CorrelationResult",
    "DatasetSchema",
    "FilterCondition",
    "OutlierResult",
    "RankingResult",
    "SegmentComparison",
    "StatisticsResult",
    "TimeTrendResult",
    # Helpers
    "build_mask",
    "categorical_columns",
    "datetime_columns",
    "describe_conditions",
    "humanize",
    "numeric_columns",
    "suggest_chart_options",
    # Exceptions
    "ColumnNotFoundError",
    "InvalidColumnTypeError",
    "InvalidDataError",
    "InvalidParameterError",
    "ToolError",
    "UnsupportedOperationError",
]
