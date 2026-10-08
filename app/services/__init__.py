"""Stateless services: loading, profiling, and chart rendering.

`app.services.visualization` is deliberately **not** re-exported here. The
layering runs ``services.profiler -> tools -> services.visualization``: the
tools import the profiler, and the renderer imports the tools. Pulling the
renderer into this package's ``__init__`` would make importing any service
re-enter the package mid-initialisation. Import it by module instead::

    from app.services.visualization import build_chart
"""

from app.services.data_loader import (
    DataLoadError,
    LoadResult,
    list_excel_sheets,
    load_dataframe,
)
from app.services.profiler import (
    descriptive_statistics,
    infer_field_type,
    profile_column,
    profile_dataframe,
)

__all__ = [
    "DataLoadError",
    "LoadResult",
    "descriptive_statistics",
    "infer_field_type",
    "list_excel_sheets",
    "load_dataframe",
    "profile_column",
    "profile_dataframe",
]
