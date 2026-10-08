"""Robust CSV / XLSX loading.

Every failure mode we know about (empty file, malformed delimiters, a `.xlsx`
that is not really a zip archive, an unsupported extension, duplicate column
names, a file bigger than the configured limit) is converted into a
:class:`DataLoadError` carrying a message that is safe to show a user.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import SUPPORTED_EXTENSIONS, Settings, get_settings

logger = logging.getLogger(__name__)

CSV_EXTENSIONS = {".csv", ".txt", ".tsv"}
EXCEL_EXTENSIONS = {".xlsx", ".xlsm", ".xls"}

# Encodings tried in order when reading a CSV.
_CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

_WHITESPACE = re.compile(r"\s+")

# pandas de-duplicates repeated headers itself, appending ".1", ".2", ...
_PANDAS_DUPE_SUFFIX = re.compile(r"^(?P<base>.+)\.(?P<index>\d+)$")

# ValueError text that means "this is not a workbook" rather than "no such sheet".
_UNREADABLE_WORKBOOK = re.compile(
    r"format cannot be determined|not supported|corrupt|zip file|excel file format",
    re.I,
)


class DataLoadError(Exception):
    """Raised when a file cannot be turned into a usable dataframe."""


@dataclass
class LoadResult:
    """A loaded dataframe plus everything we noticed while loading it."""

    dataframe: pd.DataFrame
    source_name: str
    file_format: str
    rows: int
    columns: int
    size_bytes: int = 0
    sheet_name: str | None = None
    encoding: str | None = None
    delimiter: str | None = None
    truncated: bool = False
    original_rows: int | None = None
    renamed_columns: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def df(self) -> pd.DataFrame:  # frequently used alias
        return self.dataframe


# --------------------------------------------------------------------------- #
# Column hygiene
# --------------------------------------------------------------------------- #

def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Give every column a unique, non-empty, trimmed string name.

    Returns the dataframe (modified in place) and a mapping of
    ``original -> new`` for the names that actually changed.
    """
    renamed: dict[str, str] = {}
    used: set[str] = set()
    new_names: list[str] = []
    originals = [str(c) for c in df.columns]
    original_set = set(originals)

    for position, original in enumerate(originals):
        name = _WHITESPACE.sub(" ", original).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"column_{position + 1}"
        else:
            name = _restyle_pandas_dupe(name, original_set)

        if name in used:
            base, suffix = name, 2
            while f"{base}_{suffix}" in used:
                suffix += 1
            name = f"{base}_{suffix}"

        used.add(name)
        if name != original:
            renamed[original] = name
        new_names.append(name)

    df.columns = new_names
    return df, renamed


def _restyle_pandas_dupe(name: str, original_names: set[str]) -> str:
    """Turn pandas' ``col.1`` duplicate suffix into Aurevia's ``col_2``.

    Only applied when the un-suffixed base name is also present, so a column
    genuinely called ``revision.1`` is left alone.
    """
    match = _PANDAS_DUPE_SUFFIX.match(name)
    if not match:
        return name
    base = match.group("base")
    if base not in original_names:
        return name
    return f"{base}_{int(match.group('index')) + 1}"


def _drop_fully_empty(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop all-empty rows and columns, reporting what went."""
    notes: list[str] = []

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols and len(empty_cols) < len(df.columns):
        df = df.drop(columns=empty_cols)
        listed = ", ".join(map(str, empty_cols[:5]))
        tail = " ..." if len(empty_cols) > 5 else ""
        notes.append(f"Dropped {len(empty_cols)} fully empty column(s): {listed}{tail}")

    before = len(df)
    df = df.dropna(how="all")
    if len(df) < before:
        notes.append(f"Dropped {before - len(df)} fully empty row(s).")

    return df.reset_index(drop=True), notes


# --------------------------------------------------------------------------- #
# Input plumbing
# --------------------------------------------------------------------------- #

def _read_bytes(
    source: Any,
    *,
    settings: Settings,
    name_override: str | None = None,
) -> tuple[bytes, str, str]:
    """Normalise any supported input into ``(payload, name, extension)``.

    `name_override` supplies the filename when the source itself carries none
    (raw bytes), which is how the UI preserves an upload's original name.
    """
    limit = settings.max_upload_bytes

    # Streamlit UploadedFile and other file-like objects.
    if hasattr(source, "read") and not isinstance(source, (str, bytes, Path)):
        name = getattr(source, "name", "uploaded_file")
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            payload = source.read()
        except Exception as exc:  # noqa: BLE001 - stream errors vary wildly
            raise DataLoadError(f"Could not read the uploaded file: {exc}") from exc
        if isinstance(payload, str):
            payload = payload.encode("utf-8", errors="replace")
    elif isinstance(source, bytes):
        name = "in_memory_file"
        payload = source
    else:
        path = Path(source)
        name = path.name
        if not path.exists():
            raise DataLoadError(f"File not found: {path}")
        if not path.is_file():
            raise DataLoadError(f"Not a file: {path}")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise DataLoadError(f"Could not inspect {path.name}: {exc}") from exc
        if size > limit:
            raise DataLoadError(
                f"'{name}' is {size / 1024 / 1024:.1f} MB, which exceeds the "
                f"{settings.max_upload_mb} MB limit."
            )
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise DataLoadError(f"Could not read {path.name}: {exc}") from exc

    if len(payload) > limit:
        raise DataLoadError(
            f"'{name}' is {len(payload) / 1024 / 1024:.1f} MB, which exceeds the "
            f"{settings.max_upload_mb} MB limit."
        )
    if not payload.strip():
        raise DataLoadError(f"'{name}' is empty.")

    if name_override:
        name = name_override

    extension = Path(name).suffix.lower()
    return payload, name, extension


def _validate_extension(name: str, extension: str) -> str:
    if not extension:
        raise DataLoadError(
            f"'{name}' has no file extension. Supported formats: "
            + ", ".join(SUPPORTED_EXTENSIONS)
        )
    if extension in CSV_EXTENSIONS:
        return "csv"
    if extension in EXCEL_EXTENSIONS:
        return "excel"
    raise DataLoadError(
        f"Unsupported file type '{extension}'. Supported formats: "
        + ", ".join(SUPPORTED_EXTENSIONS)
    )


# --------------------------------------------------------------------------- #
# Format readers
# --------------------------------------------------------------------------- #

def _read_csv(
    payload: bytes,
    name: str,
    *,
    max_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a CSV, sniffing both the encoding and the delimiter."""
    last_error: Exception | None = None

    for encoding in _CSV_ENCODINGS:
        try:
            df = pd.read_csv(
                io.BytesIO(payload),
                encoding=encoding,
                sep=None,                 # let the python engine sniff it
                engine="python",
                skip_blank_lines=True,
                on_bad_lines="skip",      # tolerate ragged rows
                nrows=max_rows + 1,       # +1 so we can detect truncation
            )
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.EmptyDataError as exc:
            raise DataLoadError(f"'{name}' contains no parsable rows or columns.") from exc
        except (pd.errors.ParserError, ValueError) as exc:
            last_error = exc
            # Delimiter sniffing failed; retry with a plain comma.
            try:
                df = pd.read_csv(
                    io.BytesIO(payload),
                    encoding=encoding,
                    sep=",",
                    engine="python",
                    skip_blank_lines=True,
                    on_bad_lines="skip",
                    nrows=max_rows + 1,
                )
            except pd.errors.EmptyDataError as inner:
                raise DataLoadError(
                    f"'{name}' contains no parsable rows or columns."
                ) from inner
            except Exception as inner:  # noqa: BLE001
                last_error = inner
                continue
            return df, {"encoding": encoding, "delimiter": ","}
        else:
            return df, {"encoding": encoding, "delimiter": "auto"}

    raise DataLoadError(
        f"Could not parse '{name}' as CSV. The file may be malformed or use an "
        f"unsupported encoding. ({last_error})"
    )


def _read_excel(
    payload: bytes,
    name: str,
    *,
    sheet: str | int | None,
    max_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = 0 if sheet is None else sheet
    try:
        frame = pd.read_excel(
            io.BytesIO(payload),
            sheet_name=target,
            nrows=max_rows + 1,
        )
    except ValueError as exc:
        # pandas raises ValueError both for a missing sheet name and for a file
        # whose format it cannot determine; only the latter means "corrupt".
        if _UNREADABLE_WORKBOOK.search(str(exc)):
            raise DataLoadError(
                f"'{name}' is not a valid Excel workbook, or it is corrupted. ({exc})"
            ) from exc
        raise DataLoadError(f"Could not read '{name}': {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - openpyxl/xlrd raise many types
        raise DataLoadError(
            f"'{name}' is not a valid Excel workbook, or it is corrupted. ({exc})"
        ) from exc

    if isinstance(frame, dict):  # defensive: sheet_name=None returns a dict
        if not frame:
            raise DataLoadError(f"'{name}' contains no sheets.")
        sheet_label, frame = next(iter(frame.items()))
    else:
        sheet_label = target if isinstance(target, str) else None

    return frame, {"sheet_name": sheet_label}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def load_dataframe(
    source: Any,
    *,
    name: str | None = None,
    sheet: str | int | None = None,
    settings: Settings | None = None,
    max_rows: int | None = None,
) -> LoadResult:
    """Load `source` (path, bytes, or file-like) into a :class:`LoadResult`.

    Pass `name` when `source` is raw bytes: the extension drives format
    detection, so bytes without a filename cannot be loaded.

    Raises :class:`DataLoadError` -- and nothing else -- on failure.
    """
    settings = settings or get_settings()
    row_cap = max_rows if max_rows is not None else settings.max_rows

    payload, name, extension = _read_bytes(source, settings=settings, name_override=name)
    kind = _validate_extension(name, extension)

    if kind == "csv":
        df, meta = _read_csv(payload, name, max_rows=row_cap)
    else:
        df, meta = _read_excel(payload, name, sheet=sheet, max_rows=row_cap)

    notes: list[str] = []

    truncated = len(df) > row_cap
    original_rows: int | None = None
    if truncated:
        original_rows = len(df)
        df = df.head(row_cap)
        notes.append(f"Only the first {row_cap:,} rows were loaded; the file is larger.")

    df, renamed = normalize_columns(df)
    if renamed:
        notes.append(
            f"Renamed {len(renamed)} duplicate or blank column name(s) to keep them unique."
        )

    df, empty_notes = _drop_fully_empty(df)
    notes.extend(empty_notes)

    if df.shape[1] == 0:
        raise DataLoadError(f"'{name}' has no usable columns.")
    if df.shape[0] == 0:
        raise DataLoadError(f"'{name}' parsed successfully but contains no data rows.")

    logger.info("Loaded %s: %d rows x %d columns", name, df.shape[0], df.shape[1])

    return LoadResult(
        dataframe=df,
        source_name=name,
        file_format=kind,
        rows=df.shape[0],
        columns=df.shape[1],
        size_bytes=len(payload),
        sheet_name=meta.get("sheet_name"),
        encoding=meta.get("encoding"),
        delimiter=meta.get("delimiter"),
        truncated=truncated,
        original_rows=original_rows,
        renamed_columns=renamed,
        notes=notes,
    )


def list_excel_sheets(
    source: Any,
    *,
    name: str | None = None,
    settings: Settings | None = None,
) -> list[str]:
    """Sheet names in an Excel workbook (empty list for non-Excel input)."""
    settings = settings or get_settings()
    payload, name, extension = _read_bytes(source, settings=settings, name_override=name)
    if _validate_extension(name, extension) != "excel":
        return []
    try:
        with pd.ExcelFile(io.BytesIO(payload)) as workbook:
            return list(workbook.sheet_names)
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Could not inspect '{name}': {exc}") from exc
