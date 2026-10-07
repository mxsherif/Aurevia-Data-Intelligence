"""Tests for CSV / XLSX loading and its failure modes."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from app.config import Settings
from app.services.data_loader import (
    DataLoadError,
    list_excel_sheets,
    load_dataframe,
    normalize_columns,
)


class _FakeUpload(io.BytesIO):
    """Minimal stand-in for Streamlit's UploadedFile."""

    def __init__(self, payload: bytes, name: str) -> None:
        super().__init__(payload)
        self.name = name


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #

def test_load_csv_from_path(csv_file: Path, messy_frame: pd.DataFrame, settings: Settings):
    result = load_dataframe(csv_file, settings=settings)

    assert result.file_format == "csv"
    assert result.rows == len(messy_frame)
    assert result.source_name == "data.csv"
    assert result.encoding == "utf-8"
    assert "monthly_charge" in result.dataframe.columns
    # `empty_col` is entirely null and should have been dropped.
    assert "empty_col" not in result.dataframe.columns
    assert result.columns == messy_frame.shape[1] - 1


def test_load_csv_from_file_like(csv_file: Path, settings: Settings):
    upload = _FakeUpload(csv_file.read_bytes(), "upload.csv")
    result = load_dataframe(upload, settings=settings)

    assert result.rows == 12
    assert result.source_name == "upload.csv"


def test_load_csv_semicolon_delimiter(tmp_path: Path, settings: Settings):
    path = tmp_path / "semi.csv"
    path.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="utf-8")

    result = load_dataframe(path, settings=settings)

    assert list(result.dataframe.columns) == ["a", "b", "c"]
    assert result.rows == 2


def test_load_csv_latin1_encoding(tmp_path: Path, settings: Settings):
    path = tmp_path / "latin.csv"
    path.write_bytes("name,city\nJosé,Córdoba\nAndré,Málaga\n".encode("cp1252"))

    result = load_dataframe(path, settings=settings)

    assert result.rows == 2
    assert result.encoding in {"cp1252", "latin-1"}


def test_load_csv_row_cap_truncates(tmp_path: Path, settings: Settings):
    path = tmp_path / "big.csv"
    pd.DataFrame({"x": range(500)}).to_csv(path, index=False)

    result = load_dataframe(path, settings=settings, max_rows=100)

    assert result.truncated is True
    assert result.rows == 100
    assert result.original_rows == 101  # the cap + the sentinel row
    assert any("first 100" in note for note in result.notes)


def test_load_csv_skips_ragged_rows(tmp_path: Path, settings: Settings):
    path = tmp_path / "ragged.csv"
    path.write_text("a,b,c\n1,2,3\n4,5,6,7,8\n9,10,11\n", encoding="utf-8")

    result = load_dataframe(path, settings=settings)

    # The over-long row is dropped rather than crashing the load.
    assert result.columns == 3
    assert result.rows == 2


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #

def test_load_xlsx_from_path(xlsx_file: Path, settings: Settings):
    result = load_dataframe(xlsx_file, settings=settings)

    assert result.file_format == "excel"
    assert result.rows == 12
    assert "region" in result.dataframe.columns


def test_load_xlsx_named_sheet(tmp_path: Path, settings: Settings):
    path = tmp_path / "multi.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="first", index=False)
        pd.DataFrame({"b": [3, 4, 5]}).to_excel(writer, sheet_name="second", index=False)

    assert list_excel_sheets(path, settings=settings) == ["first", "second"]

    result = load_dataframe(path, sheet="second", settings=settings)
    assert list(result.dataframe.columns) == ["b"]
    assert result.rows == 3
    assert result.sheet_name == "second"


def test_load_xlsx_missing_sheet_raises(xlsx_file: Path, settings: Settings):
    with pytest.raises(DataLoadError, match="Could not read"):
        load_dataframe(xlsx_file, sheet="nope", settings=settings)


def test_invalid_xlsx_raises(tmp_path: Path, settings: Settings):
    path = tmp_path / "fake.xlsx"
    path.write_bytes(b"this is definitely not a zip archive")

    with pytest.raises(DataLoadError, match="not a valid Excel workbook"):
        load_dataframe(path, settings=settings)


def test_list_excel_sheets_on_csv_returns_empty(csv_file: Path, settings: Settings):
    assert list_excel_sheets(csv_file, settings=settings) == []


# --------------------------------------------------------------------------- #
# Invalid input
# --------------------------------------------------------------------------- #

def test_missing_file_raises(tmp_path: Path, settings: Settings):
    with pytest.raises(DataLoadError, match="File not found"):
        load_dataframe(tmp_path / "nope.csv", settings=settings)


def test_empty_file_raises(tmp_path: Path, settings: Settings):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(DataLoadError, match="is empty"):
        load_dataframe(path, settings=settings)


def test_whitespace_only_file_raises(tmp_path: Path, settings: Settings):
    path = tmp_path / "blank.csv"
    path.write_text("\n\n   \n", encoding="utf-8")

    with pytest.raises(DataLoadError, match="is empty"):
        load_dataframe(path, settings=settings)


def test_header_only_csv_raises(tmp_path: Path, settings: Settings):
    path = tmp_path / "header_only.csv"
    path.write_text("a,b,c\n", encoding="utf-8")

    with pytest.raises(DataLoadError, match="no data rows"):
        load_dataframe(path, settings=settings)


def test_unsupported_extension_raises(tmp_path: Path, settings: Settings):
    path = tmp_path / "notes.json"
    path.write_text('{"a": 1}', encoding="utf-8")

    with pytest.raises(DataLoadError, match="Unsupported file type"):
        load_dataframe(path, settings=settings)


def test_no_extension_raises(tmp_path: Path, settings: Settings):
    path = tmp_path / "datafile"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(DataLoadError, match="no file extension"):
        load_dataframe(path, settings=settings)


def test_directory_raises(tmp_path: Path, settings: Settings):
    directory = tmp_path / "folder.csv"
    directory.mkdir()

    with pytest.raises(DataLoadError):
        load_dataframe(directory, settings=settings)


def test_oversized_file_raises(tmp_path: Path):
    tiny = Settings(max_upload_mb=1)
    path = tmp_path / "big.csv"
    path.write_text("a,b\n" + "1,2\n" * 300_000, encoding="utf-8")

    with pytest.raises(DataLoadError, match="exceeds the 1 MB limit"):
        load_dataframe(path, settings=tiny)


# --------------------------------------------------------------------------- #
# Column hygiene
# --------------------------------------------------------------------------- #

def test_duplicate_column_names_are_deduplicated(tmp_path: Path, settings: Settings):
    path = tmp_path / "dupes.csv"
    path.write_text("name,name,name,value\na,b,c,1\nd,e,f,2\n", encoding="utf-8")

    result = load_dataframe(path, settings=settings)
    columns = list(result.dataframe.columns)

    assert len(columns) == len(set(columns)), "column names must be unique"
    assert columns == ["name", "name_2", "name_3", "value"]
    assert result.renamed_columns  # the rename was reported


def test_blank_and_unnamed_columns_get_positional_names():
    df = pd.DataFrame([[1, 2, 3]], columns=["  ", "Unnamed: 1", " spaced  name "])

    df, renamed = normalize_columns(df)

    assert list(df.columns) == ["column_1", "column_2", "spaced name"]
    assert renamed["  "] == "column_1"


def test_normalize_columns_handles_collision_with_generated_name():
    df = pd.DataFrame([[1, 2]], columns=["a", "a_2"])

    df, _ = normalize_columns(df)

    assert list(df.columns) == ["a", "a_2"]


def test_fully_empty_rows_are_dropped(tmp_path: Path, settings: Settings):
    path = tmp_path / "gaps.csv"
    path.write_text("a,b\n1,2\n,\n3,4\n", encoding="utf-8")

    result = load_dataframe(path, settings=settings)

    assert result.rows == 2
    assert any("empty row" in note for note in result.notes)


def test_bytes_input_is_supported(settings: Settings):
    with pytest.raises(DataLoadError, match="no file extension"):
        # Raw bytes have no filename, so the format cannot be determined.
        load_dataframe(b"a,b\n1,2\n", settings=settings)
