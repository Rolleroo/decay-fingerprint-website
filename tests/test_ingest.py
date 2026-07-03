"""File-ingest robustness (2026-07-03). Real gamma-spec exports are messy:
non-UTF-8 encodings, ';' delimiters with European decimal commas, metadata
preamble rows before the header, and varied column names. These confirm the
reader copes and that everything still funnels through the one shared parser.
"""

import io

import pandas as pd
import pytest

from app.ingest import (
    ColumnMapping,
    IngestError,
    decode_bytes,
    detect_columns,
    find_header_row,
    paste_text_from_dataframe,
    paste_text_from_upload,
    read_upload,
    sniff_delimiter,
)
from app.parsing import parse_paste


# --- column detection ---


def test_detect_columns_by_keyword():
    df = pd.DataFrame({"Radionuclide": ["Cs-137"], "Activity (Bq)": [10], "Unc (Bq)": [1]})
    m = detect_columns(df)
    assert (m.nuclide, m.value, m.uncertainty) == ("Radionuclide", "Activity (Bq)", "Unc (Bq)")
    assert not m.uncertainty_is_percent


def test_detect_columns_positional_fallback():
    df = pd.DataFrame({"A": ["Cs-137"], "B": [10], "C": [1]})
    m = detect_columns(df)
    assert (m.nuclide, m.value, m.uncertainty) == ("A", "B", "C")


def test_detect_columns_percent_uncertainty_header():
    df = pd.DataFrame({"Nuclide": ["Cs-137"], "Bq": [10], "Unc %": [5]})
    m = detect_columns(df)
    assert m.uncertainty_is_percent


def test_detect_columns_needs_two_columns():
    with pytest.raises(IngestError):
        detect_columns(pd.DataFrame({"Nuclide": ["Cs-137"]}))


# --- paste-text rendering ---


def test_paste_text_skips_blank_and_trailer_rows():
    df = pd.DataFrame(
        {"Nuclide": ["Cs-137", "", "Total"], "Bq": ["10", "", ""]}
    )
    text = paste_text_from_dataframe(df)
    # "Total" row has a blank value -> skipped; blank row -> skipped.
    assert text == "Cs-137, 10"


def test_paste_text_appends_percent_when_header_is_percent():
    df = pd.DataFrame({"Nuclide": ["Cs-137"], "Bq": ["10"], "Unc %": ["5"]})
    text = paste_text_from_dataframe(df)
    assert text == "Cs-137, 10, 5%"


# --- decoding / delimiter / header detection ---


def test_decode_handles_cp1252_micro_and_plusminus():
    raw = "Nuclide,µCi,±\nCs-137,1.0,0.1".encode("cp1252")
    text = decode_bytes(raw)
    assert "µCi" in text and "Cs-137" in text


def test_sniff_delimiter_prefers_semicolon_when_more_consistent():
    text = "Nuclide;Activity;Unc\nCs-137;10;1\nCo-60;20;2"
    assert sniff_delimiter(text) == ";"


def test_sniff_delimiter_defaults_to_comma():
    assert sniff_delimiter("Nuclide,Bq\nCs-137,10") == ","


def test_find_header_row_skips_preamble():
    rows = [
        ["Sample ID: ABC123", "", ""],
        ["Acquired: 2026-07-01", "", ""],
        ["", "", ""],
        ["Nuclide", "Activity (Bq)", "Uncertainty"],
        ["Cs-137", "10", "1"],
    ]
    assert find_header_row(rows) == 3


# --- full read_upload on realistic files ---


def test_read_csv_semicolon_european_decimal():
    raw = "Nuclide;Activity (Bq);Unc (Bq)\nCs-137;3,7;0,4\nCo-60;1,2;0,1".encode("utf-8")
    df = read_upload("sample.csv", raw)
    text = paste_text_from_dataframe(df)
    result = parse_paste(text)
    assert result.ok, result.errors
    values = {e.nuclide: e.value for e in result.entries}
    assert values["Cs-137"] == pytest.approx(3.7)
    assert values["Co-60"] == pytest.approx(1.2)


def test_read_csv_with_instrument_preamble():
    csv_text = (
        "Gamma Spectrometry Report\n"
        "Sample ID: WASTE-001\n"
        "Acquired: 2026-06-30 14:22\n"
        "\n"
        "Nuclide,Activity (Bq),Uncertainty (Bq)\n"
        "Cs-137,3.7e3,120\n"
        "Co-60,8.8e2,40\n"
    )
    df = read_upload("report.csv", csv_text.encode("utf-8"))
    text, mapping = paste_text_from_upload("report.csv", csv_text.encode("utf-8"))
    assert mapping.nuclide == "Nuclide"
    result = parse_paste(text)
    assert result.ok, result.errors
    assert {e.nuclide for e in result.entries} == {"Cs-137", "Co-60"}
    assert result.entries[0].uncertainty is not None  # absolute unc carried through


def test_read_csv_cp1252_encoded_file_does_not_crash():
    raw = "Nuclide,Activity,Unc %\nCs-137,10,5\nCo-60,20,3".encode("cp1252")
    df = read_upload("legacy.csv", raw)
    text = paste_text_from_dataframe(df)
    result = parse_paste(text)
    assert result.ok, result.errors
    assert result.entries[0].uncertainty == pytest.approx(0.05)


def test_read_xlsx_round_trip_with_preamble():
    rows = [
        ["Lab report", "", ""],
        ["", "", ""],
        ["Nuclide", "Activity (Bq)", "Unc (Bq)"],
        ["Cs-137", "3.7e3", "120"],
        ["Sr-90", "5.0e2", "25"],
    ]
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False, header=False)
    df = read_upload("report.xlsx", buf.getvalue())
    text, mapping = paste_text_from_upload("report.xlsx", buf.getvalue())
    assert mapping.nuclide == "Nuclide"
    result = parse_paste(text)
    assert result.ok, result.errors
    assert {e.nuclide for e in result.entries} == {"Cs-137", "Sr-90"}


def test_unsupported_extension_rejected():
    with pytest.raises(IngestError):
        read_upload("data.pdf", b"whatever")


def test_oversized_file_rejected():
    from app.ingest import MAX_UPLOAD_BYTES

    big = b"Nuclide,Bq\n" + b"Cs-137,1\n" * (MAX_UPLOAD_BYTES // 8)
    assert len(big) > MAX_UPLOAD_BYTES
    with pytest.raises(IngestError, match="too large"):
        read_upload("big.csv", big)


def test_too_many_columns_rejected():
    from app.ingest import MAX_COLS

    header = ",".join(f"c{i}" for i in range(MAX_COLS + 5))
    row = ",".join("1" for _ in range(MAX_COLS + 5))
    with pytest.raises(IngestError, match="too big"):
        read_upload("wide.csv", f"{header}\n{row}".encode("utf-8"))


def test_full_pipeline_upload_to_parser_clean():
    csv_text = "Nuclide,Activity (Bq),Unc %\nCs-137,3.7e9,5\nSr-90,5.0e7,3\n"
    text, mapping = paste_text_from_upload("f.csv", csv_text.encode("utf-8"))
    result = parse_paste(text)
    assert result.ok, result.errors
    assert result.entries[0].uncertainty_is_relative
    assert result.entries[0].uncertainty == pytest.approx(0.05)


def test_mdaless_than_value_survives_ingest_as_text():
    # Detection-limit values ("<0.5") are preserved verbatim by ingest; the
    # parser currently rejects them (censored-value handling is the deferred
    # medium feature). Confirm the value reaches the parser intact rather
    # than being silently dropped or mangled during ingest.
    df = pd.DataFrame({"Nuclide": ["Cs-137", "Co-60"], "Bq": ["<0.5", "10"]})
    text = paste_text_from_dataframe(df)
    assert "<0.5" in text
    result = parse_paste(text)
    # Co-60 parses; Cs-137's "<0.5" is flagged (not a number) -- documented.
    assert any(e.nuclide == "Co-60" for e in result.entries)
    assert any("Cs-137" in err.raw for err in result.errors)
