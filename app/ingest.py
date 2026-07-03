"""Ingest a spreadsheet/CSV export (e.g. a gamma-spec results table) into
the same ``nuclide, value[, uncertainty]`` text the paste box accepts, so
uploaded files flow through the one existing parser (``parse_paste``) and
inherit all of its validation and error reporting -- one source of truth.

Deliberately a thin, transparent auto-detector, not a full column-mapping
UI (that is the deferred medium-effort feature): it picks nuclide / value /
uncertainty columns by header keyword, falling back to position, and
*returns which columns it chose* so the UI can show the user rather than
decide silently. Anything it gets wrong surfaces as an ordinary per-line
parse error downstream.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import pandas as pd

# Real lab exports (Genie, GammaVision, spreadsheets re-saved on Windows)
# are rarely clean UTF-8. Try in order of specificity; latin-1 as a final
# resort never raises, so decoding always succeeds (worst case: mojibake,
# which then surfaces as a normal per-line parse error, not a crash).
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

# Candidate CSV delimiters. Semicolon is common in European exports (where
# the comma is the decimal separator); tab from spreadsheet copy-paste.
_DELIMITERS = (",", ";", "\t", "|")

# Header keywords, most specific first. A column matches if its (lowercased)
# name contains any of the fragments.
_NUCLIDE_KEYS = ("nuclide", "radionuclide", "isotope", "nuclid", "nucleide")
_VALUE_KEYS = (
    "activity", "conc", "result", "value", "amount", "mass", "bq", "content",
)
_UNCERTAINTY_KEYS = (
    "uncert", "uncertainty", "error", "sigma", "std", "±", "+/-", "1s", "2s", "unc",
)


class IngestError(Exception):
    """Raised when a file cannot be turned into paste text at all (e.g. it
    has fewer than two usable columns). Per-row problems are NOT raised
    here -- they are left for the shared parser to report line by line."""


@dataclass(frozen=True)
class ColumnMapping:
    nuclide: str
    value: str
    uncertainty: str | None
    uncertainty_is_percent: bool  # header carried a '%' -> values are relative


def _match(columns: list[str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        for col in columns:
            if key in str(col).strip().lower():
                return col
    return None


def detect_columns(df: pd.DataFrame) -> ColumnMapping:
    """Choose nuclide / value / uncertainty columns by header keyword,
    falling back to column position (0 = nuclide, 1 = value, 2 = uncertainty).
    """
    columns = list(df.columns)
    if len(columns) < 2:
        raise IngestError(
            "The file needs at least two columns (a nuclide name and a value). "
            f"Found: {columns}."
        )

    nuclide = _match(columns, _NUCLIDE_KEYS) or columns[0]
    remaining = [c for c in columns if c != nuclide]
    value = _match(remaining, _VALUE_KEYS) or remaining[0]
    remaining = [c for c in remaining if c != value]
    uncertainty = _match(remaining, _UNCERTAINTY_KEYS) or (remaining[0] if remaining else None)

    is_percent = uncertainty is not None and "%" in str(uncertainty)
    return ColumnMapping(nuclide, value, uncertainty, is_percent)


def paste_text_from_dataframe(df: pd.DataFrame, mapping: ColumnMapping | None = None) -> str:
    """Render ``df`` into ``nuclide, value[, uncertainty]`` lines.

    Rows with a blank nuclide or blank value are skipped (spreadsheet
    exports routinely carry total/blank trailer rows). Everything else is
    passed through verbatim so the shared parser can validate it -- an
    unrecognised nuclide or non-numeric value becomes a normal per-line
    error there, not a silent drop here. A '<' detection-limit value (e.g.
    "<0.5") is preserved as-is; the parser's handling of it is a separate
    (deferred) concern -- for now it will surface as a parse error rather
    than be silently mangled.
    """
    if mapping is None:
        mapping = detect_columns(df)

    lines: list[str] = []
    for _, row in df.iterrows():
        nuclide = str(row[mapping.nuclide]).strip()
        value = str(row[mapping.value]).strip()
        if not nuclide or nuclide.lower() in ("nan", "none"):
            continue
        if not value or value.lower() in ("nan", "none"):
            continue

        parts = [nuclide, value]
        if mapping.uncertainty is not None:
            unc = str(row[mapping.uncertainty]).strip()
            if unc and unc.lower() not in ("nan", "none"):
                if mapping.uncertainty_is_percent and not unc.endswith("%"):
                    unc = f"{unc}%"
                parts.append(unc)
        lines.append(", ".join(parts))

    return "\n".join(lines)


def decode_bytes(data: bytes) -> str:
    """Decode file bytes, trying UTF-8 (with/without BOM) then the common
    Windows/Latin encodings. Never raises -- latin-1 maps every byte."""
    for enc in _ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def sniff_delimiter(text: str) -> str:
    """Pick the delimiter that splits the data rows into the most columns,
    consistently. Falls back to comma. Ignores blank lines."""
    lines = [ln for ln in text.splitlines() if ln.strip()][:50]
    if not lines:
        return ","
    best, best_score = ",", -1.0
    for delim in _DELIMITERS:
        counts = [ln.count(delim) for ln in lines]
        if max(counts) == 0:
            continue
        # Score over ALL lines (header included): a European ';'-delimited
        # file has decimal commas in the data rows, so counting only the
        # rows where ',' appears would hide that the header row has none --
        # ';' then wins on full-file consistency, as it should.
        modal = max(set(counts), key=counts.count)
        if modal == 0:
            continue
        agreement = sum(1 for c in counts if c == modal) / len(counts)
        score = modal * agreement
        if score > best_score:
            best, best_score = delim, score
    return best


def find_header_row(rows: list[list[str]]) -> int:
    """Index of the row that looks like the column header.

    Instrument exports prepend metadata lines ("Sample ID: ...", blank
    rows) before the real table. The header is the first row that carries
    a recognisable column keyword (nuclide / value / uncertainty). If none
    is found, assume the first non-empty row.
    """
    all_keys = _NUCLIDE_KEYS + _VALUE_KEYS + _UNCERTAINTY_KEYS
    for i, row in enumerate(rows):
        cells = [str(c).strip().lower() for c in row]
        if any(any(k in cell for k in all_keys) for cell in cells if cell):
            return i
    for i, row in enumerate(rows):  # fallback: first row with >=2 non-empty cells
        if sum(1 for c in row if str(c).strip()) >= 2:
            return i
    return 0


def _read_csv(data: bytes) -> pd.DataFrame:
    text = decode_bytes(data)
    delim = sniff_delimiter(text)
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    rows = [r for r in rows if any(str(c).strip() for c in r)]  # drop blank lines
    if not rows:
        raise IngestError("The file appears to be empty.")

    header_idx = find_header_row(rows)
    # A ';' delimiter almost always means the European convention where the
    # comma is the decimal separator. In that case let pandas convert the
    # decimals (dtype=str would defeat ``decimal=','`` -- it never touches
    # strings). Otherwise keep everything textual so exact forms like
    # "3.7e9" survive untouched and the shared parser owns numeric validation.
    if delim == ";":
        df = pd.read_csv(
            io.StringIO(text),
            delimiter=delim,
            skiprows=header_idx,
            decimal=",",
            engine="python",
            keep_default_na=False,
        )
    else:
        df = pd.read_csv(
            io.StringIO(text),
            delimiter=delim,
            skiprows=header_idx,
            engine="python",
            dtype=str,
            keep_default_na=False,
        )
    return df


def _read_excel(filename: str, data: bytes) -> pd.DataFrame:
    name = filename.lower()
    try:
        raw = pd.read_excel(io.BytesIO(data), header=None, dtype=str)
    except ImportError as exc:  # missing engine for this workbook type
        raise IngestError(
            f"Could not read {filename} ({exc}). Re-save it as .xlsx or .csv."
        ) from exc
    except Exception as exc:
        if name.endswith(".xls"):
            raise IngestError(
                "Legacy .xls workbooks are not supported. Re-save as .xlsx or .csv."
            ) from exc
        raise IngestError(f"Could not read {filename}: {exc}") from exc

    rows = raw.fillna("").astype(str).values.tolist()
    rows = [r for r in rows if any(str(c).strip() for c in r)]
    if not rows:
        raise IngestError("The spreadsheet appears to be empty.")
    header_idx = find_header_row(rows)
    header = [str(c).strip() for c in rows[header_idx]]
    body = rows[header_idx + 1 :]
    width = len(header)
    body = [r[:width] + [""] * (width - len(r)) for r in body]
    return pd.DataFrame(body, columns=header)


def read_upload(filename: str, data: bytes) -> pd.DataFrame:
    """Read an uploaded CSV/TSV or XLSX file into a DataFrame with the real
    header promoted and any metadata preamble stripped. UI-facing.

    Handles non-UTF-8 encodings, `,`/`;`/tab delimiters, European decimal
    commas, and instrument-export preamble rows. Everything is read as text;
    numeric validation stays with the shared parser (one source of truth).
    """
    name = filename.lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        df = _read_excel(filename, data)
    elif name.endswith((".csv", ".txt", ".tsv", ".dat")):
        df = _read_csv(data)
    else:
        raise IngestError(f"Unsupported file type: {filename}. Use CSV or XLSX.")

    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(how="all")


def paste_text_from_upload(filename: str, data: bytes) -> tuple[str, ColumnMapping]:
    """Convenience: file bytes -> (paste text, the column mapping used)."""
    df = read_upload(filename, data)
    mapping = detect_columns(df)
    return paste_text_from_dataframe(df, mapping), mapping
