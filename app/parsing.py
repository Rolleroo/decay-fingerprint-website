"""Parse the pasted "nuclide, value" fingerprint into validated entries.

Kept free of unit/decay logic so conversions.py and engine.py can be tested
and reused independently (see spec Sec 7).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import radioactivedecay as rd


@dataclass(frozen=True)
class ParsedEntry:
    """One successfully parsed line."""

    line_no: int
    raw: str
    nuclide: str  # canonical name, e.g. "Cs-137", "Tc-99m"
    value: float


@dataclass(frozen=True)
class ParseError:
    """One line that failed to parse, with a human-readable reason."""

    line_no: int
    raw: str
    message: str


@dataclass(frozen=True)
class ParseResult:
    entries: list[ParsedEntry]
    errors: list[ParseError]

    @property
    def ok(self) -> bool:
        return not self.errors


def parse_paste(text: str) -> ParseResult:
    """Parse a multi-line "nuclide, value" paste.

    Each line is split on the first comma. The nuclide token is normalized
    and validated via ``radioactivedecay.Nuclide``, which natively accepts
    all three spellings (Cs-137 / Cs137 / 137Cs), tolerates casing and
    whitespace, and preserves metastable (m/n) tags. The value token must
    parse as a finite, non-negative float.

    Blank lines are skipped. Every other malformed line is collected as a
    ``ParseError`` rather than aborting the whole paste. A nuclide that
    appears on more than one line is flagged as a duplicate error on every
    line it occurs on, rather than being silently summed.
    """
    entries: list[ParsedEntry] = []
    errors: list[ParseError] = []
    seen_at: dict[str, list[int]] = {}

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        raw = raw_line.strip()
        if not raw:
            continue

        # Comma is the documented format, but input arrives in other shapes
        # too: a row copied out of a spreadsheet lands tab-separated, and
        # someone typing by hand often just uses a space. Accept all three
        # rather than forcing one exact format. Comma takes priority (a
        # value never contains one), then tab, then any run of whitespace.
        if "," in raw:
            parts = raw.split(",", 1)
        elif "\t" in raw:
            parts = raw.split("\t", 1)
        else:
            parts = re.split(r"\s+", raw, maxsplit=1)
        if len(parts) != 2:
            errors.append(
                ParseError(line_no, raw, "Expected format 'nuclide, value' (comma, tab, or space separated).")
            )
            continue

        nuclide_token, value_token = parts[0].strip(), parts[1].strip()
        if not nuclide_token:
            errors.append(ParseError(line_no, raw, "Missing nuclide name."))
            continue
        if not value_token:
            errors.append(ParseError(line_no, raw, "Missing value."))
            continue

        try:
            nuclide = rd.Nuclide(nuclide_token).nuclide
        except Exception as exc:  # library raises NuclideStrError/ValueError/KeyError
            errors.append(ParseError(line_no, raw, f"Unrecognized nuclide '{nuclide_token}': {exc}"))
            continue

        try:
            value = float(value_token)
        except ValueError:
            errors.append(ParseError(line_no, raw, f"Value '{value_token}' is not a number."))
            continue

        if value != value:  # NaN
            errors.append(ParseError(line_no, raw, "Value must not be NaN."))
            continue
        if value in (float("inf"), float("-inf")):
            errors.append(ParseError(line_no, raw, "Value must be finite."))
            continue
        if value < 0:
            errors.append(ParseError(line_no, raw, f"Value must not be negative (got {value})."))
            continue

        entries.append(ParsedEntry(line_no, raw, nuclide, value))
        seen_at.setdefault(nuclide, []).append(line_no)

    duplicate_lines: set[int] = set()
    for nuclide, lines in seen_at.items():
        if len(lines) > 1:
            for line_no in lines:
                duplicate_lines.add(line_no)
                errors.append(
                    ParseError(
                        line_no,
                        next(e.raw for e in entries if e.line_no == line_no),
                        f"Duplicate nuclide '{nuclide}' also appears on line(s) "
                        f"{[l for l in lines if l != line_no]}.",
                    )
                )

    if duplicate_lines:
        entries = [e for e in entries if e.line_no not in duplicate_lines]

    errors.sort(key=lambda e: e.line_no)
    return ParseResult(entries=entries, errors=errors)
