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
    # Optional 1-sigma measurement uncertainty (reverse-mode input; the
    # forward tool ignores it). Relative uncertainties ("5%") are stored as
    # a fraction of the value (0.05) with uncertainty_is_relative=True;
    # absolute ones are in the same unit as the value.
    uncertainty: float | None = None
    uncertainty_is_relative: bool = False


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


# Marker forms for an attached uncertainty: "10 ± 5", "10 +- 5", "10 +/- 5".
_PLUS_MINUS_RE = re.compile(r"±|\+/-|\+-")

# A bare 3-digit group in the uncertainty position of a comma-separated line
# is far more likely to be a thousands separator ("Cs-137, 1,000") than a
# real uncertainty; parsing it as one would silently turn 1000 into 1 ± 0.
_THOUSANDS_GROUP_RE = re.compile(r"^\d{3}$")


def parse_paste(text: str) -> ParseResult:
    """Parse a multi-line "nuclide, value[, uncertainty]" paste.

    The nuclide token is normalized and validated via
    ``radioactivedecay.Nuclide``, which natively accepts all three
    spellings (Cs-137 / Cs137 / 137Cs), tolerates casing and whitespace,
    and preserves metastable (m/n) tags. The value token must parse as a
    finite, non-negative float.

    An optional third field is a 1-sigma measurement uncertainty (used by
    reverse mode, ignored by forward): either absolute in the same unit as
    the value ("Cs-137, 3.7e9, 1e8", "Cs-137, 3.7e9 ± 1e8") or relative
    with a % suffix ("Cs-137, 3.7e9, 5%").

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
        used_comma = "," in raw
        if used_comma:
            fields = [f.strip() for f in raw.split(",")]
        elif "\t" in raw:
            fields = [f.strip() for f in raw.split("\t")]
        else:
            fields = re.split(r"\s+", raw, maxsplit=1)
        if len(fields) < 2:
            errors.append(
                ParseError(line_no, raw, "Expected format 'nuclide, value' (comma, tab, or space separated).")
            )
            continue

        nuclide_token = fields[0].strip()
        # Everything after the nuclide is value + optional uncertainty, in
        # any of the accepted shapes ("10, 5%", "10 ± 5", "10 5%", ...).
        rest = _PLUS_MINUS_RE.sub(" ", " ".join(fields[1:]))
        tokens = rest.split()

        if not nuclide_token:
            errors.append(ParseError(line_no, raw, "Missing nuclide name."))
            continue
        if not tokens:
            errors.append(ParseError(line_no, raw, "Missing value."))
            continue
        if len(tokens) > 2:
            errors.append(
                ParseError(
                    line_no,
                    raw,
                    "Too many fields; expected 'nuclide, value' with at most one "
                    "uncertainty (e.g. 'Cs-137, 3.7e9, 5%'). Remove thousands "
                    "separators from numbers.",
                )
            )
            continue

        try:
            nuclide = rd.Nuclide(nuclide_token).nuclide
        except Exception as exc:  # library raises NuclideStrError/ValueError/KeyError
            errors.append(ParseError(line_no, raw, f"Unrecognized nuclide '{nuclide_token}': {exc}"))
            continue

        value_token = tokens[0]
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

        uncertainty: float | None = None
        uncertainty_is_relative = False
        if len(tokens) == 2:
            unc_token = tokens[1]
            if (
                used_comma
                and _THOUSANDS_GROUP_RE.match(unc_token)
                and "." not in value_token
                and "e" not in value_token.lower()
            ):
                errors.append(
                    ParseError(
                        line_no,
                        raw,
                        f"'{value_token},{unc_token}' looks like a number with a thousands "
                        f"separator, not a value plus uncertainty. Remove the comma from the "
                        f"number, or write the uncertainty explicitly as '± {unc_token}' or a "
                        f"percentage.",
                    )
                )
                continue
            uncertainty_is_relative = unc_token.endswith("%")
            unc_number_token = unc_token[:-1] if uncertainty_is_relative else unc_token
            try:
                uncertainty = float(unc_number_token)
            except ValueError:
                errors.append(
                    ParseError(line_no, raw, f"Uncertainty '{unc_token}' is not a number.")
                )
                continue
            if uncertainty != uncertainty or uncertainty in (float("inf"), float("-inf")):
                errors.append(ParseError(line_no, raw, "Uncertainty must be finite."))
                continue
            if uncertainty < 0:
                errors.append(
                    ParseError(line_no, raw, f"Uncertainty must not be negative (got {uncertainty}).")
                )
                continue
            if uncertainty_is_relative:
                uncertainty /= 100.0

        entries.append(
            ParsedEntry(line_no, raw, nuclide, value, uncertainty, uncertainty_is_relative)
        )
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
