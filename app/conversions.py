"""Unit dropdown definitions and conversion to the canonical base.

Single principle (spec Sec 4): never convert unit->unit directly. Every
input unit is converted *in* to one canonical base that the
``radioactivedecay`` library accepts natively (Bq, g, mol, or "num" for
atom count), decay happens in that base, and display conversion happens
*out* of that same base. The library owns all per-nuclide scaling
(activity<->mass<->atoms via each nuclide's own decay constant); this
module never hand-rolls that math, only routes to it.

Pure-scaling units (Bq family, Ci family, mass family, mol) are passed
straight through to ``radioactivedecay.Inventory``, which accepts them
natively. The two unit kinds the library does *not* know about --
specific activity (bulk-sample activity concentration) and the three
relative/fraction modes -- are normalized here before being handed off.
"""

from __future__ import annotations

from dataclasses import dataclass

import radioactivedecay as rd
from radioactivedecay.converters import UnitConverterFloat

from app.parsing import ParsedEntry

_unit_converter = UnitConverterFloat()

# Relative tolerance for the fraction-mode sum check (spec Sec 6.5: "sum
# check fires in %/fraction modes, not in absolute modes"). Chosen to
# tolerate ordinary rounding in hand-entered data while still catching the
# 100x-class errors the spec calls out -- e.g. a 90% inventory accidentally
# entered as fractions summing to 9.0.
FRACTION_SUM_TOLERANCE = 0.01


class ValidationError(Exception):
    """Raised for input that must block submission rather than be guessed at.

    Per spec Sec 2's guiding principle for the fraction/% toggle ("must
    never silently decide"), out-of-tolerance fraction sums are blocked
    here rather than auto-renormalized, so a data-entry mistake surfaces
    instead of being silently absorbed.
    """


# display label -> (library unit string, kind)
# kind drives downstream handling in engine.py; "fraction_*" kinds are
# intercepted by _canonicalize_fraction before any library unit applies.
UNIT_MAP: dict[str, tuple[str | None, str]] = {
    "Bq": ("Bq", "activity"),
    "kBq": ("kBq", "activity"),
    "MBq": ("MBq", "activity"),
    "GBq": ("GBq", "activity"),
    "TBq": ("TBq", "activity"),
    "dpm": ("dpm", "activity"),  # disintegrations per minute, common on wipe/swab reports
    "Ci": ("Ci", "activity"),
    "mCi": ("mCi", "activity"),
    "µCi": ("uCi", "activity"),
    "nCi": ("nCi", "activity"),
    "pCi": ("pCi", "activity"),
    "Bq/g": ("Bq", "specific_activity"),
    "Bq/kg": ("Bq", "specific_activity"),
    "g": ("g", "mass"),
    "kg": ("kg", "mass"),
    "mg": ("mg", "mass"),
    "µg": ("ug", "mass"),
    "t": ("t", "mass"),  # tonne
    "mol": ("mol", "amount"),
    "atoms": ("num", "amount"),
    "activity fraction": (None, "fraction_activity"),
    "mass fraction": (None, "fraction_mass"),
    "mole fraction": (None, "fraction_mole"),
}

# Grouped purely for populating the UI dropdown in the order spec Sec 2 lists.
UNIT_GROUPS: list[tuple[str, list[str]]] = [
    ("Activity", ["Bq", "kBq", "MBq", "GBq", "TBq", "dpm", "Ci", "mCi", "µCi", "nCi", "pCi"]),
    ("Specific activity", ["Bq/g", "Bq/kg"]),
    ("Mass", ["g", "kg", "mg", "µg", "t"]),
    ("Amount", ["mol", "atoms"]),
    ("Relative", ["activity fraction", "mass fraction", "mole fraction"]),
]

_FRACTION_BASE_UNIT = {
    "fraction_activity": "Bq",
    "fraction_mass": "g",
    "fraction_mole": "mol",
}


@dataclass(frozen=True)
class CanonResult:
    """Canonicalized inventory ready to hand to ``radioactivedecay.Inventory``."""

    contents: dict[str, float]
    library_unit: str  # one of 'Bq','kBq','MBq','GBq','Ci','mCi','uCi','g','kg','mg','mol','num'
    kind: str  # 'activity' | 'specific_activity' | 'mass' | 'amount' | 'fraction_activity' | 'fraction_mass' | 'fraction_mole'
    display_unit: str  # the original dropdown label, for re-labeling output
    frac_as_percent: bool = False


def canonicalize(entries: list[ParsedEntry], unit_label: str, frac_as_percent: bool = False) -> CanonResult:
    """Convert parsed entries into a canonical (nuclide -> value) inventory.

    ``unit_label`` must be a key of ``UNIT_MAP``. ``frac_as_percent`` only
    matters for the three "Relative" kinds; it is the explicit toggle from
    spec Sec 2 (fraction vs percent is never auto-detected).
    """
    if unit_label not in UNIT_MAP:
        raise KeyError(f"Unknown unit '{unit_label}'.")

    library_unit, kind = UNIT_MAP[unit_label]

    if kind.startswith("fraction"):
        return _canonicalize_fraction(entries, unit_label, kind, frac_as_percent)

    contents = {e.nuclide: e.value for e in entries}
    return CanonResult(contents=contents, library_unit=library_unit, kind=kind, display_unit=unit_label)


def _canonicalize_fraction(
    entries: list[ParsedEntry], unit_label: str, kind: str, frac_as_percent: bool
) -> CanonResult:
    target = 100.0 if frac_as_percent else 1.0
    total = sum(e.value for e in entries)

    if total == 0:
        raise ValidationError("Values sum to 0; there is nothing to decay.")

    relative_deviation = abs(total - target) / target
    if relative_deviation > FRACTION_SUM_TOLERANCE:
        unit_word = "%" if frac_as_percent else "fraction"
        raise ValidationError(
            f"{unit_label} values sum to {total:g} ({unit_word} mode expects "
            f"{target:g} ± {FRACTION_SUM_TOLERANCE * 100:g}%). Fix the input rather than "
            f"relying on auto-renormalization -- a wrong sum here is a silent 100x-class error."
        )

    library_unit = _FRACTION_BASE_UNIT[kind]
    # Anchor to an arbitrary total of 1.0 in the matching base unit. Valid
    # because the Bateman equations are linear: scaling every initial
    # quantity by the same constant scales the whole trajectory by that
    # constant, so the *fractional* composition over time is independent
    # of which absolute anchor is chosen (spec Sec 6.3 fraction-renorm check
    # relies on this).
    contents = {e.nuclide: (e.value / total) for e in entries}
    return CanonResult(
        contents=contents,
        library_unit=library_unit,
        kind=kind,
        display_unit=unit_label,
        frac_as_percent=frac_as_percent,
    )


def scale_to_display(value: float, base_unit: str, display_unit: str) -> float:
    """Pure-scaling conversion from one of the engine's fixed base units
    ('Bq', 'g', 'mol') to the unit the user actually picked, for display.

    ``engine.run_time_series`` always stores results in fixed bases so it
    doesn't need to know about display units. This is the one place that
    converts back out -- always a constant factor (spec Sec 4 "pure
    scaling"), since the per-nuclide physics already happened inside the
    engine. Atom counts ('num') and fraction/percent values pass straight
    through unscaled.

    ``display_unit`` is the user-facing dropdown label, which is not always
    the library's own unit string: the micro-sign labels 'µCi' and 'µg'
    (U+00B5) must map to the library's 'uCi'/'ug', or the converter rejects
    them. Resolve the label to its library unit via ``UNIT_MAP`` first;
    labels already equal to a library unit (or library units passed
    directly, e.g. in tests) fall through unchanged.
    """
    lib_display = UNIT_MAP.get(display_unit, (None, None))[0] or display_unit
    if base_unit == lib_display:
        return value
    if base_unit in _unit_converter.activity_units:
        return _unit_converter.activity_unit_conv(value, base_unit, lib_display)
    if base_unit in _unit_converter.mass_units:
        return _unit_converter.mass_unit_conv(value, base_unit, lib_display)
    if base_unit in _unit_converter.moles_units:
        return _unit_converter.moles_unit_conv(value, base_unit, lib_display)
    return value
