"""Thin wrapper over ``radioactivedecay`` for forward decay over a time range.

Kept separate from the UI (spec Sec 7) so a future reverse/age-dating mode
can reuse it directly. All physics is delegated to the library; this module
only builds the time grid, drives repeated ``Inventory.decay()`` calls, and
reshapes the results into per-nuclide time series.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import radioactivedecay as rd

from app.conversions import CanonResult

# Default number of log-spaced points in the auto time grid (spec Sec 5:
# "auto log-range from the half-lives present"). 100 gives smooth log-log
# plots without making the decay loop slow.
DEFAULT_STEPS = 100

# How far below the shortest half-life / above the longest half-life the
# auto grid reaches, in half-life multiples. Wide enough to show full
# ingrowth-and-decay behavior for typical chains without the user having to
# touch the advanced override for the common case.
AUTO_RANGE_LOW_FACTOR = 0.01  # start at 1% of the shortest half-life
AUTO_RANGE_HIGH_FACTOR = 10.0  # end at 10x the longest half-life


@dataclass(frozen=True)
class TimeSeriesResult:
    times_s: list[float]
    nuclides: list[str]  # union of every nuclide present at any time step, sorted
    half_lives_s: dict[str, float]  # inf for stable nuclides
    activities_bq: dict[str, list[float]]
    masses_g: dict[str, list[float]]
    moles_mol: dict[str, list[float]]
    atoms: dict[str, list[float]]
    kind: str
    display_unit: str
    frac_as_percent: bool
    # Populated only when kind starts with "fraction_"; values sum to 1 (or
    # 100 if frac_as_percent) at every time step, per spec Sec 6.3.
    fractions: dict[str, list[float]] = field(default_factory=dict)


def nuclide_half_life_s(nuclide: str) -> float:
    """Half-life in seconds; ``float('inf')`` for stable nuclides."""
    return rd.Nuclide(nuclide).half_life("s")


def is_stable(nuclide: str) -> bool:
    return math.isinf(nuclide_half_life_s(nuclide))


YEAR_S = 86400.0 * 365.25

# The library's own "readable" strings are fine through ps..My (e.g.
# '0.2111 My') but reach for By/Gy/Ty/Py beyond that, which read as
# unfamiliar next to the much more common ky/My (and the dataset isn't even
# consistent about the cutover -- e.g. U-235 at 7.04e8 y, well within "My"
# range, is stored as '0.704 By'). Scientific notation in years is clearer
# for all of these, so swap in any string using one of those unit suffixes.
_EXTRA_LARGE_UNIT_SUFFIXES = (" By", " Gy", " Ty", " Py")


def half_life_readable(nuclide: str) -> str:
    """Human-readable half-life string (e.g. '30.1671 y', '4.468e+09 y', 'stable')."""
    readable = rd.Nuclide(nuclide).half_life("readable")
    if readable.endswith(_EXTRA_LARGE_UNIT_SUFFIXES):
        hl_y = nuclide_half_life_s(nuclide) / YEAR_S
        return f"{hl_y:.3e} y"
    return readable


def auto_time_grid_s(nuclide_names: list[str], steps: int = DEFAULT_STEPS) -> list[float]:
    """Log-spaced time grid (seconds) derived from the half-lives present.

    Stable nuclides (infinite half-life) are excluded from the range
    calculation since they don't bound a meaningful decay timescale. If
    every nuclide present is stable, or the list is empty, falls back to a
    1-second-to-1-year default range so the plot still renders something.
    """
    finite_half_lives = [
        hl for n in nuclide_names if not math.isinf(hl := nuclide_half_life_s(n))
    ]
    if not finite_half_lives:
        start, stop = 1.0, 3.15e7  # 1 s .. ~1 year
    else:
        start = min(finite_half_lives) * AUTO_RANGE_LOW_FACTOR
        stop = max(finite_half_lives) * AUTO_RANGE_HIGH_FACTOR
        if start <= 0:
            start = 1e-12
        if stop <= start:
            stop = start * 10

    log_start, log_stop = math.log10(start), math.log10(stop)
    if steps < 2:
        steps = 2
    return [0.0] + [
        10 ** (log_start + (log_stop - log_start) * i / (steps - 1)) for i in range(steps)
    ]


def time_grid_to_target_s(
    nuclide_names: list[str], target_time_s: float, steps: int = DEFAULT_STEPS
) -> list[float]:
    """Log-spaced time grid (seconds) from near-zero up to an exact target time.

    Used for the optional "show the fall-off" graph that accompanies the
    single-time results table: the grid's last point is always exactly
    ``target_time_s``, so the graph's rightmost point matches the table
    exactly. The start point reuses ``auto_time_grid_s``'s heuristic (1% of
    the shortest half-life present) so short-lived ingrowth/decay is still
    resolved even when the target time is much larger.
    """
    if target_time_s <= 0:
        return [0.0]

    finite_half_lives = [
        hl for n in nuclide_names if not math.isinf(hl := nuclide_half_life_s(n))
    ]
    start = min(finite_half_lives) * AUTO_RANGE_LOW_FACTOR if finite_half_lives else target_time_s / 1000
    if start <= 0:
        start = 1e-12

    if steps < 2:
        steps = 2

    if target_time_s <= start:
        # Target is very short relative to the half-lives present (or there
        # are none) -- log spacing would invert, so just linspace instead.
        return [target_time_s * i / (steps - 1) for i in range(steps)]

    log_start, log_stop = math.log10(start), math.log10(target_time_s)
    grid = [0.0] + [
        10 ** (log_start + (log_stop - log_start) * i / (steps - 1)) for i in range(steps)
    ]
    grid[-1] = target_time_s  # exact, not a float round-trip through log10/pow
    return grid


def run_time_series(canon: CanonResult, times_s: list[float]) -> TimeSeriesResult:
    """Decay ``canon`` across every time in ``times_s`` and reshape into series.

    ``times_s`` should be sorted ascending and include 0.0 for an identity
    baseline. Every nuclide that appears at *any* time step (inputs and
    in-grown progeny) is included, per spec Sec 5's "everything present
    across the whole series" requirement for the nuclide multi-select.
    """
    base_inventory = rd.Inventory(canon.contents, units=canon.library_unit)

    nuclide_set: set[str] = set()
    per_step_activities: list[dict[str, float]] = []
    per_step_masses: list[dict[str, float]] = []
    per_step_moles: list[dict[str, float]] = []
    per_step_atoms: list[dict[str, float]] = []

    for t in times_s:
        decayed = base_inventory.decay(t, "s")
        activities = decayed.activities("Bq")
        masses = decayed.masses("g")
        moles = decayed.moles("mol")
        atoms = decayed.numbers()

        nuclide_set.update(str(n) for n in decayed.nuclides)
        per_step_activities.append({str(k): float(v) for k, v in activities.items()})
        per_step_masses.append({str(k): float(v) for k, v in masses.items()})
        per_step_moles.append({str(k): float(v) for k, v in moles.items()})
        per_step_atoms.append({str(k): float(v) for k, v in atoms.items()})

    nuclides = sorted(nuclide_set)
    half_lives_s = {n: nuclide_half_life_s(n) for n in nuclides}

    def _series(per_step: list[dict[str, float]]) -> dict[str, list[float]]:
        return {n: [step.get(n, 0.0) for step in per_step] for n in nuclides}

    activities_bq = _series(per_step_activities)
    masses_g = _series(per_step_masses)
    moles_mol = _series(per_step_moles)
    atoms_series = _series(per_step_atoms)

    fractions: dict[str, list[float]] = {}
    if canon.kind.startswith("fraction_"):
        base_series = {
            "fraction_activity": activities_bq,
            "fraction_mass": masses_g,
            "fraction_mole": moles_mol,
        }[canon.kind]
        scale = 100.0 if canon.frac_as_percent else 1.0
        totals = [sum(base_series[n][i] for n in nuclides) for i in range(len(times_s))]
        fractions = {
            n: [
                (base_series[n][i] / totals[i] * scale) if totals[i] else 0.0
                for i in range(len(times_s))
            ]
            for n in nuclides
        }

    return TimeSeriesResult(
        times_s=list(times_s),
        nuclides=nuclides,
        half_lives_s=half_lives_s,
        activities_bq=activities_bq,
        masses_g=masses_g,
        moles_mol=moles_mol,
        atoms=atoms_series,
        kind=canon.kind,
        display_unit=canon.display_unit,
        frac_as_percent=canon.frac_as_percent,
        fractions=fractions,
    )


def audit_conservation(result: TimeSeriesResult, rtol: float = 1e-4) -> list[str]:
    """Layer-3 conservation self-audit (validation spec Sec 6): laws that
    must hold on *every* calculation regardless of input, run live as a
    silently-wrong detector rather than only in the test suite.

    Returns a list of human-readable breach messages; an empty list means
    the result passed. Checks the calculation's *outputs* -- no negative
    quantities, fraction renormalization, and atom-count conservation
    across the series (requires a t=0 baseline). The static
    branching-fractions-sum-to-1 law is a property of the nuclear data, not
    of a calculation, so it stays a one-time test (test_engine) rather than
    a per-run cost here. Tolerances are deliberately loose: float64
    cancellation over extreme half-life spreads produces ~1e-12-relative
    noise, which is physics of the solver, not a bug.
    """
    breaches: list[str] = []

    # 1. No meaningfully-negative quantities anywhere. The floor scales with
    #    the largest magnitude in each series so cancellation noise is ignored.
    for label, series in (
        ("activity", result.activities_bq),
        ("mass", result.masses_g),
        ("moles", result.moles_mol),
        ("atoms", result.atoms),
    ):
        scale = max((abs(v) for vals in series.values() for v in vals), default=0.0)
        floor = 1e-9 * scale
        for nuclide, vals in series.items():
            worst = min(vals, default=0.0)
            if worst < -floor:
                breaches.append(
                    f"Negative {label} for {nuclide} ({worst:.3g}): quantities cannot go "
                    f"negative under decay."
                )

    # 2. Fraction renormalization: fraction-mode outputs sum to 1 (or 100).
    if result.fractions:
        target = 100.0 if result.frac_as_percent else 1.0
        for i, t in enumerate(result.times_s):
            total = sum(result.fractions[n][i] for n in result.nuclides)
            if total > 0 and abs(total - target) > rtol * target:
                breaches.append(
                    f"Fractions sum to {total:.6g} at t={t:.3g} s (expected {target:g})."
                )
                break

    # 3. Atom-count conservation across the series (one parent nucleus becomes
    #    one daughter nucleus at every decay, so the total is invariant). Only
    #    checkable when a t=0 baseline is present.
    if len(result.times_s) >= 2 and result.times_s[0] == 0.0:
        totals = [
            sum(result.atoms[n][i] for n in result.nuclides) for i in range(len(result.times_s))
        ]
        initial = totals[0]
        if initial > 0:
            for i, tot in enumerate(totals):
                if abs(tot - initial) > rtol * initial:
                    breaches.append(
                        f"Atom count not conserved: {tot:.6g} at t={result.times_s[i]:.3g} s "
                        f"vs {initial:.6g} at t=0."
                    )
                    break

    return breaches


def filter_nuclides_by_half_life(
    result: TimeSeriesResult,
    threshold_s: float | None,
    direction: str = "above",
    include_stable: bool = True,
) -> list[str]:
    """Apply the half-life filter from spec Sec 5.2.

    ``direction`` is "above" or "below" the threshold. Stable nuclides
    (infinite half-life) are governed by the separate ``include_stable``
    toggle rather than the threshold comparison, since "infinity > x" would
    otherwise always include them under "above" and never under "below".
    """
    if direction not in ("above", "below"):
        raise ValueError("direction must be 'above' or 'below'.")

    selected = []
    for n in result.nuclides:
        hl = result.half_lives_s[n]
        if math.isinf(hl):
            if include_stable:
                selected.append(n)
            continue
        if threshold_s is None:
            selected.append(n)
        elif direction == "above" and hl >= threshold_s:
            selected.append(n)
        elif direction == "below" and hl <= threshold_s:
            selected.append(n)
    return selected
