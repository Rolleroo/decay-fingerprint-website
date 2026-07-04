"""Compatibility check (ratio / age hypothesis test): fix the assumed
original composition AND the age, decay forward once, and score today's
measurement with zero free parameters. Same discipline as the age solve --
these tests exist before the verdict is relied on.
"""

import math

import pytest
import radioactivedecay as rd

from app.age_solve import YEAR_S, check_compatibility, check_compatibility_from_entries
from app.conversions import ValidationError, canonicalize
from app.parsing import parse_paste


def canon_atoms(contents: dict[str, float]):
    paste = "\n".join(f"{n}, {float(v)!r}" for n, v in contents.items())
    result = parse_paste(paste)
    assert result.ok, result.errors
    return canonicalize(result.entries, "atoms")


def forward_atoms(t0_contents: dict[str, float], age_s: float) -> dict[str, float]:
    decayed = rd.Inventory(t0_contents, "num").decay(age_s, "s")
    return {str(k): float(v) for k, v in decayed.numbers().items()}


# --- the core verdict: right age vs wrong age ---


def test_compatible_when_measurement_matches_the_forward_prediction():
    t0 = {"Cs-137": 1.0e15, "Sr-90": 4.0e14}
    age = 30.0 * YEAR_S
    today = forward_atoms(t0, age)
    result = check_compatibility(
        canon_atoms(t0), canon_atoms({n: today[n] for n in t0}), age,
        default_rel_sigma=0.05,
    )
    assert result.verdict == "compatible"
    assert result.chi2 == pytest.approx(0.0, abs=1e-9)
    assert result.p_value == pytest.approx(1.0, abs=1e-6)
    assert result.dof == 2  # zero free parameters: dof = number of measured


def test_incompatible_when_the_assumed_age_is_wrong():
    t0 = {"Cs-137": 1.0e15, "Sr-90": 4.0e14}
    today = forward_atoms(t0, 30.0 * YEAR_S)
    # Same data, but test it against a 5-year hypothesis: the two chains decay
    # at different rates, so no single wrong age reproduces the pattern.
    result = check_compatibility(
        canon_atoms(t0), canon_atoms({n: today[n] for n in t0}), 5.0 * YEAR_S,
        default_rel_sigma=0.05,
    )
    assert result.verdict == "incompatible"
    assert result.p_value < 1e-3
    assert result.chi2_per_dof > 3.0


# --- the ratio / pattern hypothesis (overall scale free) ---


def test_scale_free_pattern_matches_when_only_the_overall_amount_is_off():
    # Measure exactly 2x the aged composition: the isotopic *pattern* is a
    # perfect match, only the overall level is wrong. The absolute-amount
    # verdict must reject it; the ratio verdict must accept it with scale ~2.
    t0 = {"Cs-137": 1.0e15, "Sr-90": 4.0e14}
    age = 20.0 * YEAR_S
    aged = forward_atoms(t0, age)
    today = {n: 2.0 * aged[n] for n in t0}
    result = check_compatibility(
        canon_atoms(t0), canon_atoms(today), age, default_rel_sigma=0.05,
    )
    assert result.verdict == "incompatible"  # absolute amounts are 2x off
    assert result.ratio_testable
    assert result.scale == pytest.approx(2.0, rel=1e-6)
    assert result.verdict_scaled == "compatible"
    assert result.chi2_scaled == pytest.approx(0.0, abs=1e-6)
    assert result.dof_scaled == 1


def test_single_nuclide_has_no_ratio_to_test():
    t0 = {"Cs-137": 1.0e15}
    age = 10.0 * YEAR_S
    today = forward_atoms(t0, age)
    result = check_compatibility(
        canon_atoms(t0), canon_atoms({"Cs-137": today["Cs-137"]}), age,
    )
    assert not result.ratio_testable
    assert result.verdict_scaled == "n/a"
    assert math.isnan(result.scale)


# --- producibility: a measured nuclide the origin can't make ---


def test_unproducible_nuclide_is_flagged_as_a_compatibility_breaker():
    t0 = {"Sr-90": 1.0e20}
    age = 10.0 * YEAR_S
    today = forward_atoms(t0, age)
    # Cs-137 cannot come from an Sr-90 source at any age.
    result = check_compatibility(
        canon_atoms(t0),
        canon_atoms({"Sr-90": today["Sr-90"], "Cs-137": 1.0e10}),
        age,
    )
    assert result.excluded_unproducible == ["Cs-137"]
    assert any("cannot arise" in w for w in result.warnings)
    # The producible part still matches, so the numerical score stays clean.
    assert result.verdict == "compatible"


# --- input guards ---


def test_zero_or_negative_age_is_refused():
    t0 = canon_atoms({"Cs-137": 1.0e15})
    today = canon_atoms({"Cs-137": 1.0e15})
    with pytest.raises(ValidationError, match="positive amount of time"):
        check_compatibility(t0, today, 0.0)


def test_fraction_units_are_refused():
    parse = parse_paste("Cs-137, 0.6\nSr-90, 0.4")
    canon_frac = canonicalize(parse.entries, "activity fraction")
    with pytest.raises(ValidationError, match="fraction/percent"):
        check_compatibility(canon_frac, canon_atoms({"Cs-137": 1.0e15}), 10.0 * YEAR_S)


# --- the from_entries convenience threads uncertainties through ---


def test_from_entries_uses_pasted_uncertainties():
    t0 = {"Cs-137": 1.0e15, "Sr-90": 4.0e14}
    age = 30.0 * YEAR_S
    aged = forward_atoms(t0, age)
    # A 3-sigma high miss on Cs-137 at a stated 1% uncertainty should register
    # as a ~3-sigma pull, i.e. tension, not a clean pass.
    cs = aged["Cs-137"] * 1.03
    today_paste = f"Cs-137, {cs!r}, 1%\nSr-90, {aged['Sr-90']!r}, 1%"
    parse_today = parse_paste(today_paste)
    canon_today = canonicalize(parse_today.entries, "atoms")
    result = check_compatibility_from_entries(
        canon_atoms(t0), parse_today.entries, canon_today, age,
    )
    cs_row = next(r for r in result.residuals if r.nuclide == "Cs-137")
    assert cs_row.mismatch_sigma == pytest.approx(-3.0, abs=0.3)
    assert result.verdict in ("marginal", "incompatible")
