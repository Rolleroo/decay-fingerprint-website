import math

import pytest
import radioactivedecay as rd

import dataclasses

from app.conversions import canonicalize
from app.engine import (
    audit_conservation,
    auto_time_grid_s,
    filter_nuclides_by_half_life,
    half_life_readable,
    is_stable,
    nuclide_half_life_s,
    run_time_series,
    time_grid_to_target_s,
)
from app.parsing import parse_paste


def entries_of(text):
    result = parse_paste(text)
    assert result.ok, result.errors
    return result.entries


def series_for(text, unit, frac_as_percent=False, times_s=None):
    canon = canonicalize(entries_of(text), unit, frac_as_percent=frac_as_percent)
    if times_s is None:
        times_s = auto_time_grid_s([e.nuclide for e in entries_of(text)])
    return run_time_series(canon, times_s)


# --- single-nuclide / identity / monotonicity ---


def test_single_nuclide_half_life_halves_activity():
    hl = nuclide_half_life_s("Co-60")
    result = series_for("Co-60, 1000", "Bq", times_s=[0.0, hl])
    assert result.activities_bq["Co-60"][0] == pytest.approx(1000.0)
    assert result.activities_bq["Co-60"][1] == pytest.approx(500.0, rel=1e-3)


def test_identity_decay_time_zero_returns_input_unchanged():
    result = series_for("Cs-137, 777", "Bq", times_s=[0.0])
    assert result.activities_bq["Cs-137"][0] == pytest.approx(777.0)


def test_lone_parent_activity_only_ever_decreases():
    times = auto_time_grid_s(["Co-60"])
    result = series_for("Co-60, 1000", "Bq", times_s=times)
    series = result.activities_bq["Co-60"]
    assert all(series[i + 1] <= series[i] + 1e-9 for i in range(len(series) - 1))


# --- two-member chain vs analytic Bateman ---


def test_two_member_chain_matches_analytic_bateman():
    # Sr-90 -> Y-90 -> (stable Zr-90), no branching on either step, so the
    # closed-form two-member Bateman solution applies directly.
    lam1 = math.log(2) / nuclide_half_life_s("Sr-90")
    lam2 = math.log(2) / nuclide_half_life_s("Y-90")
    n1_0 = 1.0e15

    times = [1e5, 1e6, 1e7]
    result = series_for("Sr-90, 1.0e15", "atoms", times_s=[0.0] + times)

    for t in times:
        n1_t = n1_0 * math.exp(-lam1 * t)
        n2_t = n1_0 * lam1 / (lam2 - lam1) * (math.exp(-lam1 * t) - math.exp(-lam2 * t))
        idx = result.times_s.index(t)
        assert result.atoms["Sr-90"][idx] == pytest.approx(n1_t, rel=1e-6)
        assert result.atoms["Y-90"][idx] == pytest.approx(n2_t, rel=1e-6)


# --- progeny ingrowth ---


def test_progeny_ingrowth_mo99_to_tc99m():
    times = [0.0, 3600.0, 6 * 3600.0, 24 * 3600.0, 200 * 3600.0]
    result = series_for("Mo-99, 1.0e6", "Bq", times_s=times)
    assert "Tc-99m" in result.nuclides
    tc99m = result.activities_bq["Tc-99m"]
    assert tc99m[0] == 0.0
    assert tc99m[1] > tc99m[0]  # ingrowth: starts rising from zero
    assert tc99m[-1] < max(tc99m)  # eventually falls back off after the peak


# --- branching ---


def test_branching_fractions_sum_to_one():
    for nuclide in ["K-40", "Mo-99"]:
        assert sum(rd.Nuclide(nuclide).branching_fractions()) == pytest.approx(1.0, abs=1e-6)


# --- secular equilibrium ---


def test_secular_equilibrium_long_parent_short_daughter():
    # Sr-90 (~28.8 y) >> Y-90 (~64 h): after many Y-90 half-lives but a
    # negligible fraction of a Sr-90 half-life, daughter activity should
    # approach parent activity.
    y90_hl = nuclide_half_life_s("Y-90")
    t = 15 * y90_hl
    result = series_for("Sr-90, 1.0e6", "Bq", times_s=[0.0, t])
    parent = result.activities_bq["Sr-90"][1]
    daughter = result.activities_bq["Y-90"][1]
    assert daughter == pytest.approx(parent, rel=0.02)


# --- conservation ---


def test_atom_conservation_across_decay_chain():
    times = [0.0, 1e9, 1e15, 1e17]
    result = series_for("U-238, 1.0e20", "atoms", times_s=times)
    initial_total = sum(result.atoms[n][0] for n in result.nuclides)
    for i in range(len(times)):
        total_at_t = sum(result.atoms[n][i] for n in result.nuclides)
        assert total_at_t == pytest.approx(initial_total, rel=1e-6)


def test_fraction_renormalization_sums_to_one_at_every_step():
    times = [0.0, 1e5, 1e7, 1e9]
    result = series_for("Cs-137, 60\nCo-60, 40", "activity fraction", frac_as_percent=True, times_s=times)
    for i in range(len(times)):
        total = sum(result.fractions[n][i] for n in result.nuclides)
        assert total == pytest.approx(100.0, abs=1e-6)


# --- time grid ---


def test_auto_time_grid_starts_at_zero_and_is_ascending():
    grid = auto_time_grid_s(["Co-60", "Sr-90"])
    assert grid[0] == 0.0
    assert all(grid[i] < grid[i + 1] for i in range(len(grid) - 1))


def test_auto_time_grid_all_stable_falls_back_to_default_range():
    grid = auto_time_grid_s(["Pb-208"])  # stable
    assert grid[0] == 0.0
    assert grid[-1] > grid[1] > 0


def test_time_grid_to_target_ends_exactly_at_target():
    grid = time_grid_to_target_s(["Cs-137", "Co-60"], 1.0e9)
    assert grid[-1] == 1.0e9
    assert grid[0] == 0.0
    assert all(grid[i] < grid[i + 1] for i in range(len(grid) - 1))


def test_time_grid_to_target_has_100_points_by_default():
    grid = time_grid_to_target_s(["Cs-137"], 1.0e9)
    assert len(grid) == 101  # 0.0 plus DEFAULT_STEPS log-spaced points


def test_time_grid_to_target_zero_returns_single_point():
    assert time_grid_to_target_s(["Cs-137"], 0.0) == [0.0]


def test_time_grid_to_target_short_relative_to_half_lives_falls_back_to_linspace():
    grid = time_grid_to_target_s(["U-238"], 10.0)  # 10s vs billions of years
    assert grid[0] == 0.0
    assert grid[-1] == 10.0
    assert all(grid[i] <= grid[i + 1] for i in range(len(grid) - 1))


# --- half-life filter ---


def test_half_life_filter_above_and_below():
    result = series_for("Mo-99, 1.0e6", "Bq", times_s=[0.0, 3600.0, 200 * 3600.0])
    one_day_s = 86400.0
    above = filter_nuclides_by_half_life(result, one_day_s, direction="above")
    below = filter_nuclides_by_half_life(result, one_day_s, direction="below")
    assert "Mo-99" in above  # ~66 h half-life
    assert "Tc-99m" in below  # ~6 h half-life
    assert "Tc-99m" not in above


def test_half_life_filter_stable_toggle():
    result = series_for("Mo-99, 1.0e6", "Bq", times_s=[0.0, 200 * 3600.0])
    stable = [n for n in result.nuclides if is_stable(n)]
    assert stable, "expected a stable end product in the Mo-99 chain"
    included = filter_nuclides_by_half_life(result, None, direction="above", include_stable=True)
    excluded = filter_nuclides_by_half_life(result, None, direction="above", include_stable=False)
    for n in stable:
        assert n in included
        assert n not in excluded


# --- numerical stability ---


def test_half_life_readable_uses_scientific_notation_beyond_mega_years():
    # U-235's half-life (7.04e8 y) is stored by the library as '0.704 By',
    # which is well within My range numerically -- it should read as
    # scientific notation in years instead of the unfamiliar By unit.
    assert half_life_readable("U-235") == "7.040e+08 y"


def test_half_life_readable_keeps_kilo_and_mega_year_units():
    assert half_life_readable("Tc-99") == "0.2111 My"


def test_half_life_readable_keeps_sub_year_units_unchanged():
    assert half_life_readable("Cs-137") == "30.1671 y"
    assert half_life_readable("Tc-99m") == "6.015 h"


def test_half_life_readable_stable():
    assert half_life_readable("Pb-208") == "stable"


# --- Layer-3 runtime conservation audit ---


def test_audit_clean_on_normal_chain():
    result = series_for("U-238, 1.0e20", "atoms", times_s=[0.0, 1e9, 1e15])
    assert audit_conservation(result) == []


def test_audit_clean_on_fraction_mode():
    result = series_for(
        "Cs-137, 60\nCo-60, 40", "activity fraction", frac_as_percent=True,
        times_s=[0.0, 1e7, 1e9],
    )
    assert audit_conservation(result) == []


def test_audit_clean_under_extreme_half_life_spread():
    # The Po-214 / U-238 span produces ~1e-12-relative float noise (incl.
    # tiny negatives); the audit's tolerances must not flag it.
    times = auto_time_grid_s(["Po-214", "U-238"])
    result = series_for("Po-214, 1.0e6\nU-238, 1.0e6", "Bq", times_s=times)
    assert audit_conservation(result) == []


def test_audit_detects_injected_negative():
    result = series_for("Co-60, 1000", "Bq", times_s=[0.0, 3600.0])
    poisoned = dataclasses.replace(
        result, activities_bq={**result.activities_bq, "Co-60": [1000.0, -5.0]}
    )
    breaches = audit_conservation(poisoned)
    assert any("Negative activity for Co-60" in b for b in breaches)


def test_audit_detects_atom_nonconservation():
    result = series_for("U-238, 1.0e20", "atoms", times_s=[0.0, 1e15])
    atoms = {n: list(v) for n, v in result.atoms.items()}
    atoms["U-238"][1] *= 0.5  # destroy half the atoms out of nowhere
    poisoned = dataclasses.replace(result, atoms=atoms)
    breaches = audit_conservation(poisoned)
    assert any("Atom count not conserved" in b for b in breaches)


def test_audit_detects_bad_fraction_sum():
    result = series_for(
        "Cs-137, 0.6\nCo-60, 0.4", "activity fraction", frac_as_percent=False,
        times_s=[0.0, 1e7],
    )
    fractions = {n: list(v) for n, v in result.fractions.items()}
    fractions["Cs-137"][1] += 0.5  # now sums to ~1.5
    poisoned = dataclasses.replace(result, fractions=fractions)
    breaches = audit_conservation(poisoned)
    assert any("Fractions sum to" in b for b in breaches)


def test_extreme_half_life_spread_does_not_break_the_solve():
    # Po-214 (~164 microseconds) alongside U-238 (~4.5e9 years) in one
    # inventory, decayed across a grid spanning both timescales. The
    # float64 solver can produce ~1e-14-scale negative noise from
    # cancellation at this span (not a bug -- this is what the high-
    # precision InventoryHP path exists for); the bar here is just "no
    # crash, no NaN/Inf, no *meaningful* negative activity".
    times = auto_time_grid_s(["Po-214", "U-238"])
    result = series_for("Po-214, 1.0e6\nU-238, 1.0e6", "Bq", times_s=times)
    for nuclide, series in result.activities_bq.items():
        for value in series:
            assert math.isfinite(value), f"{nuclide} produced a non-finite activity"
            assert value >= -1e-6
