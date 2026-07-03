"""Mode A validation, per docs/mode-a-addendum.md Sec 5: analytical anchor,
round-trips, the resolvability/ambiguity/consistency gates, MC coverage,
and the producibility rule. Same discipline as Mode B: these exist before
the output is relied on.
"""

import math

import pytest
import radioactivedecay as rd

from app.age_solve import YEAR_S, age_readable, solve_age
from app.conversions import ValidationError, canonicalize
from app.engine import nuclide_half_life_s
from app.parsing import parse_paste


def canon_atoms(contents: dict[str, float]):
    paste = "\n".join(f"{n}, {float(v)!r}" for n, v in contents.items())
    result = parse_paste(paste)
    assert result.ok, result.errors
    return canonicalize(result.entries, "atoms")


def forward_atoms(t0_contents: dict[str, float], age_s: float) -> dict[str, float]:
    decayed = rd.Inventory(t0_contents, "num").decay(age_s, "s")
    return {str(k): float(v) for k, v in decayed.numbers().items()}


def exact_solve(t0: dict[str, float], today: dict[str, float], **kwargs):
    """Deterministic solve (exact measurements) for anchor tests."""
    return solve_age(canon_atoms(t0), canon_atoms(today), default_rel_sigma=0.0, **kwargs)


# --- Layer 1: analytical anchor ---


def test_single_nuclide_three_half_lives_exact():
    # today = t0/8, so age = 3 half-lives, digit-for-digit from the
    # closed form t = T1/2 * log2(N0/N) -- no library value used anywhere.
    hl = nuclide_half_life_s("Cs-137")
    result = exact_solve({"Cs-137": 1.0e15}, {"Cs-137": 1.0e15 / 8.0})
    assert result.age_s == pytest.approx(3.0 * hl, rel=1e-6)
    assert result.resolvable
    assert not result.ambiguous_ages_s


# --- Layer 4: round-trips against the trusted forward engine ---


def test_round_trip_pu241_am241():
    truth_age = 25.0 * YEAR_S
    t0 = {"Pu-241": 1.0e15, "Am-241": 2.0e13}
    today = forward_atoms(t0, truth_age)
    # Realistic 1% uncertainties: consistent data must fit essentially
    # perfectly (chi2/dof ~ 0) and the central age must still be exact,
    # since the central solve uses the unperturbed values.
    result = solve_age(
        canon_atoms(t0),
        canon_atoms({n: today[n] for n in t0}),
        default_rel_sigma=0.01,
        n_trials=2_000,
        seed=7,
    )
    assert result.age_s == pytest.approx(truth_age, rel=1e-6)
    assert result.resolvable
    assert result.chi2_per_dof < 1e-3
    assert result.age_s_lo < truth_age < result.age_s_hi


def test_round_trip_stable_daughter_chronometer():
    # Zr-90 is stable: excluded by Mode B, but in Mode A radiogenic
    # accumulation is a classic clock. It must participate in the fit.
    truth_age = 30.0 * YEAR_S
    t0 = {"Sr-90": 1.0e20}
    today = forward_atoms(t0, truth_age)
    result = exact_solve(t0, {n: today[n] for n in ("Sr-90", "Y-90", "Zr-90")})
    assert result.age_s == pytest.approx(truth_age, rel=1e-6)
    zr = next(r for r in result.residuals if r.nuclide == "Zr-90")
    assert zr.informative  # the stable daughter genuinely constrains the age


# --- input guards ---


def test_stable_nuclide_in_activity_unit_is_refused_with_a_message():
    parse = parse_paste("Sr-90, 1000\nZr-90, 50")
    canon_bq = canonicalize(parse.entries, "Bq")
    with pytest.raises(ValidationError, match="activity unit"):
        solve_age(canon_atoms({"Sr-90": 1.0e20}), canon_bq)


def test_fraction_units_are_refused_in_v1():
    parse = parse_paste("Cs-137, 0.6\nCo-60, 0.4")
    canon_frac = canonicalize(parse.entries, "activity fraction")
    with pytest.raises(ValidationError, match="relative"):
        solve_age(canon_frac, canon_atoms({"Cs-137": 1.0e15}))


def test_unproducible_measured_nuclide_is_excluded_not_fitted():
    truth_age = 10.0 * YEAR_S
    t0 = {"Sr-90": 1.0e20}
    today = forward_atoms(t0, truth_age)
    result = exact_solve(t0, {"Sr-90": today["Sr-90"], "Cs-137": 1.0e10})
    assert result.excluded_unproducible == ["Cs-137"]
    assert any("cannot be produced" in w for w in result.warnings)
    # The intruder must not have corrupted the age.
    assert result.age_s == pytest.approx(truth_age, rel=1e-6)


# --- gates ---


def test_unresolvable_when_nothing_measurably_decays():
    # U-238 is unchanged over any plausible lab age at 5% uncertainty:
    # the age must be refused, not fabricated.
    result = solve_age(
        canon_atoms({"U-238": 1.0e20}),
        canon_atoms({"U-238": 1.0e20}),
        default_rel_sigma=0.05,
        n_trials=4_000,
        seed=3,
    )
    assert not result.resolvable
    assert any("Not resolvable" in w for w in result.warnings)


def test_ambiguous_age_from_single_ingrowth_measurement():
    # A lone Tc-99m level from pure Mo-99 occurs once before and once
    # after the ingrowth peak (~23 h): both candidate ages must be
    # reported, never silently collapsed to one.
    t0 = {"Mo-99": 1.0e15}
    early = 6 * 3600.0
    today_tc99m = forward_atoms(t0, early)["Tc-99m"]
    result = exact_solve(t0, {"Tc-99m": today_tc99m})
    assert len(result.ambiguous_ages_s) == 2
    assert result.ambiguous_ages_s[0] == pytest.approx(early, rel=0.05)
    assert result.ambiguous_ages_s[1] > 23 * 3600.0
    assert any("Ambiguous age" in w for w in result.warnings)


def test_inconsistent_inputs_flagged_by_chi2():
    # Two chains that imply different ages (as if the sample were opened):
    # the fit must complain loudly, not average silently.
    t0 = {"Sr-90": 1.0e20, "Cs-137": 1.0e20}
    today_sr = forward_atoms({"Sr-90": 1.0e20}, 10.0 * YEAR_S)["Sr-90"]
    today_cs = forward_atoms({"Cs-137": 1.0e20}, 25.0 * YEAR_S)["Cs-137"]
    result = solve_age(
        canon_atoms(t0),
        canon_atoms({"Sr-90": today_sr, "Cs-137": today_cs}),
        default_rel_sigma=0.02,
        n_trials=2_000,
        seed=5,
    )
    assert result.chi2_per_dof > 3.0
    assert any("Poor fit" in w for w in result.warnings)


def test_trace_daughter_far_below_parent_scale_keeps_its_real_sigma():
    # Regression (found in revalidation): the sigma floor for "exact"
    # inputs was a single global value keyed to the largest magnitude in
    # the fit, so a trace-level nuclide (in-grown Th-234 at ~1e13 atoms)
    # measured alongside its enormous parent (U-238 at 1e24 atoms) had its
    # real 2% uncertainty silently replaced by a floor a million times
    # larger -- erasing its weight in the fit and leaving the age
    # unconstrained. The floor must be per-nuclide and must only fill in
    # where sigma is exactly zero.
    truth_age = 30.0 * 86400.0  # ~1.24 Th-234 half-lives: strong sensitivity
    t0 = {"U-238": 1.0e24}
    today = forward_atoms(t0, truth_age)
    assert today["Th-234"] < 1e-9 * today["U-238"]  # the scale gap under test

    result = solve_age(
        canon_atoms(t0),
        canon_atoms({"U-238": today["U-238"], "Th-234": today["Th-234"]}),
        default_rel_sigma=0.02,
        n_trials=4_000,
        seed=13,
    )
    # Th-234 carries all the age information (U-238 is unchanged over 30
    # days); with its sigma intact the age resolves cleanly, and the
    # measured parent kills the late-branch solution -> no ambiguity.
    assert result.resolvable
    assert not result.ambiguous_ages_s
    assert result.age_s_lo < truth_age < result.age_s_hi
    assert result.age_s == pytest.approx(truth_age, rel=1e-3)


def test_lone_ingrowth_measurement_under_long_parent_is_ambiguous():
    # Companion physics check (documented during revalidation): a *lone*
    # Th-234 measurement under a U-238 source is genuinely two-valued --
    # the ingrowth curve rises to equilibrium (~30 d branch) then falls
    # with U-238's own decay, so a ~3.5-Gyr age reproduces the same value.
    # The solver must surface both candidates rather than certify either.
    truth_age = 30.0 * 86400.0
    t0 = {"U-238": 1.0e24}
    today = forward_atoms(t0, truth_age)

    result = solve_age(
        canon_atoms(t0),
        canon_atoms({"Th-234": today["Th-234"]}),
        default_rel_sigma=0.02,
        n_trials=4_000,
        seed=13,
    )
    assert len(result.ambiguous_ages_s) == 2
    assert result.ambiguous_ages_s[0] == pytest.approx(truth_age, rel=0.05)
    assert result.ambiguous_ages_s[1] > 1.0e9 * YEAR_S  # the falling branch
    assert not result.resolvable  # interval spans the branches; refuse


# --- Monte Carlo coverage ---


def test_mc_interval_covers_truth_with_realistic_noise():
    truth_age = nuclide_half_life_s("Cs-137")  # one half-life back
    t0 = {"Cs-137": 1.0e15}
    perturbed_today = forward_atoms(t0, truth_age)["Cs-137"] * 1.05  # a +1 sigma miss
    result = solve_age(
        canon_atoms(t0),
        canon_atoms({"Cs-137": perturbed_today}),
        default_rel_sigma=0.05,
        n_trials=20_000,
        seed=11,
    )
    assert result.resolvable
    assert result.age_s_lo < truth_age < result.age_s_hi
    assert result.age_s_median == pytest.approx(truth_age, rel=0.25)


def test_zero_sigma_solve_is_deterministic():
    hl = nuclide_half_life_s("Co-60")
    a = exact_solve({"Co-60": 1.0e12}, {"Co-60": 2.5e11})
    b = exact_solve({"Co-60": 1.0e12}, {"Co-60": 2.5e11})
    assert a.age_s == b.age_s == pytest.approx(2.0 * hl, rel=1e-6)
    assert a.n_trials == 0
    assert a.age_s_median == a.age_s


# --- formatting ---


def test_age_readable_picks_sensible_units():
    assert "seconds" in age_readable(45.0)
    assert "hours" in age_readable(7200.0)
    assert "days" in age_readable(30 * 86400.0)
    assert "years" in age_readable(30 * YEAR_S)
    assert "e+" in age_readable(1.0e9 * YEAR_S)  # scientific beyond 1e5 years
