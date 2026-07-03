"""Mode B validation: analytical anchors (Layer 1), round-trips (Layer 4),
the unmeasured-nuclide rule check (Layer 4a), and the gating/flagging
behaviour that makes backward results trustworthy (reverse spec Sec 6).

Round-trip tests use the *library's* forward decay to manufacture the
present-day measurement from a known t=0 truth, then require Mode B to
return that truth. The input is the ground truth, so no external source
is needed; what this cannot prove (that the forward model itself is
right) is covered once by the independent cross-check in
test_validation.py (Layer 2).
"""

import math

import pytest
import radioactivedecay as rd

from app.conversions import canonicalize
from app.engine import nuclide_half_life_s
from app.parsing import parse_paste
from app.reverse import (
    GATE_HALF_LIVES,
    reconstruct_t0,
    sigma_atoms_from_entries,
)

YEAR_S = 86400.0 * 365.25


def canon_atoms(contents: dict[str, float]):
    """Build a CanonResult as if the user pasted these atom counts."""
    paste = "\n".join(f"{n}, {float(v)!r}" for n, v in contents.items())
    result = parse_paste(paste)
    assert result.ok, result.errors
    return canonicalize(result.entries, "atoms")


def forward_atoms(t0_contents: dict[str, float], age_s: float) -> dict[str, float]:
    """Manufacture the present-day measurement from a known t=0 truth."""
    decayed = rd.Inventory(t0_contents, "num").decay(age_s, "s")
    return {str(k): float(v) for k, v in decayed.numbers().items()}


def exact(canon, age_s, **kwargs):
    """Deterministic reconstruction (no MC spread) for anchor tests."""
    return reconstruct_t0(canon, age_s, default_rel_sigma=0.0, **kwargs)


def row(result, nuclide):
    return next(r for r in result.rows if r.nuclide == nuclide)


# --- Layer 1: analytical anchors (exact, external truth) ---


def test_single_nuclide_one_half_life_back_doubles_exactly():
    hl = nuclide_half_life_s("Cs-137")
    measured = 1.0e15  # atoms today
    result = exact(canon_atoms({"Cs-137": measured}), hl)
    r = row(result, "Cs-137")
    assert r.median_atoms == pytest.approx(2.0 * measured, rel=1e-9)
    assert r.conditioning == "pass"
    assert result.forward_check_ok


def test_two_member_chain_matches_hand_evaluated_bateman():
    # Sr-90 -> Y-90 (no branching): manufacture the present-day values from
    # the closed-form two-member Bateman solution -- evaluated here by hand,
    # independent of the library -- and require Mode B to return the chosen
    # t=0 amounts.
    lam1 = math.log(2) / nuclide_half_life_s("Sr-90")
    lam2 = math.log(2) / nuclide_half_life_s("Y-90")
    n1_0, n2_0 = 1.0e15, 3.0e12
    t = 30 * 86400.0  # 30 days: ~11 Y-90 half-lives, well inside the gate

    n1_t = n1_0 * math.exp(-lam1 * t)
    n2_t = (
        n1_0 * lam1 / (lam2 - lam1) * (math.exp(-lam1 * t) - math.exp(-lam2 * t))
        + n2_0 * math.exp(-lam2 * t)
    )

    result = exact(canon_atoms({"Sr-90": n1_t, "Y-90": n2_t}), t)
    assert row(result, "Sr-90").median_atoms == pytest.approx(n1_0, rel=1e-6)
    assert row(result, "Y-90").median_atoms == pytest.approx(n2_0, rel=1e-4)
    assert result.forward_check_ok


# --- Layer 4: round-trip (forward with the trusted engine, then back) ---


def test_round_trip_pu241_am241_dating_case():
    # The Pu-241/Am-241 pair from the spec's minimum viable targets.
    truth = {"Pu-241": 1.0e15, "Am-241": 2.0e13}
    age = 30 * YEAR_S
    today = forward_atoms(truth, age)

    result = exact(canon_atoms({n: today[n] for n in truth}), age)
    assert row(result, "Pu-241").median_atoms == pytest.approx(truth["Pu-241"], rel=1e-8)
    assert row(result, "Am-241").median_atoms == pytest.approx(truth["Am-241"], rel=1e-6)
    assert result.forward_check_ok
    assert all(r.conditioning == "pass" for r in result.rows)
    assert not any(r.chain_tainted for r in result.rows)


def test_round_trip_u238_chain_million_year_reachback():
    # Long-chain case: only the members whose half-lives can carry
    # information back 1e6 years are measured; the short-lived intermediates
    # between them (Th-234, Pa-234m) are unmeasured and must be pruned as
    # negligible, per the DQPB rule, without breaking the reconstruction.
    truth = {"U-238": 1.0e24, "U-234": 3.0e20, "Th-230": 5.0e19}
    age = 1.0e6 * YEAR_S
    today = forward_atoms(truth, age)

    result = exact(canon_atoms({n: today[n] for n in truth}), age)
    assert row(result, "U-238").median_atoms == pytest.approx(truth["U-238"], rel=1e-9)
    assert row(result, "U-234").median_atoms == pytest.approx(truth["U-234"], rel=1e-6)
    assert row(result, "Th-230").median_atoms == pytest.approx(truth["Th-230"], rel=1e-5)
    assert "Th-234" in result.pruned
    assert "Pa-234m" in result.pruned
    assert result.forward_check_ok
    assert not any(r.chain_tainted for r in result.rows)


# --- Layer 4a: unmeasured-nuclide split rule (gamma-spec pattern) ---


def test_midchain_gap_is_modelled_not_dropped_and_forward_check_closes():
    # Measured parent (Mo-99) above and measured daughter (Tc-99) below an
    # unmeasured intermediate (Tc-99m) -- the common gamma-spec case. The
    # gap must stay in the model as an assumed equilibrium state whose
    # uncertainty propagates into the daughter's interval.
    truth = {"Mo-99": 1.0e15, "Tc-99m": 1.2e14, "Tc-99": 1.0e14}
    age = 24 * 3600.0
    today = forward_atoms(truth, age)

    canon = canon_atoms({"Mo-99": today["Mo-99"], "Tc-99": today["Tc-99"]})
    result = reconstruct_t0(
        canon, age, default_rel_sigma=0.01, n_trials=20_000, seed=42
    )

    gap = row(result, "Tc-99m")
    assert gap.assumed
    assert "equilibrium" in gap.assumption_note

    daughter = row(result, "Tc-99")
    assert daughter.assumption_dependent
    assert "Tc-99m" in daughter.assumption_note
    # The deliberately loose gap prior widens the daughter's interval; it
    # must cover the truth, and the median must stay in the right ballpark.
    assert daughter.lo_atoms < truth["Tc-99"] < daughter.hi_atoms
    assert daughter.median_atoms == pytest.approx(truth["Tc-99"], rel=0.35)

    parent = row(result, "Mo-99")
    assert not parent.assumption_dependent
    assert parent.median_atoms == pytest.approx(truth["Mo-99"], rel=0.05)

    # The 4a safeguard: with the gap modelled, the forward check closes.
    assert result.forward_check_ok
    # A loose *assumed* gap must not taint an otherwise healthy chain.
    assert not any(r.chain_tainted for r in result.rows)


# --- gating and flags ---


def test_reachback_beyond_gate_is_refused_not_fabricated():
    age = 2000 * YEAR_S  # ~66 Cs-137 half-lives
    result = exact(canon_atoms({"Cs-137": 1.0e15}), age)
    r = row(result, "Cs-137")
    assert r.half_lives_back > GATE_HALF_LIVES
    assert r.conditioning == "fail"
    assert math.isnan(r.median_atoms)  # shown as not reconstructable, not a number
    assert any("Not reconstructable" in w for w in result.warnings)


def test_gated_daughter_does_not_taint_its_long_lived_parent():
    # At 100 years, Y-90 (64 h) is ~13,700 half-lives back -> gated. Refined
    # 2026-07-03: a gated *daughter* taints only its descendants, never its
    # ANCESTOR -- the parent's back-solve uses only its own measurement, so
    # Sr-90 stays clean and reliable. (Old coarse rule tainted the whole
    # chain and made progeny-rich inputs read as all-unreliable.) Unrelated
    # Cs-137 also stays clean.
    age = 100 * YEAR_S
    result = exact(canon_atoms({"Sr-90": 1.0e15, "Y-90": 1.0e11, "Cs-137": 1.0e14}), age)

    assert row(result, "Y-90").conditioning == "fail"  # correctly not traceable back
    sr = row(result, "Sr-90")
    assert sr.conditioning == "pass"
    assert not sr.chain_tainted  # a gated daughter no longer taints the parent
    assert not sr.unreliable
    cs = row(result, "Cs-137")
    assert not cs.chain_tainted
    assert not cs.unreliable


def test_hard_failure_still_taints_the_whole_chain():
    # A NEGATIVE reconstruction (internally inconsistent input, e.g. open
    # system) is different from a gate: it can cast doubt on the whole
    # connected chain, so the coarse taint is kept. Sr-90 present with Y-90
    # exactly zero at 10 days is impossible -> Y-90 goes negative -> Sr-90
    # (its parent) is tainted.
    age = 10 * 86400.0
    result = exact(canon_atoms({"Sr-90": 1.0e15, "Y-90": 0.0}), age)
    assert row(result, "Y-90").median_atoms < 0
    assert row(result, "Sr-90").chain_tainted


def test_inconsistent_input_yields_loud_negative_flag_and_failed_forward_check():
    # Sr-90 present but Y-90 exactly zero today is impossible after 10 days
    # of closed-system decay (the daughter must have grown in): the
    # back-solve must expose this as a negative t=0 amount, not absorb it.
    age = 10 * 86400.0
    result = exact(canon_atoms({"Sr-90": 1.0e15, "Y-90": 0.0}), age)

    y = row(result, "Y-90")
    assert y.median_atoms < 0
    assert y.conditioning == "fail"
    assert any("Negative reconstructed amounts" in w for w in result.warnings)
    assert not result.forward_check_ok
    assert row(result, "Sr-90").chain_tainted


def test_stable_nuclides_are_excluded_from_reconstruction():
    hl = nuclide_half_life_s("Cs-137")
    result = exact(canon_atoms({"Cs-137": 1.0e15, "Ba-137": 5.0e14}), hl)
    assert result.excluded_stable == ["Ba-137"]
    assert [r.nuclide for r in result.rows] == ["Cs-137"]


# --- Monte Carlo distributions ---


def test_mc_interval_scales_with_stated_measurement_uncertainty():
    parse = parse_paste("Cs-137, 1000, 5%")
    assert parse.ok
    canon = canonicalize(parse.entries, "Bq")
    from app.reverse import measured_atoms_from_canon

    atoms = measured_atoms_from_canon(canon)
    sigmas = sigma_atoms_from_entries(parse.entries, canon, atoms)
    assert sigmas["Cs-137"] == pytest.approx(0.05 * atoms["Cs-137"])

    hl = nuclide_half_life_s("Cs-137")
    result = reconstruct_t0(canon, hl, sigma_atoms=sigmas, n_trials=50_000, seed=7)
    r = row(result, "Cs-137")
    # Back one half-life the relative uncertainty is preserved: the 95%
    # interval half-width should be ~2 sigma = ~10% of the median.
    assert r.median_atoms == pytest.approx(2.0 * atoms["Cs-137"], rel=0.01)
    assert 0.07 < r.rel_half_width < 0.13
    assert r.conditioning == "pass"


def test_coverage_factor_divides_user_uncertainty_not_the_default():
    # A value stated at 2 sigma must yield half the 1-sigma the MC uses;
    # the seeded default (a 1-sigma assumption) must be untouched by k.
    parse = parse_paste("Cs-137, 1000, 10%\nCo-60, 500")  # Co-60 has no uncertainty
    assert parse.ok
    canon = canonicalize(parse.entries, "Bq")
    from app.reverse import measured_atoms_from_canon

    atoms = measured_atoms_from_canon(canon)
    sig_1 = sigma_atoms_from_entries(parse.entries, canon, atoms, default_rel_sigma=0.05)
    sig_2 = sigma_atoms_from_entries(
        parse.entries, canon, atoms, default_rel_sigma=0.05, coverage_k=2.0
    )
    # User-supplied (Cs-137): halved by k=2.
    assert sig_2["Cs-137"] == pytest.approx(sig_1["Cs-137"] / 2.0)
    # Defaulted (Co-60): identical -- k does not touch the seeded 1-sigma.
    assert sig_2["Co-60"] == pytest.approx(sig_1["Co-60"])


def test_coverage_factor_rejects_nonpositive_k():
    parse = parse_paste("Cs-137, 1000, 5%")
    canon = canonicalize(parse.entries, "Bq")
    from app.reverse import measured_atoms_from_canon

    atoms = measured_atoms_from_canon(canon)
    with pytest.raises(ValueError):
        sigma_atoms_from_entries(parse.entries, canon, atoms, coverage_k=0.0)


def test_absolute_and_relative_uncertainty_forms_agree():
    absolute = parse_paste("Cs-137, 1000 ± 50")
    relative = parse_paste("Cs-137, 1000, 5%")
    assert absolute.ok and relative.ok
    canon_abs = canonicalize(absolute.entries, "Bq")
    canon_rel = canonicalize(relative.entries, "Bq")
    from app.reverse import measured_atoms_from_canon

    atoms = measured_atoms_from_canon(canon_abs)
    sig_abs = sigma_atoms_from_entries(absolute.entries, canon_abs, atoms)
    sig_rel = sigma_atoms_from_entries(relative.entries, canon_rel, atoms)
    assert sig_abs["Cs-137"] == pytest.approx(sig_rel["Cs-137"], rel=1e-12)


def test_zero_sigma_reconstruction_is_deterministic():
    hl = nuclide_half_life_s("Co-60")
    result_a = exact(canon_atoms({"Co-60": 1.0e12}), hl)
    result_b = exact(canon_atoms({"Co-60": 1.0e12}), hl)
    assert row(result_a, "Co-60").median_atoms == row(result_b, "Co-60").median_atoms
    assert row(result_a, "Co-60").rel_half_width == 0.0


# --- coherence: fraction mode ---


def test_fraction_mode_round_trip_preserves_relative_composition():
    # Activity fractions carry no absolute scale; the reconstruction is
    # only meaningful as a composition. Forward-decay a known mix, feed the
    # present-day *activity fractions* back, and check the reconstructed
    # t=0 activity ratio matches the truth.
    truth = {"Cs-137": 6.0e14, "Co-60": 4.0e14}  # atoms
    age = 10 * YEAR_S
    inv = rd.Inventory(truth, "num")
    today = {str(k): float(v) for k, v in inv.decay(age, "s").activities("Bq").items()}
    total = today["Cs-137"] + today["Co-60"]
    paste = f"Cs-137, {today['Cs-137'] / total!r}\nCo-60, {today['Co-60'] / total!r}"
    parse = parse_paste(paste)
    assert parse.ok
    canon = canonicalize(parse.entries, "activity fraction")

    result = exact(canon, age)
    ratio = row(result, "Cs-137").median_atoms / row(result, "Co-60").median_atoms
    assert ratio == pytest.approx(truth["Cs-137"] / truth["Co-60"], rel=1e-6)
    assert result.forward_check_ok
