"""End-to-end input-validation checks (spec Sec 6.5) and the Layer-2
independent-implementation cross-check (reverse spec Sec 6).

Single-layer checks (does the parser flag a bad nuclide; does the
fraction-sum check fire) already live in test_parsing.py and
test_conversions.py. This module exercises the full
parse -> canonicalize -> decay pipeline together, to confirm a problem
caught at one layer actually blocks the pipeline before reaching the next,
and that a realistic multi-row paste makes it through end to end.
"""

import math
from collections import defaultdict

import pytest
import radioactivedecay as rd

from app.conversions import ValidationError, canonicalize
from app.engine import auto_time_grid_s, nuclide_half_life_s, run_time_series
from app.parsing import parse_paste
from app.reverse import direct_progeny


def test_unrecognized_nuclide_blocks_before_reaching_the_engine():
    result = parse_paste("Cs-137, 10\nNotANuclide, 5")
    assert not result.ok
    # The caller must check result.ok before calling canonicalize -- with
    # only the valid entries, the pipeline should still proceed for those.
    canon = canonicalize(result.entries, "Bq")
    assert canon.contents == {"Cs-137": 10.0}


def test_bad_fraction_sum_blocks_before_reaching_the_engine():
    result = parse_paste("Cs-137, 10\nCo-60, 10")  # sums to 20, not 1 or 100
    assert result.ok
    with pytest.raises(ValidationError):
        canonicalize(result.entries, "activity fraction")


def test_duplicate_nuclide_blocks_with_a_clear_message():
    result = parse_paste("Cs-137, 10\nCs-137, 5\nCo-60, 3")
    assert not result.ok
    assert any("Duplicate" in e.message and "Cs-137" in e.message for e in result.errors)
    # Only the unaffected line should survive for downstream use.
    assert [e.nuclide for e in result.entries] == ["Co-60"]


def test_empty_paste_does_not_crash_the_pipeline():
    result = parse_paste("   \n\n  ")
    assert result.ok
    assert result.entries == []
    # Calling canonicalize on an empty entry list should not raise for an
    # absolute unit -- it just yields an empty inventory.
    canon = canonicalize(result.entries, "Bq")
    assert canon.contents == {}


def test_realistic_multirow_paste_runs_end_to_end():
    paste = "\n".join(
        [
            "Cs-137, 3.7e9",
            "Co-60, 1.2e8",
            "Sr-90, 5.0e7",
            "Mo-99, 2.0e6",
            "Tc-99m, 1.0e5",
            "  cs134 , 4.4e6",  # casing + whitespace tolerance
            "I-131, 8.8e8",
            "Am-241, 3.1e5",
            "U-238, 9.99e3",
            "Pu-239, 7.7e4",
        ]
    )
    result = parse_paste(paste)
    assert result.ok, result.errors
    assert len(result.entries) == 10

    canon = canonicalize(result.entries, "Bq")
    times = auto_time_grid_s([e.nuclide for e in result.entries])
    series = run_time_series(canon, times)

    # every pasted nuclide, plus at least its direct progeny, should show up
    for e in result.entries:
        assert e.nuclide in series.nuclides
    assert len(series.nuclides) > len(result.entries)  # progeny grew in


def test_negative_value_never_reaches_canonicalize():
    result = parse_paste("Cs-137, -1")
    assert not result.ok
    assert result.entries == []


def test_zero_value_flows_through_to_a_harmless_zero_quantity():
    result = parse_paste("Cs-137, 0\nCo-60, 100")
    assert result.ok
    canon = canonicalize(result.entries, "Bq")
    times = auto_time_grid_s([e.nuclide for e in result.entries])
    series = run_time_series(canon, times)
    assert all(v == 0.0 for v in series.activities_bq["Cs-137"])


# --- Layer 2: independent-implementation cross-check (reverse spec Sec 6).
#
# The one thing round-trips and the forward-check overlay *cannot* catch is
# a bug in the forward model itself (both directions would share it and the
# loop would close while both are wrong). So the forward engine is checked
# once against a separately written solver: classic Bateman path enumeration
# over the branching decay graph, evaluated in 60-digit arithmetic (mpmath),
# with no matrix exponential anywhere. The two implementations share only
# the nuclear *data* (half-lives, branching fractions) -- which is the
# point: this validates the implementation, and the data is ICRP-107's
# responsibility. Two independently-authored solvers agreeing on a
# ~20-member chain to 6 significant figures are almost certainly both right;
# this is how chains too long to inspect by eye get validated.


def independent_bateman_atoms(root: str, n0: float, t_s: float) -> dict[str, float]:
    """Atoms of every nuclide at time t from n0 atoms of ``root`` at t=0.

    Direct evaluation of the general chain solution of H. Bateman,
    "Solution of a system of differential equations occurring in the
    theory of radioactive transformations", Proc. Cambridge Philos. Soc.
    15, 423-427 (1910), extended over a branching graph by summing every
    decay path. For each path root -> ... -> k with decay constants
    l1..lk and branching fractions f1..f_{k-1}:

        N_k(t) = n0 * (prod_{i<k} f_i*l_i) * sum_i e^{-l_i t} / prod_{j!=i} (l_j - l_i)

    Contributions from all paths (and all path prefixes, for intermediate
    members) are summed. Stable nuclides enter with lambda = 0.
    """
    from mpmath import mp

    mp.dps = 60
    t = mp.mpf(t_s)
    totals: dict[str, object] = defaultdict(lambda: mp.mpf(0))

    def lam(nuclide: str):
        hl = nuclide_half_life_s(nuclide)
        return mp.mpf(0) if math.isinf(hl) else mp.log(2) / mp.mpf(hl)

    def visit(nuclide: str, lams: list, coeff) -> None:
        lams = lams + [lam(nuclide)]
        term = mp.mpf(0)
        for i, li in enumerate(lams):
            denom = mp.mpf(1)
            for j, lj in enumerate(lams):
                if j != i:
                    denom *= lj - li
            term += mp.exp(-li * t) / denom
        totals[nuclide] += coeff * term
        for child, bf in direct_progeny(nuclide):
            visit(child, lams, coeff * mp.mpf(bf) * lams[-1])

    visit(root, [], mp.mpf(n0))
    return {k: float(v) for k, v in totals.items()}


def library_atoms(root: str, n0: float, t_s: float) -> dict[str, float]:
    decayed = rd.Inventory({root: n0}, "num").decay(t_s, "s")
    return {str(k): float(v) for k, v in decayed.numbers().items()}


def assert_implementations_agree(root: str, t_s: float, sig_figs_rel: float = 1e-6):
    n0 = 1.0e24
    independent = independent_bateman_atoms(root, n0, t_s)
    library = library_atoms(root, n0, t_s)
    # Members below float64's resolving power in the library's own solve
    # (~1e-14 of the total) are excluded from the comparison; everything
    # else must agree to ~6 significant figures.
    threshold = 1e-10 * n0
    compared = 0
    for nuclide, expected in independent.items():
        if expected < threshold:
            continue
        assert library.get(nuclide, 0.0) == pytest.approx(expected, rel=sig_figs_rel), nuclide
        compared += 1
    assert compared >= 2, "cross-check compared too few nuclides to mean anything"


def test_layer2_cross_check_mo99_chain_one_day():
    # Branching two ways at Mo-99 (Tc-99m / Tc-99) plus a metastable state.
    assert_implementations_agree("Mo-99", 24 * 3600.0)


def test_layer2_cross_check_u238_chain_100k_years():
    # The full ~20-member U-238 series with multiple branch points, deep
    # into ingrowth: the long-chain case that cannot be judged by eye.
    assert_implementations_agree("U-238", 1.0e5 * 86400.0 * 365.25)


def test_layer2_cross_check_u238_chain_4_5_billion_years():
    # One full U-238 half-life: every member at or near secular equilibrium.
    assert_implementations_agree("U-238", 4.468e9 * 86400.0 * 365.25)


def test_layer2_cross_check_u235_chain_50_years():
    # The actinium series (used by the examples/ synthetic datasets),
    # including the Ac-227 -> Th-227 / Fr-223 branch point.
    assert_implementations_agree("U-235", 50.0 * 86400.0 * 365.25)


def test_layer2_cross_check_th232_chain_100_years():
    # Added in revalidation (2026-07-03): the thorium series was the one
    # major natural chain the cross-check had not covered. Includes the
    # Bi-212 alpha/beta branch (~36/64) feeding Tl-208 and Po-212.
    assert_implementations_agree("Th-232", 100.0 * 86400.0 * 365.25)


def test_layer2_cross_check_pu241_chain_100_years():
    # Added in revalidation (2026-07-03): the chain behind the round-trip
    # dating cases, cross-checked independently -- including Pu-241's own
    # tiny alpha branch to U-237 (~2.5e-5) alongside the beta to Am-241.
    assert_implementations_agree("Pu-241", 100.0 * 86400.0 * 365.25)
