"""Deep revalidation sweep — checks too slow or too statistical for the
per-commit pytest suite. Run manually; findings go in docs/revalidation-*.md.

    .venv/Scripts/python.exe validation/deep_revalidation.py

Four blocks, each producing evidence that did not exist when the code and
its tests were originally written together:

1. MC interval calibration, Mode B — the suite checks that single noisy
   cases land inside their 95% intervals; this measures the *coverage
   rate* over many independent noisy repetitions. A miscalibrated
   pipeline (intervals systematically too wide or too narrow) passes
   single-case tests and fails here.
2. MC interval calibration, Mode A — same, for the solved age.
3. Cross-mode consistency — Mode B's reconstruction fed to Mode A as the
   known t=0 must return the original age. The two solvers were built and
   tested separately; this chains them.
4. Randomized round-trip fuzz — fresh seeded random compositions/ages,
   none of which appear in the test suite, through the Mode B round trip.

Exit code 0 = all blocks passed their acceptance criteria.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import radioactivedecay as rd

from app.age_solve import solve_age
from app.conversions import canonicalize
from app.engine import nuclide_half_life_s
from app.parsing import parse_paste
from app.reverse import reconstruct_t0

YEAR_S = 86400.0 * 365.25
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    tag = "ok  " if condition else "FAIL"
    print(f"  [{tag}] {message}")
    if not condition:
        FAILURES.append(message)


def canon_atoms(contents: dict[str, float]):
    paste = "\n".join(f"{n}, {float(v)!r}" for n, v in contents.items())
    parsed = parse_paste(paste)
    assert parsed.ok, parsed.errors
    return canonicalize(parsed.entries, "atoms")


def forward_atoms(t0: dict[str, float], age_s: float) -> dict[str, float]:
    decayed = rd.Inventory(t0, "num").decay(age_s, "s")
    return {str(k): float(v) for k, v in decayed.numbers().items()}


# --- 1. Mode B interval calibration -----------------------------------------


def calibrate_mode_b(reps: int = 50, rel_sigma: float = 0.05) -> None:
    print(f"\n[1] Mode B 95%-interval calibration ({reps} noisy reps, {rel_sigma:.0%} noise)")
    truth = {"Cs-137": 1.0e15, "Sr-90": 4.0e14, "Co-60": 1.5e14}
    age = 10.0 * YEAR_S
    today = forward_atoms(truth, age)

    hits = total = 0
    for rep in range(reps):
        rng = np.random.default_rng(1000 + rep)
        noisy = {n: today[n] * (1.0 + rel_sigma * rng.standard_normal()) for n in truth}
        result = reconstruct_t0(
            canon_atoms(noisy), age, default_rel_sigma=rel_sigma, n_trials=20_000, seed=rep
        )
        rows = {r.nuclide: r for r in result.rows}
        for n in truth:
            total += 1
            if rows[n].lo_atoms <= truth[n] <= rows[n].hi_atoms:
                hits += 1

    coverage = hits / total
    print(f"  coverage: {hits}/{total} = {coverage:.1%} (nominal 95%)")
    # 3-sigma binomial band around 0.95 for n=total.
    lo_accept = 0.95 - 3 * math.sqrt(0.95 * 0.05 / total)
    check(coverage >= lo_accept, f"Mode B coverage {coverage:.1%} >= {lo_accept:.1%}")
    check(coverage <= 1.0, "coverage sane")


# --- 2. Mode A interval calibration -----------------------------------------


def calibrate_mode_a(reps: int = 25, rel_sigma: float = 0.03) -> None:
    print(f"\n[2] Mode A 95%-age-interval calibration ({reps} noisy reps, {rel_sigma:.0%} noise)")
    truth_age = 20.0 * YEAR_S
    t0 = {"Pu-241": 1.0e15}
    today = forward_atoms(t0, truth_age)

    hits = 0
    for rep in range(reps):
        rng = np.random.default_rng(2000 + rep)
        noisy = {
            n: today[n] * (1.0 + rel_sigma * rng.standard_normal())
            for n in ("Pu-241", "Am-241")
        }
        result = solve_age(
            canon_atoms(t0),
            canon_atoms(noisy),
            default_rel_sigma=rel_sigma,
            n_trials=4_000,
            seed=rep,
        )
        if result.age_s_lo <= truth_age <= result.age_s_hi:
            hits += 1

    coverage = hits / reps
    print(f"  coverage: {hits}/{reps} = {coverage:.1%} (nominal 95%)")
    lo_accept = 0.95 - 3 * math.sqrt(0.95 * 0.05 / reps)
    check(coverage >= lo_accept, f"Mode A coverage {coverage:.1%} >= {lo_accept:.1%}")


# --- 3. Cross-mode consistency ----------------------------------------------


def cross_mode() -> None:
    print("\n[3] Cross-mode: forward -> Mode B reconstruction -> Mode A age solve")
    truth_age = 25.0 * YEAR_S
    t0 = {"Pu-241": 1.0e15, "Am-241": 2.0e13}
    today = forward_atoms(t0, truth_age)
    measured = {n: today[n] for n in t0}

    rev = reconstruct_t0(canon_atoms(measured), truth_age, default_rel_sigma=0.0)
    reconstructed = {r.nuclide: r.median_atoms for r in rev.rows}
    check(rev.forward_check_ok, "Mode B forward check closed")

    age = solve_age(canon_atoms(reconstructed), canon_atoms(measured), default_rel_sigma=0.0)
    rel = abs(age.age_s - truth_age) / truth_age
    print(f"  B-reconstructed t0 -> A solved age: {age.age_s / YEAR_S:.6f} y (truth 25) rel err {rel:.2e}")
    check(rel < 1e-4, f"cross-mode age recovered (rel err {rel:.2e} < 1e-4)")
    check(age.resolvable, "cross-mode age resolvable")


# --- 4. Randomized round-trip fuzz ------------------------------------------


def fuzz_round_trips(cases: int = 25) -> None:
    print(f"\n[4] Randomized Mode B round-trips ({cases} fresh seeded cases)")
    pool = [
        "Cs-137", "Sr-90", "Co-60", "Am-241", "Pu-241",
        "U-238", "Ra-226", "Eu-154", "Ir-192", "H-3",
    ]
    rng = np.random.default_rng(20260703)
    worst_rel = 0.0
    worst_case = ""
    for case in range(cases):
        k = int(rng.integers(1, 4))
        chosen = list(rng.choice(pool, size=k, replace=False))
        truth = {n: float(10 ** rng.uniform(12, 20)) for n in chosen}
        min_hl = min(nuclide_half_life_s(n) for n in chosen)
        age = float(10 ** rng.uniform(math.log10(0.02), math.log10(3.0))) * min_hl

        today = forward_atoms(truth, age)
        result = reconstruct_t0(
            canon_atoms({n: today[n] for n in chosen}), age, default_rel_sigma=0.0
        )
        rows = {r.nuclide: r for r in result.rows}

        scale = max(truth.values())
        ok = result.forward_check_ok
        for n in chosen:
            err = abs(rows[n].median_atoms - truth[n]) / (truth[n] + 1e-9 * scale)
            if err > worst_rel:
                worst_rel = err
                worst_case = f"case {case}: {chosen} age {age:.3g}s nuclide {n}"
            ok = ok and err < 1e-5 and rows[n].median_atoms >= 0
        if not ok:
            check(False, f"case {case} failed: {chosen}, age {age:.3g} s")

    print(f"  worst relative recovery error: {worst_rel:.2e} ({worst_case})")
    check(worst_rel < 1e-5, f"all {cases} random round-trips recovered to < 1e-5")


if __name__ == "__main__":
    print("Deep revalidation sweep -- see docs/revalidation report for context")
    calibrate_mode_b()
    calibrate_mode_a()
    cross_mode()
    fuzz_round_trips()
    print(f"\n{'=' * 60}")
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("RESULT: all deep revalidation blocks passed")
