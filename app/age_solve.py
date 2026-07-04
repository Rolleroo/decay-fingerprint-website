"""Mode A: known t=0 composition + measured present-day composition ->
solve for the age. Contract: docs/mode-a-addendum.md.

A weighted least-squares fit of one scalar (the age) against the trusted
forward engine -- no back-solve, no transfer matrix. chi2(t) is evaluated
on a log-spaced time grid (each point one forward decay with the library),
the optimum refined with true engine evaluations, and uncertainty comes
from Monte Carlo re-minimization of every trial over the same grid.

Gates follow the DQPB pattern (analytical resolvability before MC): a flat
chi2 across the whole window means no measured nuclide changed
meaningfully over any admissible age -- refused, not fitted. Ambiguity
(ingrowth curves are not monotonic, so two ages can fit equally well) is
detected from the local minima of the chi2 curve and reported, never
silently collapsed to one answer.

References: see app/reverse.py -- the same engine (Malins & Lemoine 2022)
and method lineage (Pollard et al. 2023, DQPB) apply.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import radioactivedecay as rd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2 as _chi2_dist

from app.conversions import CanonResult, ValidationError
from app.engine import is_stable, nuclide_half_life_s
from app.parsing import ParsedEntry
from app.reverse import descendants, measured_atoms_from_canon, sigma_atoms_from_entries

DEFAULT_TRIALS = 20_000  # each trial re-minimizes the age, hence lower than Mode B's
GRID_POINTS = 300
WINDOW_LOW_S = 1.0
WINDOW_HALF_LIVES = 40.0  # upper edge: beyond this nothing measurable is left to fit

# All local chi2 minima within this of the global minimum are reported as
# candidate ages (delta-chi2 = 4 ~ a 2-sigma-equivalent acceptance band).
AMBIGUITY_DELTA_CHI2 = 4.0

# Interval gate: a 95% age interval spanning more than this factor (or
# pinned to a window edge) is reported as not resolvable.
INTERVAL_SPAN_LIMIT = 1e4

# chi2/dof above this at the optimum flags the inputs as inconsistent with
# closed-system decay from the stated t=0.
CONSISTENCY_CHI2_PER_DOF = 3.0

# Relative floor applied to sigma = 0 ("exact") measurements so they act
# as near-hard constraints without dividing by zero. Not smaller: the
# optimizer refines ages to ~1e-10 in log-time, and the ambiguity band
# (delta-chi2 = 4) must not reject a genuine second solution just because
# its refinement residual, measured in floor units, exceeds 4.
SIGMA_FLOOR_REL = 1e-6

YEAR_S = 86400.0 * 365.25


@dataclass(frozen=True)
class ResidualRow:
    """One measured nuclide's forward-check line at the solved age."""

    nuclide: str
    measured_atoms: float
    modeled_atoms: float  # known t=0 decayed forward by the solved age
    sigma_atoms: float
    mismatch_rel: float  # (modeled - measured) / measured; inf if measured == 0
    mismatch_sigma: float  # (modeled - measured) / sigma
    sensitivity: float  # |f(2t) - f(t/2)| / sigma: does it constrain the age?
    informative: bool


@dataclass(frozen=True)
class AgeResult:
    age_s: float  # central weighted-least-squares age
    age_s_median: float
    age_s_lo: float  # 2.5th percentile
    age_s_hi: float  # 97.5th percentile
    resolvable: bool
    chi2_per_dof: float
    ambiguous_ages_s: list[float]  # all candidate ages when the fit is ambiguous
    residuals: list[ResidualRow]
    excluded_unproducible: list[str]
    warnings: list[str] = field(default_factory=list)
    n_trials: int = 0


def age_readable(age_s: float) -> str:
    """Human-readable age string, unit chosen by magnitude."""
    if not math.isfinite(age_s):
        return "—"
    if age_s < 120.0:
        return f"{age_s:.4g} seconds"
    if age_s < 120.0 * 60.0:
        return f"{age_s / 60.0:.4g} minutes"
    if age_s < 48.0 * 3600.0:
        return f"{age_s / 3600.0:.4g} hours"
    if age_s < 120.0 * 86400.0:
        return f"{age_s / 86400.0:.4g} days"
    years = age_s / YEAR_S
    return f"{years:.4g} years" if years < 1e5 else f"{years:.4e} years"


def _guard_canon(canon: CanonResult, which: str) -> None:
    if canon.kind.startswith("fraction_"):
        raise ValidationError(
            f"The {which} composition uses a fraction/percent unit. Finding an age "
            f"needs absolute amounts (activity, mass, or atoms) — please switch the unit."
        )
    if canon.kind in ("activity", "specific_activity"):
        stable = sorted(n for n in canon.contents if is_stable(n))
        if stable:
            raise ValidationError(
                f"Stable nuclide(s) {', '.join(stable)} in the {which} composition "
                f"cannot be expressed in an activity unit -- use mass, mol, or atoms "
                f"(a stable daughter is a useful chronometer, so it is refused, not dropped)."
            )
    if not canon.contents:
        raise ValidationError(f"The {which} composition is empty.")


def solve_age(
    canon_t0: CanonResult,
    canon_today: CanonResult,
    sigma_atoms_today: dict[str, float] | None = None,
    sigma_atoms_t0: dict[str, float] | None = None,
    default_rel_sigma: float = 0.05,
    n_trials: int = DEFAULT_TRIALS,
    seed: int | None = None,
) -> AgeResult:
    """Solve for the age given the known t=0 and measured today compositions.

    Sigmas are 1-sigma absolute uncertainties in atoms (see
    ``sigma_atoms_from_entries``); measured nuclides missing from
    ``sigma_atoms_today`` get ``default_rel_sigma`` times their value, t=0
    nuclides missing from ``sigma_atoms_t0`` are treated as exact.
    """
    _guard_canon(canon_t0, "t=0")
    _guard_canon(canon_today, "present-day")

    warnings: list[str] = []
    atoms_t0 = measured_atoms_from_canon(canon_t0)
    atoms_today = measured_atoms_from_canon(canon_today)

    sigma_atoms_today = dict(sigma_atoms_today or {})
    for n, v in atoms_today.items():
        sigma_atoms_today.setdefault(n, default_rel_sigma * v)
    sigma_atoms_t0 = dict(sigma_atoms_t0 or {})

    # Producibility: a measured nuclide outside the t=0 decay closure can
    # never fit and would poison the age -- excluded loudly.
    closure = set(atoms_t0)
    for n in atoms_t0:
        closure |= descendants(n)
    excluded = sorted(n for n in atoms_today if n not in closure)
    if excluded:
        warnings.append(
            f"Ignored (these can't come from the original composition you gave, so "
            f"they can't help date it): {', '.join(excluded)}. If they're really in "
            f"the sample, the original composition you entered is incomplete."
        )
    fit_nuclides = sorted(n for n in atoms_today if n in closure)
    if not fit_nuclides:
        raise ValidationError(
            "None of the measured nuclides could have come from the original "
            "composition you gave, so there's nothing to work an age out from."
        )

    finite_hls = [
        hl for n in closure if not math.isinf(hl := nuclide_half_life_s(n))
    ]
    if not finite_hls:
        raise ValidationError(
            "The original composition contains nothing radioactive, so there's no "
            "decay clock to date the sample by."
        )
    t_lo = WINDOW_LOW_S
    t_hi = max(WINDOW_HALF_LIVES * max(finite_hls), t_lo * 1e6)

    # --- forward model on a log time grid (one engine call per point) ---
    base_inv = rd.Inventory(
        {n: v for n, v in atoms_t0.items() if v > 0} or dict(atoms_t0), "num"
    )
    grid = np.geomspace(t_lo, t_hi, GRID_POINTS)
    log_grid = np.log(grid)

    def forward_atoms(t_s: float) -> dict[str, float]:
        decayed = base_inv.decay(float(t_s), "s")
        return {str(k): float(v) for k, v in decayed.numbers().items()}

    F = np.zeros((GRID_POINTS, len(fit_nuclides)))
    for k, t in enumerate(grid):
        numbers = forward_atoms(t)
        for i, n in enumerate(fit_nuclides):
            F[k, i] = numbers.get(n, 0.0)

    m = np.array([atoms_today[n] for n in fit_nuclides])
    sigma = np.array([sigma_atoms_today[n] for n in fit_nuclides])
    # The floor stands in for sigma ONLY where the input is exact (sigma =
    # 0), and is keyed per-nuclide to the measured value itself -- never to
    # the forward model's largest value, which can exceed a trace-level
    # measurement by many orders of magnitude and would silently replace a
    # real stated uncertainty (revalidation finding, 2026-07-03).
    measured_scale = max(float(m.max(initial=0.0)), 1.0)
    floor = SIGMA_FLOOR_REL * np.where(m > 0, m, measured_scale)
    sigma_eff = np.where(sigma > 0, sigma, floor)
    deterministic = bool(np.all(sigma <= 0))

    def chi2_true(t_s: float) -> float:
        numbers = forward_atoms(t_s)
        return float(
            np.sum(
                ((m - np.array([numbers.get(n, 0.0) for n in fit_nuclides])) / sigma_eff)
                ** 2
            )
        )

    chi2_grid = np.sum(((m[None, :] - F) / sigma_eff[None, :]) ** 2, axis=1)

    # --- gate 1: analytical resolvability, before any MC (DQPB pattern) ---
    if float(chi2_grid.max() - chi2_grid.min()) < 1.0:
        warnings.append(
            "Can't determine an age: none of these nuclides change enough between "
            "the original and today's composition to measure a decay time. Check "
            "that today's values really differ from the original — if they're the "
            "same (or nearly), there's no decay to date."
        )
        k = int(np.argmin(chi2_grid))
        return AgeResult(
            age_s=float(grid[k]),
            age_s_median=math.nan,
            age_s_lo=math.nan,
            age_s_hi=math.nan,
            resolvable=False,
            chi2_per_dof=float(chi2_grid[k]) / max(len(fit_nuclides) - 1, 1),
            ambiguous_ages_s=[],
            residuals=_residuals(fit_nuclides, m, sigma_eff, forward_atoms(grid[k]), math.nan),
            excluded_unproducible=excluded,
            warnings=warnings + [_CLOSED_SYSTEM_NOTE],
        )

    def refine(k: int) -> float:
        """True-engine refinement of a grid minimum (no grid error in the age)."""
        lo = log_grid[max(k - 1, 0)]
        hi = log_grid[min(k + 1, GRID_POINTS - 1)]
        res = minimize_scalar(
            lambda logt: chi2_true(math.exp(logt)),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-10},
        )
        return float(math.exp(res.x))

    def optimize(c_grid: np.ndarray) -> tuple[float, list[float]]:
        """Refine every grid-level local minimum with true engine calls and
        judge candidates on their *refined* chi2 -- grid values adjacent to
        a sharp optimum can be astronomically large when sigmas are tiny,
        so an on-grid delta-chi2 band would miss genuine second solutions.

        Returns (central age, all candidate ages within the acceptance band).
        """
        minima = [
            k
            for k in range(GRID_POINTS)
            if c_grid[k] <= (c_grid[k - 1] if k > 0 else math.inf)
            and c_grid[k] <= (c_grid[k + 1] if k < GRID_POINTS - 1 else math.inf)
        ]
        minima = sorted(minima, key=lambda k: c_grid[k])[:8]  # cap engine work
        refined = [refine(k) for k in minima]
        scores = [chi2_true(t) for t in refined]
        best = min(scores)
        candidates = sorted(
            t for t, s in zip(refined, scores) if s < best + AMBIGUITY_DELTA_CHI2
        )
        distinct: list[float] = []
        for t in candidates:
            if not distinct or t > distinct[-1] * 1.05:
                distinct.append(t)
        central = refined[scores.index(best)]
        return central, distinct

    # --- t=0 uncertainty (v1): folded into sigma_eff at the central age ---
    t_central, distinct = optimize(chi2_grid)
    if any(s > 0 for s in sigma_atoms_t0.values()):
        sigma_model_sq = np.zeros(len(fit_nuclides))
        for j, s_j in sigma_atoms_t0.items():
            if s_j <= 0:
                continue
            unit_numbers = {
                str(k): float(v)
                for k, v in rd.Inventory({j: 1.0}, "num").decay(t_central, "s").numbers().items()
            }
            for i, n in enumerate(fit_nuclides):
                sigma_model_sq[i] += (unit_numbers.get(n, 0.0) * s_j) ** 2
        sigma_eff = np.sqrt(sigma_eff**2 + sigma_model_sq)
        chi2_grid = np.sum(((m[None, :] - F) / sigma_eff[None, :]) ** 2, axis=1)
        t_central, distinct = optimize(chi2_grid)
        warnings.append(
            "You also gave uncertainties on the original composition. These are "
            "included in the age uncertainty, but approximately — treat the interval "
            "as a good guide rather than an exact figure."
        )

    # --- gate 3: ambiguity -- multiple refined optima in the acceptance band ---
    ambiguous = distinct if len(distinct) > 1 else []
    if ambiguous:
        warnings.append(
            "Ambiguous age: " + ", ".join(age_readable(t) for t in ambiguous)
            + " fit the measurements comparably (ingrowth curves are not "
            "monotonic). The interval below spans the candidates; more measured "
            "nuclides would break the tie."
        )

    # --- Monte Carlo: re-minimize every trial over the precomputed grid ---
    if deterministic:
        median = lo_age = hi_age = t_central
        effective_trials = 0
    else:
        rng = np.random.default_rng(seed)
        ages = np.empty(n_trials)
        chunk = 2_000
        for start in range(0, n_trials, chunk):
            size = min(chunk, n_trials - start)
            samples = rng.normal(m[:, None], sigma_eff[:, None], size=(len(fit_nuclides), size))
            np.clip(samples, 0.0, None, out=samples)
            c = np.zeros((GRID_POINTS, size))
            for i in range(len(fit_nuclides)):
                c += ((samples[i][None, :] - F[:, i, None]) / sigma_eff[i]) ** 2
            idx = np.argmin(c, axis=0)
            # Parabolic refinement in log-t around each trial's grid minimum.
            k0 = np.clip(idx, 1, GRID_POINTS - 2)
            cols = np.arange(size)
            c_l, c_m, c_r = c[k0 - 1, cols], c[k0, cols], c[k0 + 1, cols]
            denom = c_l - 2 * c_m + c_r
            shift = np.where(np.abs(denom) > 0, 0.5 * (c_l - c_r) / np.where(denom == 0, 1, denom), 0.0)
            shift = np.clip(shift, -1.0, 1.0)
            step = log_grid[1] - log_grid[0]
            ages[start : start + size] = np.exp(log_grid[k0] + shift * step)
        lo_age, median, hi_age = (float(v) for v in np.percentile(ages, [2.5, 50.0, 97.5]))
        effective_trials = n_trials

    # --- gate 2: interval sanity ---
    resolvable = True
    if not deterministic:
        if hi_age / max(lo_age, 1e-300) > INTERVAL_SPAN_LIMIT:
            resolvable = False
            warnings.append(
                "Can't determine an age: the result is far too uncertain to be "
                "useful — the ages that fit these numbers span an enormous range."
            )
        if lo_age <= t_lo * 1.5 or hi_age >= t_hi / 1.5:
            resolvable = False
            warnings.append(
                "Can't determine an age: the best fit sits at the very edge of the "
                "time range this tool searches. The sample is likely far younger or "
                "older than these nuclides can reveal, or the two compositions are "
                "too similar to show measurable decay."
            )

    # --- consistency + residual table (this is the forward-check overlay) ---
    modeled = forward_atoms(t_central)
    chi2_final = chi2_true(t_central)
    dof = max(len(fit_nuclides) - 1, 1)
    chi2_per_dof = chi2_final / dof
    if not deterministic and chi2_per_dof > CONSISTENCY_CHI2_PER_DOF:
        warnings.append(
            f"Poor fit: the original and today's compositions don't agree on a "
            f"single age (goodness-of-fit chi²/dof = {chi2_per_dof:.1f}, where ~1 is "
            f"a good fit). They may not be from the same sample, the sample may have "
            f"gained or lost material, or the stated uncertainties are too small. "
            f"The table below shows which nuclide disagrees most."
        )

    residuals = _residuals(
        fit_nuclides, m, sigma_eff, modeled, t_central, forward_atoms=forward_atoms
    )
    if not any(r.informative for r in residuals):
        warnings.append(
            "No single nuclide pins the age down on its own at its stated "
            "uncertainty — the answer relies on all of them together."
        )

    warnings.append(_CLOSED_SYSTEM_NOTE)
    return AgeResult(
        age_s=t_central,
        age_s_median=median,
        age_s_lo=lo_age,
        age_s_hi=hi_age,
        resolvable=resolvable,
        chi2_per_dof=chi2_per_dof,
        ambiguous_ages_s=ambiguous,
        residuals=residuals,
        excluded_unproducible=excluded,
        warnings=warnings,
        n_trials=effective_trials,
    )


_CLOSED_SYSTEM_NOTE = (
    "This result assumes a closed system — nothing was added to or removed from "
    "the sample except by radioactive decay — and that the original composition "
    "you gave is complete for every chain measured."
)


def _residuals(
    fit_nuclides: list[str],
    m: np.ndarray,
    sigma_eff: np.ndarray,
    modeled: dict[str, float],
    t_central: float,
    forward_atoms=None,
) -> list[ResidualRow]:
    f_hi = forward_atoms(2.0 * t_central) if forward_atoms and math.isfinite(t_central) else {}
    f_lo = forward_atoms(0.5 * t_central) if forward_atoms and math.isfinite(t_central) else {}
    rows = []
    for i, n in enumerate(fit_nuclides):
        mod = modeled.get(n, 0.0)
        rel = (mod - m[i]) / m[i] if m[i] > 0 else (math.inf if mod > 0 else 0.0)
        sens = (
            abs(f_hi.get(n, 0.0) - f_lo.get(n, 0.0)) / sigma_eff[i] if f_hi else math.nan
        )
        rows.append(
            ResidualRow(
                nuclide=n,
                measured_atoms=float(m[i]),
                modeled_atoms=mod,
                sigma_atoms=float(sigma_eff[i]),
                mismatch_rel=rel,
                mismatch_sigma=(mod - float(m[i])) / float(sigma_eff[i]),
                sensitivity=float(sens),
                informative=bool(sens > 1.0) if not math.isnan(sens) else False,
            )
        )
    return rows


def solve_age_from_entries(
    entries_t0: list[ParsedEntry],
    canon_t0: CanonResult,
    entries_today: list[ParsedEntry],
    canon_today: CanonResult,
    default_rel_sigma: float = 0.05,
    n_trials: int = DEFAULT_TRIALS,
    seed: int | None = None,
    coverage_k: float = 1.0,
) -> AgeResult:
    """UI-facing convenience: derive sigmas from the parsed lines. t=0
    lines default to exact (0%) -- a known reference, not a measurement --
    while present-day lines default to ``default_rel_sigma``. ``coverage_k``
    is the coverage factor of the pasted uncertainties on both sides."""
    _guard_canon(canon_t0, "t=0")
    _guard_canon(canon_today, "present-day")
    atoms_t0 = measured_atoms_from_canon(canon_t0)
    atoms_today = measured_atoms_from_canon(canon_today)
    sig_t0 = sigma_atoms_from_entries(
        entries_t0, canon_t0, atoms_t0, default_rel_sigma=0.0, coverage_k=coverage_k
    )
    sig_today = sigma_atoms_from_entries(
        entries_today, canon_today, atoms_today,
        default_rel_sigma=default_rel_sigma, coverage_k=coverage_k,
    )
    return solve_age(
        canon_t0,
        canon_today,
        sigma_atoms_today=sig_today,
        sigma_atoms_t0=sig_t0,
        default_rel_sigma=default_rel_sigma,
        n_trials=n_trials,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Compatibility check: assumed original composition + assumed age, is today's
# measurement compatible? Both age and t=0 are fixed, so this is a pure
# goodness-of-fit test with zero free parameters (dof = number of measured
# nuclides) -- one forward decay, no solve. It reuses the same forward engine,
# producibility closure, and residual machinery as the age solve. A second,
# scale-free score answers the *ratio/pattern* question (is the isotopic
# pattern right, whatever the overall amount?) with the overall level fitted
# out in closed form (dof = n - 1).
# ---------------------------------------------------------------------------

# p-value thresholds for the plain-language verdict. p is the probability of a
# chi-squared this large from measurement scatter alone: high p -> the data are
# consistent with the hypothesis; very low p -> the hypothesis is rejected.
COMPAT_P_COMPATIBLE = 0.05
COMPAT_P_INCOMPATIBLE = 1e-3


@dataclass(frozen=True)
class CompatResult:
    age_s: float
    n_fit: int
    # absolute-amount hypothesis (overall scale = 1, zero free parameters)
    chi2: float
    dof: int
    chi2_per_dof: float
    p_value: float
    verdict: str  # 'compatible' | 'marginal' | 'incompatible'
    # ratio / pattern hypothesis (overall scale fitted out, dof = n - 1)
    ratio_testable: bool
    scale: float  # best-fit overall multiplier on the assumed amounts
    chi2_scaled: float
    dof_scaled: int
    chi2_per_dof_scaled: float
    p_value_scaled: float
    verdict_scaled: str
    residuals: list[ResidualRow]
    excluded_unproducible: list[str]
    warnings: list[str] = field(default_factory=list)


def _compat_verdict(p: float) -> str:
    if not math.isfinite(p):
        return "n/a"
    if p >= COMPAT_P_COMPATIBLE:
        return "compatible"
    if p >= COMPAT_P_INCOMPATIBLE:
        return "marginal"
    return "incompatible"


def check_compatibility(
    canon_t0: CanonResult,
    canon_today: CanonResult,
    age_s: float,
    sigma_atoms_today: dict[str, float] | None = None,
    default_rel_sigma: float = 0.05,
) -> CompatResult:
    """Test whether today's measurement is compatible with the assumed
    original composition aged by ``age_s``. Decays the original forward once
    and scores the measurement against it with zero free parameters (the
    absolute-amount hypothesis), plus a scale-free score (the ratio/pattern
    hypothesis). Sigmas are 1-sigma absolute uncertainties in atoms; measured
    nuclides missing from ``sigma_atoms_today`` get ``default_rel_sigma`` times
    their value.
    """
    _guard_canon(canon_t0, "original")
    _guard_canon(canon_today, "present-day")
    if not (math.isfinite(age_s) and age_s > 0):
        raise ValidationError("The assumed age must be a positive amount of time.")

    warnings: list[str] = []
    atoms_t0 = measured_atoms_from_canon(canon_t0)
    atoms_today = measured_atoms_from_canon(canon_today)

    sigma_atoms_today = dict(sigma_atoms_today or {})
    for n, v in atoms_today.items():
        sigma_atoms_today.setdefault(n, default_rel_sigma * v)

    # Producibility: a measured nuclide outside the assumed original's decay
    # closure cannot arise from it at any age -- unlike the age solve (where it
    # is merely uninformative), here it is direct evidence *against* the
    # hypothesis, so it is flagged loudly as well as left out of the score.
    closure = set(atoms_t0)
    for n in atoms_t0:
        closure |= descendants(n)
    excluded = sorted(n for n in atoms_today if n not in closure)
    if excluded:
        warnings.append(
            f"Measured nuclide(s) {', '.join(excluded)} cannot arise from the assumed "
            f"original composition at any age. On their own they make the sample "
            f"incompatible with this origin — unless the original composition you "
            f"entered is incomplete. They are left out of the score below."
        )
    fit_nuclides = sorted(n for n in atoms_today if n in closure)
    if not fit_nuclides:
        raise ValidationError(
            "None of the measured nuclides could have come from the assumed original "
            "composition, so there is nothing to compare — the sample is incompatible "
            "with this origin (or the original composition is incomplete)."
        )

    base_inv = rd.Inventory(
        {n: v for n, v in atoms_t0.items() if v > 0} or dict(atoms_t0), "num"
    )
    decayed = base_inv.decay(float(age_s), "s")
    modeled = {str(k): float(v) for k, v in decayed.numbers().items()}

    m = np.array([atoms_today[n] for n in fit_nuclides])
    f = np.array([modeled.get(n, 0.0) for n in fit_nuclides])
    sigma = np.array([sigma_atoms_today[n] for n in fit_nuclides])
    # Same per-nuclide floor as the age solve: stands in only where a measured
    # value was given as exact (sigma = 0), keyed to the value itself.
    measured_scale = max(float(m.max(initial=0.0)), 1.0)
    floor = SIGMA_FLOOR_REL * np.where(m > 0, m, measured_scale)
    sigma_eff = np.where(sigma > 0, sigma, floor)

    # --- absolute-amount hypothesis: scale fixed at 1, zero free parameters ---
    chi2 = float(np.sum(((m - f) / sigma_eff) ** 2))
    dof = len(fit_nuclides)
    p_value = float(_chi2_dist.sf(chi2, dof))
    verdict = _compat_verdict(p_value)

    # --- ratio / pattern hypothesis: fit the one overall scale in closed form ---
    weight = float(np.sum((f / sigma_eff) ** 2))
    ratio_testable = len(fit_nuclides) >= 2 and weight > 0.0
    if ratio_testable:
        scale = float(np.sum(m * f / sigma_eff**2) / weight)
        chi2_scaled = float(np.sum(((m - scale * f) / sigma_eff) ** 2))
        dof_scaled = len(fit_nuclides) - 1
        p_value_scaled = float(_chi2_dist.sf(chi2_scaled, dof_scaled))
        verdict_scaled = _compat_verdict(p_value_scaled)
    else:
        scale = math.nan
        chi2_scaled = math.nan
        dof_scaled = 0
        p_value_scaled = math.nan
        verdict_scaled = "n/a"

    residuals = _residuals(fit_nuclides, m, sigma_eff, modeled, math.nan)
    warnings.append(_CLOSED_SYSTEM_NOTE)

    return CompatResult(
        age_s=float(age_s),
        n_fit=len(fit_nuclides),
        chi2=chi2,
        dof=dof,
        chi2_per_dof=chi2 / dof,
        p_value=p_value,
        verdict=verdict,
        ratio_testable=ratio_testable,
        scale=scale,
        chi2_scaled=chi2_scaled,
        dof_scaled=dof_scaled,
        chi2_per_dof_scaled=(chi2_scaled / dof_scaled) if dof_scaled else math.nan,
        p_value_scaled=p_value_scaled,
        verdict_scaled=verdict_scaled,
        residuals=residuals,
        excluded_unproducible=excluded,
        warnings=warnings,
    )


def check_compatibility_from_entries(
    canon_t0: CanonResult,
    entries_today: list[ParsedEntry],
    canon_today: CanonResult,
    age_s: float,
    default_rel_sigma: float = 0.05,
    coverage_k: float = 1.0,
) -> CompatResult:
    """UI-facing convenience: derive today's sigmas from the parsed lines
    (the assumed original is an exact hypothesis, so it carries no
    uncertainty). ``coverage_k`` is the coverage factor of the pasted
    uncertainties."""
    _guard_canon(canon_t0, "original")
    _guard_canon(canon_today, "present-day")
    atoms_today = measured_atoms_from_canon(canon_today)
    sig_today = sigma_atoms_from_entries(
        entries_today, canon_today, atoms_today,
        default_rel_sigma=default_rel_sigma, coverage_k=coverage_k,
    )
    return check_compatibility(
        canon_t0,
        canon_today,
        age_s,
        sigma_atoms_today=sig_today,
        default_rel_sigma=default_rel_sigma,
    )
