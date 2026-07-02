"""Reverse Mode B: reconstruct the t=0 composition from a present-day
measurement and a *known* age (reverse spec Sec 3-4).

Method is forward-model-based, never raw inversion of the decay matrix:
the library's own forward decay builds a transfer matrix A where
``A[i, j]`` = atoms of nuclide i present today per atom of nuclide j at
t=0. In topological (parent-first) order A is lower triangular, so the
back-solve is a triangular solve -- exactly "subtract the parent's
ingrowth, divide by own survival" -- and the unreliability of the
backward direction is handled by gating and flagging, not by pretending
the inverse is well-behaved.

Everything is solved in atoms (the one basis in which the problem is
linear); conversion in from the user's unit and back out for display is
delegated to the same library scaling the forward tool uses.

Uncertainty is Monte Carlo (reverse spec Sec 4: DQPB pattern): sample the
measured values within their uncertainties, back-solve every trial with
the same triangular factor, and report each nuclide as a distribution
(median + 95% interval), never a bare point estimate.

References
----------
- Forward engine: A. Malins & T. Lemoine, "radioactivedecay: A Python
  package for radioactive decay calculations", J. Open Source Softw. 7
  (71), 3318 (2022). DOI 10.21105/joss.03318. ICRP-107 decay data.
- Method pattern (forward-model back-solve, Monte Carlo uncertainty
  propagation, analytical-resolvability gating before MC, short-lived-
  intermediate pruning): T. Pollard, J. Woodhead, J. Hellstrom, J. Engel,
  R. Powell & R. Drysdale, "DQPB: software for calculating disequilibrium
  U-Pb ages", Geochronology 5, 181-196 (2023).
  DOI 10.5194/gchron-5-181-2023. Reference implementation studied:
  pysoplot (MIT License, https://pypi.org/project/pysoplot/); the pattern
  is transferred, no code is copied.
- Governing chain equations: H. Bateman, "Solution of a system of
  differential equations occurring in the theory of radioactive
  transformations", Proc. Cambridge Philos. Soc. 15, 423-427 (1910) --
  evaluated directly by the validation suite's independent solver.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import radioactivedecay as rd

from app.conversions import CanonResult, scale_to_display
from app.engine import is_stable, nuclide_half_life_s
from app.parsing import ParsedEntry

LN2 = math.log(2)

# MC defaults (reverse spec Sec 4: 1e5-1e6 trials, seconds to run; a
# default per-measurement uncertainty is seeded so MC runs out of the box
# and the user overrides per line with real measurement precision).
DEFAULT_TRIALS = 100_000
DEFAULT_REL_SIGMA = 0.05

# Analytical-resolvability gate (DQPB pattern, applied before MC): beyond
# this many half-lives of reach-back a nuclide's t=0 amount is simply not
# recoverable from its present-day value -- the memory is gone -- so it is
# excluded from the solve and reported as not reconstructable rather than
# letting e^{+lambda*t} amplification manufacture a number.
GATE_HALF_LIVES = 40.0

# The same threshold prunes unmeasured mid-chain intermediates (DQPB
# pruning rule): with T1/2 < age/40, any physically plausible t=0
# inventory of the intermediate (bounded by equilibrium with its parent)
# contributes at most ~T1/2/age <= 2.5% to what grew into its descendants
# over the interval -- below typical measurement uncertainty. Decay
# *through* the intermediate is unaffected (the library's forward model
# inside A always includes the full chain).

# A mid-chain gap's state is supplied as an assumed *initial* (t=0) value
# with an uncertainty (reverse spec Sec 4 engine rules), never as a
# pseudo-measurement of its present-day state -- assuming the present and
# back-solving would amplify the assumption's uncertainty by e^{+lambda*t}
# and swamp the daughter below the gap. A gap short-lived enough to have
# equilibrated within the interval is assumed at equilibrium with its
# nearest measured ancestor at t=0; longer-lived gaps are assumed absent
# at t=0. Either way the assumption carries this (loose, deliberately
# honest) 1-sigma relative uncertainty, propagated through the MC and
# surfaced via the assumption flag.
ASSUMED_GAP_REL_SIGMA = 0.5

# Conditioning thresholds. The underlying figures (reach-back in
# half-lives; relative half-width of the 95% MC interval) are always
# reported alongside the colour, per reverse spec Sec 4 output 3.
REL_WIDTH_PASS = 0.25  # 95% interval within ~+/-25% of the median
REL_WIDTH_MARGINAL = 1.0  # interval wider than +/-100% of the median
HALF_LIVES_MARGINAL = 20.0
NEG_FRACTION_MARGINAL = 0.05


# --- decay-chain graph helpers (library data only; user never declares
# --- parent/daughter relationships, reverse spec Sec 4 "not user inputs")


@lru_cache(maxsize=None)
def direct_progeny(nuclide: str) -> tuple[tuple[str, float], ...]:
    """(daughter, branching fraction) pairs, canonical names, from library data.

    Spontaneous-fission placeholders (the library lists 'SF' as a progeny
    string with no fission yields behind it) are filtered out -- consistent
    with the forward tool's declared scope boundary of decay only.
    """
    nuc = rd.Nuclide(nuclide)
    pairs = []
    for child, bf in zip(nuc.progeny(), nuc.branching_fractions()):
        try:
            pairs.append((rd.Nuclide(child).nuclide, float(bf)))
        except Exception:
            continue
    return tuple(pairs)


@lru_cache(maxsize=None)
def descendants(nuclide: str) -> frozenset[str]:
    """Every nuclide reachable from ``nuclide`` by decay (not including itself)."""
    seen: set[str] = set()
    stack = [nuclide]
    while stack:
        for child, _bf in direct_progeny(stack.pop()):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return frozenset(seen)


def _chain_distance(ancestor: str, target: str) -> int:
    """Shortest number of decay steps from ancestor down to target (BFS)."""
    frontier = {ancestor}
    steps = 0
    seen = set(frontier)
    while frontier:
        steps += 1
        frontier = {
            child
            for n in frontier
            for child, _bf in direct_progeny(n)
            if child not in seen and not seen.add(child)
        }
        if target in frontier:
            return steps
    return -1


def _toposort(nuclides: list[str]) -> list[str]:
    """Parent-first order; ties broken alphabetically for determinism."""
    desc = {n: descendants(n) for n in nuclides}
    remaining = set(nuclides)
    order: list[str] = []
    while remaining:
        ready = sorted(n for n in remaining if not any(n in desc[m] for m in remaining if m != n))
        if not ready:  # decay data is a DAG; a cycle here means corrupt data
            raise RuntimeError(f"Cycle detected among {sorted(remaining)}")
        order.extend(ready)
        remaining -= set(ready)
    return order


# --- per-nuclide unit scaling (library-owned physics, cached) ---


@lru_cache(maxsize=None)
def _per_atom_factors(nuclide: str) -> tuple[float, float, float]:
    """(Bq, g, mol) represented by one atom of ``nuclide``, from the library."""
    inv = rd.Inventory({nuclide: 1.0}, "num")
    return (
        float(inv.activities("Bq")[nuclide]),
        float(inv.masses("g")[nuclide]),
        float(inv.moles("mol")[nuclide]),
    )


def atoms_to_display(nuclide: str, atoms: float, canon: CanonResult) -> float:
    """Convert an atom count to the user's chosen unit (reverse spec Sec 4
    output 1: results in the user's units, same conversion module as forward).

    For the three fraction kinds this returns the *base quantity* (Bq, g,
    or mol); the caller renormalizes across the whole composition so the
    reconstructed fractions sum to 1 (coherence check, Sec 4 output 6).
    """
    if not math.isfinite(atoms):
        return atoms
    bq, g, mol = _per_atom_factors(nuclide)
    kind = canon.kind
    if kind in ("activity", "specific_activity"):
        display = "Bq" if kind == "specific_activity" else canon.display_unit
        return scale_to_display(bq * atoms, "Bq", display)
    if kind == "mass":
        return scale_to_display(g * atoms, "g", canon.display_unit)
    if kind == "amount":
        return atoms if canon.library_unit == "num" else mol * atoms
    base = {"fraction_activity": bq, "fraction_mass": g, "fraction_mole": mol}[kind]
    return base * atoms


# --- result containers ---


@dataclass(frozen=True)
class ReconstructedNuclide:
    nuclide: str
    measured_atoms: float  # present-day atoms; NaN for assumed gap fills (their prior is at t=0)
    median_atoms: float  # NaN when gated (not reconstructable)
    lo_atoms: float  # 2.5th percentile
    hi_atoms: float  # 97.5th percentile
    half_lives_back: float
    rel_half_width: float  # (hi-lo)/(2*|median|); inf when undefined
    negative_fraction: float  # fraction of MC trials that went negative
    conditioning: str  # 'pass' | 'marginal' | 'fail'
    conditioning_note: str
    assumed: bool  # True for unmeasured mid-chain gap fills
    assumption_dependent: bool  # separate axis from conditioning (Sec 4 output 4)
    assumption_note: str
    chain_tainted: bool  # whole-chain unreliability flag (Sec 4 output 5)

    @property
    def unreliable(self) -> bool:
        return self.conditioning == "fail" or self.chain_tainted


@dataclass(frozen=True)
class ForwardCheckRow:
    nuclide: str
    measured_atoms: float
    modeled_atoms: float  # reconstructed t=0 decayed forward by the known age
    rel_diff: float  # (modeled - measured) / measured; inf if measured == 0


@dataclass(frozen=True)
class ReverseResult:
    age_s: float
    rows: list[ReconstructedNuclide]  # topological order, parents first
    forward_check: list[ForwardCheckRow]
    forward_check_ok: bool
    excluded_stable: list[str]  # carry no timing information; never reconstructed
    pruned: list[str]  # negligible intermediates (T1/2 << age)
    warnings: list[str] = field(default_factory=list)
    n_trials: int = 0


# --- input preparation ---


def measured_atoms_from_canon(canon: CanonResult) -> dict[str, float]:
    """Present-day composition converted to atoms via the library."""
    if not canon.contents:
        return {}
    inv = rd.Inventory(canon.contents, canon.library_unit)
    numbers = {str(k): float(v) for k, v in inv.numbers().items()}
    # Inventory only reports nuclides it kept; make sure explicit zeros
    # survive, since "measured zero" is information in reverse mode.
    for n in canon.contents:
        numbers.setdefault(str(rd.Nuclide(n).nuclide), 0.0)
    return numbers


def sigma_atoms_from_entries(
    entries: list[ParsedEntry],
    canon: CanonResult,
    atoms: dict[str, float],
    default_rel_sigma: float = DEFAULT_REL_SIGMA,
) -> dict[str, float]:
    """Per-nuclide 1-sigma uncertainty in atoms.

    Relative uncertainties (and the seeded default) scale the atom count
    directly; absolute ones are converted through the same unit path as
    the value itself, so the sigma is expressed in exactly the same basis.
    """
    total_raw = sum(e.value for e in entries)
    sigmas: dict[str, float] = {}
    for e in entries:
        n = e.nuclide
        if n not in atoms:
            continue
        if e.uncertainty is None:
            sigmas[n] = default_rel_sigma * atoms[n]
        elif e.uncertainty_is_relative:
            sigmas[n] = e.uncertainty * atoms[n]
        elif e.value > 0:
            sigmas[n] = e.uncertainty * (atoms[n] / e.value)
        else:
            # Value is zero (e.g. below detection limit with a quoted
            # bound): derive atoms-per-pasted-unit from the library.
            per_unit = float(rd.Inventory({n: 1.0}, canon.library_unit).numbers()[n])
            if canon.kind.startswith("fraction_") and total_raw > 0:
                per_unit /= total_raw
            sigmas[n] = e.uncertainty * per_unit
    return sigmas


# --- the Mode B solve ---


def reconstruct_t0(
    canon: CanonResult,
    age_s: float,
    sigma_atoms: dict[str, float] | None = None,
    default_rel_sigma: float = DEFAULT_REL_SIGMA,
    n_trials: int = DEFAULT_TRIALS,
    seed: int | None = None,
) -> ReverseResult:
    """Reconstruct the t=0 composition from a present-day measurement.

    ``canon`` is the canonicalized present-day paste (same module as
    forward). ``sigma_atoms`` gives per-nuclide 1-sigma uncertainties in
    atoms (see ``sigma_atoms_from_entries``); nuclides missing from it get
    ``default_rel_sigma`` times their measured atoms.
    """
    if age_s <= 0:
        raise ValueError("Known age must be positive.")

    warnings: list[str] = []

    # Stable nuclides carry no timing information and cannot be un-grown
    # (reverse spec Sec 4 engine rules): excluded entirely -- and before
    # the unit conversion, since a stable nuclide has no activity to
    # convert when the paste is in an activity unit.
    excluded_stable = sorted(n for n in canon.contents if is_stable(n))
    if excluded_stable:
        canon = dataclasses.replace(
            canon,
            contents={n: v for n, v in canon.contents.items() if n not in excluded_stable},
        )
    measured = measured_atoms_from_canon(canon)

    if not measured:
        return ReverseResult(
            age_s=age_s,
            rows=[],
            forward_check=[],
            forward_check_ok=True,
            excluded_stable=excluded_stable,
            pruned=[],
            warnings=["No radioactive nuclides to reconstruct."],
        )

    sigma_atoms = dict(sigma_atoms or {})
    for n in measured:
        sigma_atoms.setdefault(n, default_rel_sigma * measured[n])

    def half_lives_back(nuclide: str) -> float:
        return age_s / nuclide_half_life_s(nuclide)

    # Resolvability gate, before anything touches the matrix.
    gated = sorted(n for n in measured if half_lives_back(n) > GATE_HALF_LIVES)
    if gated:
        warnings.append(
            f"Not reconstructable at this age (reach-back beyond {GATE_HALF_LIVES:g} "
            f"half-lives; the t=0 amount is no longer encoded in today's value): "
            f"{', '.join(gated)}. Their entire chains are flagged unreliable."
        )

    # --- unmeasured-nuclide split rule (reverse spec Sec 4 engine rules) ---
    # Mid-chain gaps: unmeasured, unstable, with measured nuclides both
    # above and below. Top-of-chain phantoms need no enumeration: anything
    # unmeasured with nothing measured above it is simply absent from the
    # solve, i.e. held at zero -- recorded as a global assumption below.
    measured_set = set(measured)
    gap_candidates: set[str] = set()
    for m in measured_set:
        for u in descendants(m):
            if u in measured_set or is_stable(u):
                continue
            if descendants(u) & measured_set:
                gap_candidates.add(u)

    pruned = sorted(u for u in gap_candidates if half_lives_back(u) > GATE_HALF_LIVES)
    gaps = sorted(gap_candidates - set(pruned))

    # (equilibrium ratio to anchor, anchor nuclide, equilibrium?, note)
    assumed_members: dict[str, tuple[float, str, bool, str]] = {}
    for u in gaps:
        anc_candidates = [
            m for m in measured_set if m not in gated and u in descendants(m)
        ]
        if not anc_candidates:
            warnings.append(
                f"{u} sits below only non-reconstructable nuclides; it is dropped "
                f"from the solve (its chain is already flagged)."
            )
            continue
        nearest = min(anc_candidates, key=lambda m: _chain_distance(m, u))
        # Equal-activity (secular equilibrium) inventory relative to the
        # anchor's own t=0 amount: N_u = N_anc * lambda_anc / lambda_u.
        eq_ratio = nuclide_half_life_s(u) / nuclide_half_life_s(nearest)
        if nuclide_half_life_s(u) <= age_s / 2:
            is_eq, note = True, f"assumed in equilibrium with {nearest} at t=0"
        else:
            is_eq, note = False, "assumed absent at t=0 (T1/2 comparable to the age)"
        assumed_members[u] = (eq_ratio, nearest, is_eq, note)

    # --- assemble the (square, triangular) system ---
    solve_names = _toposort(
        [n for n in measured_set if n not in gated] + list(assumed_members)
    )
    n_solve = len(solve_names)
    index = {n: i for i, n in enumerate(solve_names)}

    values = np.zeros(n_solve)
    sigmas = np.zeros(n_solve)
    for i, name in enumerate(solve_names):
        if name not in assumed_members:
            values[i] = measured[name]
            sigmas[i] = sigma_atoms[name]

    # Transfer matrix from the trusted forward engine: column j is one atom
    # of solve_names[j] decayed forward by the known age.
    A = np.zeros((n_solve, n_solve))
    for j, name in enumerate(solve_names):
        decayed = rd.Inventory({name: 1.0}, "num").decay(age_s, "s")
        numbers = {str(k): float(v) for k, v in decayed.numbers().items()}
        for i, target in enumerate(solve_names):
            A[i, j] = numbers.get(target, 0.0)

    def back_solve(b: np.ndarray, gap_z: np.ndarray | None) -> np.ndarray:
        """Forward substitution down the (lower-triangular, parent-first)
        system: subtract every ancestor's ingrowth, divide by own survival.

        ``b`` holds present-day atoms per measured row, shape (n, trials).
        Gap rows take no equation; their t=0 amount is drawn directly from
        the assumed prior, anchored per-trial to the already-reconstructed
        t=0 amount of their nearest measured ancestor (``gap_z`` holds the
        standard-normal draws; None means central values).
        """
        trials = b.shape[1]
        x = np.zeros((n_solve, trials))
        for i, name in enumerate(solve_names):
            if name in assumed_members:
                eq_ratio, anchor, is_eq, _note = assumed_members[name]
                eq_amount = eq_ratio * x[index[anchor]]
                center = eq_amount if is_eq else 0.0
                if gap_z is None:
                    x[i] = center
                else:
                    x[i] = np.clip(
                        center + ASSUMED_GAP_REL_SIGMA * np.abs(eq_amount) * gap_z[i],
                        0.0,
                        None,
                    )
            else:
                ingrowth = A[i, :i] @ x[:i] if i else 0.0
                # The gate keeps every diagonal survival factor >= 2^-40,
                # so this division is never near-singular in float64.
                x[i] = (b[i] - ingrowth) / A[i, i]
        return x

    central = back_solve(values[:, None], None)[:, 0]

    # --- Monte Carlo ---
    if n_solve and (np.any(sigmas > 0) or assumed_members):
        rng = np.random.default_rng(seed)
        samples = rng.normal(values[:, None], sigmas[:, None], size=(n_solve, n_trials))
        np.clip(samples, 0.0, None, out=samples)  # measured amounts are never negative
        gap_z = rng.standard_normal((n_solve, n_trials))
        trials = back_solve(samples, gap_z)
        lo, med, hi = np.percentile(trials, [2.5, 50.0, 97.5], axis=1)
        neg_frac = np.mean(trials < 0, axis=1)
        effective_trials = n_trials
    else:
        lo = med = hi = central
        neg_frac = np.zeros(n_solve)
        effective_trials = 0

    # --- per-nuclide flags ---

    def assumption_status(name: str) -> tuple[bool, str]:
        """A reconstruction is assumption-dependent if its own t=0 state
        was assumed, or any solve-set/gated ancestor's was (in the
        back-solve, dependency flows parent -> descendant)."""
        if name in assumed_members:
            return True, assumed_members[name][3]
        reasons = []
        for anc in solve_names:
            if anc != name and name in descendants(anc) and anc in assumed_members:
                reasons.append(f"{anc} ({assumed_members[anc][3]})")
        for anc in gated:
            if name in descendants(anc):
                reasons.append(f"{anc} not reconstructable (held at zero at t=0)")
        return bool(reasons), "; ".join(reasons)

    rows: list[ReconstructedNuclide] = []
    for name in solve_names:
        i = index[name]
        hlb = half_lives_back(name)
        median = float(med[i])
        width = float(hi[i] - lo[i])
        if not math.isfinite(median) or median == 0.0:
            rel_width = 0.0 if width == 0.0 else math.inf
        else:
            rel_width = width / (2 * abs(median))

        notes = []
        if not math.isfinite(median):
            conditioning = "fail"
            notes.append("non-finite reconstruction")
        elif median < 0:
            conditioning = "fail"
            notes.append("negative reconstructed amount (bad-reconstruction signature)")
        elif rel_width > REL_WIDTH_MARGINAL:
            conditioning = "fail"
            notes.append(f"95% interval spans +/-{rel_width:.0%} of the median")
        elif rel_width > REL_WIDTH_PASS or hlb > HALF_LIVES_MARGINAL or neg_frac[i] > NEG_FRACTION_MARGINAL:
            conditioning = "marginal"
            if rel_width > REL_WIDTH_PASS:
                notes.append(f"95% interval spans +/-{rel_width:.0%} of the median")
            if hlb > HALF_LIVES_MARGINAL:
                notes.append(f"{hlb:.1f} half-lives of reach-back")
            if neg_frac[i] > NEG_FRACTION_MARGINAL:
                notes.append(f"{neg_frac[i]:.0%} of MC trials went negative")
        else:
            conditioning = "pass"

        dependent, dep_note = assumption_status(name)
        rows.append(
            ReconstructedNuclide(
                nuclide=name,
                measured_atoms=math.nan if name in assumed_members else float(values[i]),
                median_atoms=median,
                lo_atoms=float(lo[i]),
                hi_atoms=float(hi[i]),
                half_lives_back=hlb,
                rel_half_width=rel_width,
                negative_fraction=float(neg_frac[i]),
                conditioning=conditioning,
                conditioning_note="; ".join(notes),
                assumed=name in assumed_members,
                assumption_dependent=dependent,
                assumption_note=dep_note,
                chain_tainted=False,  # filled in below
            )
        )

    for name in gated:
        hlb = half_lives_back(name)
        dependent, dep_note = assumption_status(name)
        rows.append(
            ReconstructedNuclide(
                nuclide=name,
                measured_atoms=measured[name],
                median_atoms=math.nan,
                lo_atoms=math.nan,
                hi_atoms=math.nan,
                half_lives_back=hlb,
                rel_half_width=math.inf,
                negative_fraction=0.0,
                conditioning="fail",
                conditioning_note=(
                    f"{hlb:.0f} half-lives of reach-back exceeds the "
                    f"{GATE_HALF_LIVES:g} half-life resolvability gate"
                ),
                assumed=False,
                assumption_dependent=dependent,
                assumption_note=dep_note,
                chain_tainted=False,
            )
        )

    # --- whole-chain unreliability (reverse spec Sec 4 output 5): coarse
    # --- and safe -- one bad *measured* member taints every member of its
    # --- chain; unrelated chains stay clean. Assumed gap fills are judged
    # --- on the assumption axis instead, so a deliberately loose gap prior
    # --- does not taint an otherwise healthy chain.
    row_by_name = {r.nuclide: r for r in rows}
    names = list(row_by_name)
    chain_of = {n: n for n in names}

    def find(n: str) -> str:
        while chain_of[n] != n:
            chain_of[n] = chain_of[chain_of[n]]
            n = chain_of[n]
        return n

    for a in names:
        for b in names:
            if a < b and (b in descendants(a) or a in descendants(b)):
                chain_of[find(a)] = find(b)

    bad_chains = {find(r.nuclide) for r in rows if r.conditioning == "fail" and not r.assumed}
    tainted_names = {n for n in names if find(n) in bad_chains}
    if tainted_names:
        rows = [
            dataclasses.replace(r, chain_tainted=r.nuclide in tainted_names) for r in rows
        ]

    negative_medians = [r.nuclide for r in rows if math.isfinite(r.median_atoms) and r.median_atoms < 0]
    if negative_medians:
        warnings.append(
            f"Negative reconstructed amounts for {', '.join(negative_medians)} -- the classic "
            f"signature of a bad reconstruction. The present-day input is inconsistent with "
            f"pure closed-system decay over this age."
        )

    # --- forward-check overlay (reverse spec Sec 4 output 7, default ON):
    # --- reconstruct t=0 -> decay forward by the known age with the same
    # --- trusted forward engine -> compare against the measured input.
    forward_check: list[ForwardCheckRow] = []
    forward_check_ok = True
    if n_solve:
        t0_contents = {
            name: max(float(central[index[name]]), 0.0)
            for name in solve_names
            if math.isfinite(central[index[name]])
        }
        modeled = {
            str(k): float(v)
            for k, v in rd.Inventory(t0_contents, "num").decay(age_s, "s").numbers().items()
        }
        atol = 1e-9 * max(measured.values(), default=0.0)
        for name in sorted(measured_set):
            m_val = measured[name]
            mod_val = modeled.get(name, 0.0)
            rel = (mod_val - m_val) / m_val if m_val > 0 else (math.inf if mod_val > atol else 0.0)
            forward_check.append(ForwardCheckRow(name, m_val, mod_val, rel))
            if name not in gated and abs(mod_val - m_val) > 1e-6 * m_val + atol:
                forward_check_ok = False
        if not forward_check_ok:
            warnings.append(
                "Forward check failed: decaying the reconstructed t=0 composition forward "
                "does not reproduce the measured values. The reconstruction is suspect "
                "regardless of the per-nuclide flags."
            )

    warnings.append(
        "Results are conditional on closed-system behaviour (no gain or loss except decay) "
        "and on the interim rule that unlisted nuclides are absent today and that any "
        "unmeasured parent above the measured set was absent throughout."
    )

    # Preserve topological order for solved rows, then gated ones.
    return ReverseResult(
        age_s=age_s,
        rows=rows,
        forward_check=forward_check,
        forward_check_ok=forward_check_ok,
        excluded_stable=excluded_stable,
        pruned=pruned,
        warnings=warnings,
        n_trials=effective_trials,
    )


def reconstruct_from_entries(
    entries: list[ParsedEntry],
    canon: CanonResult,
    age_s: float,
    default_rel_sigma: float = DEFAULT_REL_SIGMA,
    n_trials: int = DEFAULT_TRIALS,
    seed: int | None = None,
) -> ReverseResult:
    """UI-facing convenience: derive per-nuclide sigmas from the parsed
    lines (stable nuclides skipped before any unit conversion, since a
    stable nuclide has no activity) and run the Mode B reconstruction."""
    radioactive = {n: v for n, v in canon.contents.items() if not is_stable(n)}
    canon_radioactive = dataclasses.replace(canon, contents=radioactive)
    atoms = measured_atoms_from_canon(canon_radioactive)
    sigma = sigma_atoms_from_entries(entries, canon_radioactive, atoms, default_rel_sigma)
    return reconstruct_t0(
        canon,
        age_s,
        sigma_atoms=sigma,
        default_rel_sigma=default_rel_sigma,
        n_trials=n_trials,
        seed=seed,
    )
