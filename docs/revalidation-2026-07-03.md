# Revalidation Report — 2026-07-03

End-to-end revalidation of the decay tool (forward engine, reverse Mode B,
age Mode A), performed after all three modes landed and before the code
cleanup pass. Scope, method, findings, and fixes.

**Verdict: sound, with one real bug found and fixed (Mode A sigma floor),
one piece of genuine physics documented as expected behaviour (ingrowth
ambiguity), and two recorded spec deviations to address later.**

Why revalidation is more than re-running the suite: the tests and the code
share an author, so a shared blind spot passes both. This pass therefore
(a) audited the existing coverage against the validation spec's layers,
(b) re-read the solvers with fresh eyes, and (c) added checks of *kinds*
that did not exist during the build — statistical calibration, cross-mode
chaining, randomized cases, and new chains for the independent solver.

---

## 1. Baseline

- Full suite before changes: **109 passed** (parsing, conversions, engine,
  validation, reverse, age, UI smoke).
- After this pass: **113 passed** (two Layer-2 chains, the sigma-floor
  regression test, the ambiguity physics test), plus the deep sweep script.

## 2. Coverage audit against the spec's validation layers

| Spec layer | Status | Evidence |
|---|---|---|
| 1 — Analytical anchors | ✅ covered | Forward: half-life halving, hand Bateman, secular equilibrium. Mode B: one-half-life doubling, hand two-member Bateman. Mode A: t = T½·log2(N0/N) exact. |
| 2 — Independent implementation | ✅ covered, **extended this pass** | Path-enumeration Bateman (60-digit mpmath, no matrix exponential) vs the library: Mo-99, U-238 (×2 epochs), U-235 — now also **Th-232** (Bi-212 α/β branch) and **Pu-241** (2.5e-5 α branch). All agree to ~6 s.f. |
| 3 — Conservation laws | ⚠️ **partial (recorded deviation)** | Atom conservation, branching sums, fraction renorm are asserted in tests. The spec calls for these to run **on every calculation at runtime**, not just in tests. Not implemented. → Recommendation R1. |
| 4 — Round-trips | ✅ covered, extended | Suite round-trips (Pu-241/Am-241, U-238 chain, stable-daughter chronometer) plus this pass's randomized fuzz sweep (§4.4). |
| 4a — Unmeasured-nuclide rule | ✅ covered | Gamma-spec gap case (Mo-99/Tc-99m/Tc-99) closes the forward check with the gap as an assumed t=0 state. |
| 5 — Published reference cases | ⚠️ **open gap** | The Pu-241/Am-241 case is a round-trip, not a reproduction of a published worked example with a literature answer. → Recommendation R2. |
| Conversions (spec "minimum viable") | ✅ covered | 1 Ci = 3.7e10 Bq; Co-60 ≈ 4.18e13 Bq/g; Bq→atoms→Bq on two very different half-lives; unit round-trips. |

## 3. Findings

### F1 (bug, fixed): Mode A sigma floor could silently erase real uncertainties

`app/age_solve.py` replaced zero sigmas with a floor computed as
`1e-6 × max(measured values, forward-model maximum)` and applied it with
`np.maximum(sigma, floor)` — i.e. to **every** nuclide, not just exact
ones. When one measured value dwarfs another (U-238 at ~1e24 atoms next to
in-grown Th-234 at ~1e13), the trace nuclide's real 2% uncertainty was
silently replaced by a floor ~10⁶× larger, erasing its weight in the fit.
Since the trace daughter is precisely the nuclide carrying the age
information on short timescales, the age came back unconstrained /
falsely "not resolvable".

**Fix:** per-nuclide floor (`1e-6 ×` the nuclide's own measured value),
applied only where sigma is exactly zero (`np.where(sigma > 0, sigma,
floor)`). **Regression test:**
`test_trace_daughter_far_below_parent_scale_keeps_its_real_sigma` — fails
on the old code, passes on the fix, and confirms the measured parent also
kills the late-branch ambiguity.

Why the original suite missed it: every Mode A test measured nuclides of
comparable magnitude, so the global floor never exceeded a real sigma.
Classic shared-blind-spot failure — the fresh-eyes scenario (trace
daughter under a huge parent) had simply never been written down.

### F2 (physics, documented as expected behaviour): lone ingrowth measurements are two-valued

Constructing the F1 test revealed real physics: a **lone** Th-234
measurement under a U-238 source is genuinely ambiguous. The ingrowth
curve rises to secular equilibrium (~30-day branch) and then falls with
U-238's own decay — a ~3.5 Gyr age reproduces the same measured value
exactly. The solver behaves correctly: it surfaces both candidate ages and
refuses to certify either (interval spans the branches → not resolvable).
Documented as `test_lone_ingrowth_measurement_under_long_parent_is_ambiguous`,
which also serves as an ambiguity-detection check on natural (not
contrived) physics. Measuring the parent as well breaks the tie.

### F3–F4 (spec deviations, recorded): see Recommendations R1–R2.

## 4. New evidence added this pass

All in `validation/deep_revalidation.py` (manual-run, seeded, exits
nonzero on failure) unless noted.

### 4.1 MC interval calibration — Mode B
The suite checks single noisy cases land in their intervals; this measures
the **coverage rate**. 50 independent noisy repetitions (5% noise) of a
three-nuclide reconstruction at 10 years: truth fell inside the 95%
interval **140/150 = 93.3%** of the time — consistent with nominal 95%
(3σ binomial acceptance ≥ 89.7%). Intervals are neither inflated nor
overconfident.

### 4.2 MC interval calibration — Mode A
25 noisy repetitions (3% noise) of the Pu-241/Am-241 age solve at 20
years: truth age inside the 95% interval **23/25 = 92%** — consistent with
nominal (acceptance ≥ 81.9% at 3σ for n=25).

### 4.3 Cross-mode consistency
Forward (library) → Mode B reconstruction (deterministic) → reconstruction
fed to Mode A as the known t=0 → solved age **25.000003 y** vs truth 25 y
(rel. err. 1.4e-7). The two solvers, built and tested separately, agree
when chained.

### 4.4 Randomized round-trip fuzz
25 seeded random cases (1–3 nuclides from a 10-nuclide pool spanning H-3
to U-238, log-uniform amounts 1e12–1e20 atoms, ages 0.02–3× the shortest
chosen half-life), none of which appear in the suite. Worst recovery
error across all cases: **1.6e-12 relative**; every forward check closed;
no negative reconstructions.

### 4.5 Layer-2 extension (permanent, in the suite)
Th-232 and Pu-241 chains added to the independent cross-check
(`tests/test_validation.py`); both agree with the library to ~6 s.f.

## 5. Recommendations (carried to the roadmap)

- **R1 (Layer 3):** add cheap runtime self-audits (atom-conservation and
  no-negatives checks on every engine call), or record a conscious
  decision that test-time-only is acceptable. Natural home: the cleanup
  pass.
- **R2 (Layer 5):** reproduce at least one published worked example with a
  literature answer (a Pu-241/Am-241 dating from the literature, or a
  U-Pb reference material) — candidates should fall out of the
  I/O-format scan of similar sites/tools.
- **R3:** `validation/deep_revalidation.py` should be re-run after any
  change to the solvers or the MC machinery (it is deliberately not part
  of the per-commit pytest run; ~1 minute).

## 6. What this pass does NOT establish

Consistent with the spec's honesty discipline: the nuclear *data*
(ICRP-107 half-lives, branching fractions) is shared by both
implementations in the Layer-2 check and is taken on the authority of the
peer-reviewed dataset — no check here validates it. Closed-system
behaviour of any real sample remains an assumption the tool cannot test.
