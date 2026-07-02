# Mode A Addendum — Known t=0 Composition → Solve for Age

Companion to the reverse/validation spec, which specifies Mode B fully and
only *records* the deferred design for A. This addendum fixes A's contract
before implementation, in the same style: inputs, outputs, gates, and the
validation cases that exist before the output is relied on.

Mode A is what the spec calls it: **a solver around the same forward model
Mode B built.** No back-solve, no transfer matrix — the forward direction
is the well-conditioned one, so A is a weighted least-squares fit of one
scalar (the age) against the trusted forward engine.

---

## 1. Inputs

1. **Known t=0 composition** — paste box, same parser and unit module as
   forward/Mode B. Optional third-column uncertainties accepted; lines
   without one default to **exact** (0%), since the t=0 composition is the
   "known" reference (often nominal or certified), unlike a measurement.
2. **Measured present-day composition** — second paste box, same parser
   and unit module. Lines without an uncertainty get the seeded default
   (5%, configurable), same as Mode B.
3. **Units** — each paste has its own unit picker, since the solve runs in
   atoms. **Absolute units only in v1** (activity / specific activity /
   mass / amount). The three relative (fraction/percent) kinds are refused
   with a clear error: a fraction-mode fit changes the objective (shape
   only, no scale) and is deferred rather than silently approximated.
4. **Closed-system acknowledgement** — same as Mode B; conditions every
   result.
5. *(Advanced)* MC trial count (default 20,000 — each trial re-solves the
   age, so the default is lower than B's) and the default measurement
   uncertainty.

### Differences from Mode B, decided here

- **Stable nuclides participate.** Mode B excludes them (they cannot be
  un-grown); Mode A is a forward fit, where radiogenic accumulation in a
  stable daughter is a classic chronometer (e.g. Zr-90 from Sr-90). A
  stable nuclide cannot be *expressed* in an activity unit — pasting one
  under an activity unit is refused with a message, not dropped.
- **A missing line means "not measured", never "measured zero".** Only
  pasted present-day lines create equations. (Mode B's interim
  missing-equals-zero rule is about completing a back-solve; A needs no
  such completion.)
- **Unmeasured mid-chain intermediates need no special rule** — the
  forward model carries decay through them automatically. The whole gap /
  phantom / pruning apparatus of Mode B does not apply.
- **Present-day nuclides that cannot be produced from the known t=0
  composition** (not in its decay closure) are excluded from the fit with
  a loud warning — they can never fit, and would otherwise poison the age.

## 2. Method

- Everything converts to atoms (the linear basis), as in Mode B.
- The forward model f_i(t) = atoms of nuclide i at age t from the known
  t=0 composition is evaluated by the trusted engine on a log-spaced time
  grid spanning [1 s, 40 × the longest finite half-life present], then
  refined near the optimum with true engine evaluations (no grid error in
  the reported age).
- **Central solve:** weighted least squares,
  chi²(t) = Σ_i ((m_i − f_i(t)) / σ_i)², minimized over log t. Nuclides
  with σ_i = 0 get a tiny relative floor so "exact" inputs behave as
  near-hard constraints.
- **t=0 uncertainties (v1 approximation):** propagated as an additive
  model term σ_model,i evaluated at the central age (linear in the t=0
  amounts), folded into σ_eff,i² = σ_i² + σ_model,i². Full re-sampling of
  the t=0 composition inside the MC is deferred; the approximation is
  recorded in the output notes whenever t=0 uncertainties are present.
- **Monte Carlo:** sample the measured values within σ_eff, re-minimize
  every trial (vectorized over the precomputed grid + parabolic
  refinement). Age reported as **median + 95% interval**, never a bare
  point estimate — same output philosophy as Mode B.

## 3. Gates and flags (before/around the MC, DQPB pattern)

1. **Flatness gate (before MC):** if chi²(t) varies by less than O(1)
   across the entire search window, no measured nuclide changed
   meaningfully over any admissible age — the age is analytically
   unresolvable from this input. Refuse; do not fabricate.
2. **Interval gate (after MC):** if the 95% age interval collapses onto a
   window edge or spans more than a factor 10⁴, report *not resolvable*
   rather than certifying a number. The interval is still shown.
3. **Ambiguity flag:** ingrowth curves are not monotonic, so distinct ages
   can fit equally well (e.g. a lone Tc-99m level from pure Mo-99 occurs
   once before and once after the ingrowth peak). All local chi² minima
   within Δchi² ≤ 4 of the global minimum are reported as candidate ages
   and flagged; the MC interval naturally widens across them.
4. **Consistency flag:** chi²/dof ≫ 1 at the optimum means the inputs are
   not consistent with closed-system decay from the stated t=0 (open
   system, wrong t=0, or understated uncertainties). Flag loudly; the
   residual table localizes which nuclide misfits.
5. **Forward-check overlay (default ON):** the per-nuclide residual table
   *is* the forward check — the known t=0 decayed forward by the solved
   age, overlaid on the measured values, with the mismatch in both percent
   and sigma units.

## 4. Outputs

1. Solved age: median + 95% interval (and the central WLS age), in
   human-readable units.
2. Per-nuclide residual table (measured vs modeled at the solved age,
   mismatch in % and σ), including each nuclide's sensitivity — whether it
   actually constrains the age or just comes along for the ride.
3. chi²/dof, candidate ambiguous ages if any, and every gate/flag above.
4. All warnings inherited from the shared modules (parse errors, unit
   validation) plus the closed-system conditionality note.

## 5. Validation (exists before the output is relied on)

- **Analytical anchor:** single nuclide, today = t0/8 → age = 3 half-lives
  exactly, digit-for-digit against t = T½ · log2(N0/N).
- **Round-trips (Layer 4):** forward-decay a known t=0 by a chosen age
  with the library, feed both compositions back, require the solved age to
  match — Pu-241/Am-241 (the spec's dating pair) and a stable-daughter
  chronometer (Sr-90 → Zr-90).
- **Flatness gate:** a nuclide that cannot change over the window (U-238
  over laboratory timescales) is refused, not fitted.
- **Ambiguity:** the Tc-99m-from-Mo-99 double-solution case is detected
  and both candidate ages reported.
- **MC coverage:** with realistic noise, the truth falls inside the 95%
  age interval.
- **Producibility:** a measured nuclide outside the t=0 decay closure is
  excluded with a warning and does not corrupt the age.
- Layer 2 (independent forward-model cross-check) already covers the
  forward engine that A fits against; nothing new needed.

## 6. Deferred (recorded, not built)

- Relative/fraction units for either paste (shape-only objective).
- Full MC re-sampling of t=0 uncertainties (v1 folds them into σ_eff at
  the central age).
- Mode C (age + initial split jointly), multi-parent attribution, and the
  Ludwig (1977) / Wendt & Carl (1985) disequilibrium forms — all per the
  main spec, to be designed in a dedicated session before any code.
