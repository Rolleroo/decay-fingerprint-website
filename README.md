---
title: Rad Decay Tool
emoji: ☢️
colorFrom: blue
colorTo: gray
sdk: streamlit
app_file: app.py
pinned: false
---

<!-- The YAML block above configures a Hugging Face Space (Streamlit SDK,
     entry point app.py). It is metadata for HF only; it does not affect
     GitHub or the app itself. -->

# Decay Fingerprint Tool

A Streamlit app that takes a pasted list of nuclides + amounts (a
"fingerprint") and runs it through time in either direction:

- **Forward decay** -- decay the fingerprint forward to a chosen target
  time and show the resulting composition, including progeny that grow in
  along the way.
- **Reverse (Mode B)** -- given today's measured fingerprint and a *known*
  age, reconstruct the composition at t=0. Forward-model based (never raw
  matrix inversion), with Monte Carlo uncertainty on every value,
  per-nuclide conditioning and assumption flags, whole-chain unreliability
  flagging, and a default-on forward-check that re-decays the answer to
  verify it reproduces the measured input.
- **Age (Mode A)** -- given the *known* t=0 composition and today's
  measured fingerprint, solve for the age: a weighted least-squares fit of
  one scalar against the forward engine, with Monte Carlo uncertainty,
  resolvability gates (a flat clock is refused, not fitted), ambiguity
  detection (two ages that fit equally well are both reported), and a
  chi-squared consistency flag. Contract: `docs/mode-a-addendum.md`.

All decay physics, unit conversions, and nuclide data come from the
published, validated
[`radioactivedecay`](https://github.com/radioactivedecay/radioactivedecay)
library (ICRP-107 data) -- this app is a thin, tested wrapper around it.
The reverse mode's design (forward-model back-solve, MC uncertainty,
resolvability gating, intermediate pruning) follows the pattern published
in [DQPB](https://doi.org/10.5194/gchron-5-181-2023) (Pollard et al. 2023).

**Scope boundary:** the underlying library does not model spontaneous-fission
ingrowth or neutron-induced activation. Results are decay-only -- not
suitable for reactor-burnup fingerprints without understanding this.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate   # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
streamlit run app/ui.py
```

## Usage

**Forward tab**

1. Paste `nuclide, value` lines, one per line (comma, tab, or space
   separated -- pasting straight out of a spreadsheet works).
2. Pick the unit the values are in: Activity, Specific activity, Mass,
   Amount, or a Relative fraction/percent.
3. Set how far forward in time to decay, and click **Decay**.
4. The table shows every nuclide present at that time -- inputs and
   in-grown progeny -- sorted by quantity. Optionally filter out
   short-lived nuclides by half-life, and copy the table as
   tab-separated text for pasting into Excel.

**Reverse tab**

1. Paste today's measured fingerprint the same way, optionally adding a
   third column with each line's 1-sigma measurement uncertainty
   (`Cs-137, 3.7e9, 5%` or `Cs-137, 3.7e9 ± 1e8`). Lines without one get a
   configurable default so the Monte Carlo runs out of the box.
2. Pick the unit (same list as forward) and enter the **known age**.
3. Acknowledge the closed-system assumption and click **Reconstruct t=0**.
4. Every nuclide is shown with a median + 95% interval -- none are
   withheld -- alongside a **conditioning** flag (is the back-solve
   numerically trustworthy at this reach-back?), an **assumptions** flag
   (does the value depend on something that wasn't measured?), and
   whole-chain flagging (one bad member taints its chain). A default-on
   forward check re-decays the answer and compares it to your input.

## Project layout

```
app/
  parsing.py        # nuclide normalization + validation (+ uncertainty column)
  conversions.py     # unit <-> canonical base (scaling + per-nuclide)
  engine.py          # thin wrapper over radioactivedecay: decay + filtering
  reverse.py         # Mode B back-solve: transfer matrix, MC, gates, flags
  age_solve.py       # Mode A age fit: WLS vs forward engine, MC, gates
  ui.py               # Streamlit front end (Forward + Reverse + Age tabs)
docs/
  mode-a-addendum.md # Mode A contract (inputs/outputs/gates/validation)
  revalidation-*.md  # end-to-end revalidation reports (findings + fixes)
examples/             # synthetic datasets + generator (U-235 chain)
tests/                # parsing/conversions/engine/validation/reverse/age/UI suite
validation/           # deep revalidation sweep (manual-run: MC calibration,
                      # cross-mode consistency, randomized round-trips)
```

The parser, conversion layer, and decay engines are kept independent of the
UI so they're directly reusable (and testable) without Streamlit.

## Tests & validation

```bash
pytest
```

Validation is a first-class deliverable here: no long-chain result is
trusted on visual plausibility. The suite covers:

- **Analytical anchors** -- single-nuclide and hand-evaluated two-member
  Bateman cases with closed-form answers.
- **Independent-implementation cross-check** -- the forward engine is
  compared against a separately written Bateman path-enumeration solver
  (60-digit arithmetic, no matrix exponential) on the full ~20-member
  U-238 chain, agreeing to ~6 significant figures.
- **Round-trips** -- known composition, decayed forward by the library,
  reconstructed by Mode B, must return the original (Pu-241/Am-241 and
  U-238-chain cases).
- **Unmeasured-nuclide rule** -- the gamma-spec pattern (measured parent
  above, measured daughter below an unmeasured intermediate) closes the
  forward check with the gap modelled as an assumed t=0 state.
- **Gating/flagging behaviour** -- reach-back beyond ~40 half-lives is
  refused rather than fabricated; inconsistent input produces a loud
  negative-amount flag; one bad chain member taints the whole chain.
- **Mode A gates** -- the age anchor (today = t0/8 -> exactly 3
  half-lives), stable-daughter chronometry (Zr-90 from Sr-90), refusal
  when nothing measurably decays, detection of ambiguous double-solution
  ages (the Tc-99m ingrowth peak; the Th-234 ingrowth/decay two-branch
  case), and a chi-squared flag when the inputs are inconsistent with
  closed-system decay.
- **Deep revalidation sweep** (`validation/deep_revalidation.py`,
  manual-run) -- statistical calibration of the 95% MC intervals over many
  noisy repetitions (both modes), cross-mode consistency (Mode B's
  reconstruction fed to Mode A recovers the age), and randomized
  round-trip fuzzing. Findings and fixes are written up in
  `docs/revalidation-*.md`.

## Status

- **Done:** forward decay to a single target time, full unit support
  (activity/specific activity/mass/amount/relative), half-life filtering,
  Excel-friendly export, reverse Mode B (known age -> t=0 composition)
  with MC uncertainty, age Mode A (known t=0 -> solve for age) per
  `docs/mode-a-addendum.md`, and the validation suite above.
- **Not yet built:** a decay-over-time graph (deliberately deferred),
  Mode C (solve age + initial split together -- needs a design session
  and spec revision first), multi-parent attribution, relative-unit
  support in Mode A, deployment.

## Credits & references

**Decay engine** -- built on
[`radioactivedecay`](https://github.com/radioactivedecay/radioactivedecay),
created by Alex Malins and Thom Lemoine, with Ian Cullen and other
contributors (MIT License, © 2020-2024 Japan Atomic Energy Agency &
contributors). It performs all decay-chain physics and unit conversions;
this app adds none of its own.

If you use results from this tool in research, please cite the project's
own paper, per their request:

> Alex Malins & Thom Lemoine, *radioactivedecay: A Python package for
> radioactive decay calculations*. Journal of Open Source Software, 7 (71),
> 3318 (2022). DOI: [10.21105/joss.03318](https://doi.org/10.21105/joss.03318).

**Reverse-mode method** -- the reverse mode transfers the pattern published
in DQPB: the forward-model + numerical back-solve structure, Monte Carlo
uncertainty propagation (10^5-10^6 trials), the analytical-resolvability
gate applied before MC, and short-lived-intermediate pruning:

> Timothy Pollard, Jon Woodhead, John Hellstrom, John Engel, Roger Powell &
> Russell Drysdale, *DQPB: software for calculating disequilibrium U-Pb
> ages*. Geochronology, 5, 181-196 (2023).
> DOI: [10.5194/gchron-5-181-2023](https://doi.org/10.5194/gchron-5-181-2023).

DQPB's underlying pure-Python package
[`pysoplot`](https://pypi.org/project/pysoplot/) (MIT License) was studied
as the reference implementation; the pattern is transferred, no code is
copied.

**Governing equations** -- the decay-chain mathematics is Bateman's general
solution, which the validation suite also evaluates directly (path
enumeration in 60-digit arithmetic) as an implementation-independent
cross-check of the engine:

> H. Bateman, *Solution of a system of differential equations occurring in
> the theory of radioactive transformations*. Proceedings of the Cambridge
> Philosophical Society, 15, 423-427 (1910).

Nuclear data:

- Decay data: ICRP, 2008. *Nuclear Decay Data for Dosimetric Calculations*,
  ICRP Publication 107, Ann. ICRP 38(3). © 2008 A. Endo and K.F. Eckerman.
- Atomic mass data: W.J. Huang et al. 2021, *AME2020 (I)*, Chinese Phys. C
  45 030002; Meng Wang et al. 2021, *AME2020 (II)*, Chinese Phys. C 45
  030003; F.G. Kondev et al. 2021, *NUBASE2020*, Chinese Phys. C 45 030001.
  Source: [AMDC](https://www-nds.iaea.org/amdc/).
