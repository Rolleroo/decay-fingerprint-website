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

# ☢️ Decay Fingerprint Tool

**Live app: <https://huggingface.co/spaces/RadDecay/rad-decay-tool>**

Radionuclide decay, reverse reconstruction, age-dating, and hypothesis
checking for **waste-management and health-physics work**. Paste a list of
nuclides + amounts (a "fingerprint") — or upload a CSV/XLSX results table —
and run it through time in either direction.

> **Beta.** The inverse modes are labelled beta in the app for a reason:
> read the flags, not just the numbers. Spotted a wrong number, a confusing
> label, or something missing?
> [Open an issue](https://github.com/Rolleroo/decay-fingerprint-website/issues)
> — include the tab, your input lines, and what you expected.

## The four modes

- **Forward decay** — decay the fingerprint forward to a chosen time (or
  between two dates) and show the resulting composition, including progeny
  that grow in along the way. A runtime conservation audit checks every
  result for negative amounts and atom-count violations.
- **Reconstruct original** *(beta)* — given today's measured fingerprint
  and a *known* age, reconstruct the original composition. Forward-model
  based (never raw matrix inversion), with Monte Carlo uncertainty on every
  value, per-nuclide confidence and assumption flags, whole-chain
  unreliability flagging, and a default-on self-check that re-decays the
  answer to verify it reproduces the measured input.
- **Find the age** *(beta)* — given the *known* original composition and
  today's measured fingerprint, solve for the age: a weighted least-squares
  fit of one scalar against the forward engine, with Monte Carlo
  uncertainty, resolvability gates (a flat clock is refused, not fitted),
  ambiguity detection (two ages that fit equally well are both reported),
  and a chi-squared consistency flag. Contract: `docs/mode-a-addendum.md`.
- **Compatibility check** *(beta)* — you already have a theory (the sample
  started as *this* and is *this* old): does today's measurement back it
  up? Nothing is solved — the assumed original is decayed forward once and
  scored against the measurement with zero free parameters. Answers two
  questions plainly: *is it the right amount of the right stuff?* and *are
  at least the proportions right?* (total amount set aside).

All decay physics, unit conversions, and nuclide data come from the
published, validated
[`radioactivedecay`](https://github.com/radioactivedecay/radioactivedecay)
library (ICRP-107 data) — this app is a thin, tested wrapper around it.
The inverse-mode design (forward-model back-solve, MC uncertainty,
resolvability gating, intermediate pruning) follows the pattern published
in [DQPB](https://doi.org/10.5194/gchron-5-181-2023) (Pollard et al. 2023).

**Scope boundary:** the underlying library does not model
spontaneous-fission ingrowth or neutron-induced activation. Results are
decay-only — not suitable for reactor-burnup fingerprints without
understanding this.

## Input conveniences

- **Paste or upload** — free-text `nuclide, value[, uncertainty]` lines, or
  a CSV/XLSX results table (e.g. a gamma-spec export). Columns are
  auto-detected and the mapping is shown; non-UTF-8 encodings, `;`
  delimiters, decimal commas, and instrument preamble rows are handled.
- **Units** — activity (Bq…TBq, dpm, Ci…pCi), specific activity (Bq/g,
  Bq/kg), mass (µg…t), amount (mol, atoms), and relative fraction/percent.
- **Time as dates** — give a reference date and a measurement date instead
  of an elapsed interval; the Age tab converts a solved age into an implied
  origin/production date.
- **Uncertainty conventions** — tell the tool whether quoted uncertainties
  are 1σ, 2σ, or 95% (k ≈ 1.96) so Monte Carlo intervals are scaled
  correctly.
- **Export** — CSV/JSON download buttons on every results table, plus an
  Excel-ready tab-separated copy block. Exports are CSV-injection-safe.

## Quick start (run it locally)

```bash
python -m venv .venv
.venv/Scripts/activate   # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

For the test/validation toolchain add `pip install -r requirements-dev.txt`.

## Project layout

```
app.py                 # entry-point shim (HF Spaces / streamlit run)
app/
  parsing.py           # nuclide normalization + validation (+ uncertainty column)
  conversions.py       # unit <-> canonical base (scaling + per-nuclide)
  engine.py            # thin wrapper over radioactivedecay + conservation audit
  reverse.py           # reconstruct-original back-solve: transfer matrix, MC, gates, flags
  age_solve.py         # age fit (WLS vs forward engine, MC, gates) + compatibility check
  ingest.py            # CSV/XLSX upload -> paste text (encodings, delimiters, preambles)
  dates.py             # date-pair -> interval, age -> implied origin date
  ui.py                # Streamlit front end (all four tabs)
docs/
  mode-a-addendum.md   # age-mode contract (inputs/outputs/gates/validation)
  revalidation-*.md    # end-to-end revalidation reports (findings + fixes)
  security-audit-*.md  # input-hardening / injection-safety audit
examples/              # synthetic datasets + generator (U-235 chain)
tests/                 # parsing/conversions/engine/validation/reverse/age/compat/UI suite
validation/            # deep revalidation sweep (manual-run: MC calibration,
                       # cross-mode consistency, randomized round-trips)
```

The parser, conversion layer, and decay engines are kept independent of the
UI so they're directly reusable (and testable) without Streamlit.

## Tests & validation

```bash
pytest
```

Validation is a first-class deliverable here: no long-chain result is
trusted on visual plausibility. The suite (170 tests) covers:

- **Analytical anchors** — single-nuclide and hand-evaluated two-member
  Bateman cases with closed-form answers.
- **Independent-implementation cross-check** — the forward engine is
  compared against a separately written Bateman path-enumeration solver
  (60-digit arithmetic, no matrix exponential) on the full ~20-member
  U-238 chain, agreeing to ~6 significant figures.
- **Round-trips** — known composition, decayed forward by the library,
  reconstructed by the reverse mode, must return the original
  (Pu-241/Am-241 and U-238-chain cases).
- **Unmeasured-nuclide rule** — the gamma-spec pattern (measured parent
  above, measured daughter below an unmeasured intermediate) closes the
  self-check with the gap modelled as an assumed original state.
- **Gating/flagging behaviour** — tracing back beyond ~40 half-lives is
  refused rather than fabricated; inconsistent input produces a loud
  negative-amount flag; a hard chain failure taints the whole chain, while
  a merely-untraceable short-lived daughter taints only its descendants.
- **Age-mode gates** — the age anchor (today = t0/8 → exactly 3
  half-lives), stable-daughter chronometry (Zr-90 from Sr-90), refusal
  when nothing measurably decays, detection of ambiguous double-solution
  ages (the Tc-99m ingrowth peak; the Th-234 ingrowth/decay two-branch
  case), and a chi-squared flag when the inputs are inconsistent with
  closed-system decay.
- **Compatibility check** — exact match accepted, wrong age rejected,
  pattern-vs-amount separation (2× scaled measurement fails on amounts,
  passes on ratios with the scale recovered), unproducible nuclides
  flagged as compatibility-breakers.
- **UI smoke tests** — every tab driven end-to-end through Streamlit's
  AppTest harness: paste goes in, table comes out, gating checkboxes work.
- **Deep revalidation sweep** (`validation/deep_revalidation.py`,
  manual-run) — statistical calibration of the 95% MC intervals over many
  noisy repetitions (both inverse modes), cross-mode consistency (the
  reverse mode's reconstruction fed to the age mode recovers the age), and
  randomized round-trip fuzzing. Findings and fixes are written up in
  `docs/revalidation-*.md`.

Input handling is hardened per `docs/security-audit-2026-07-03.md`: paste
and upload size caps, XXE-safe XLSX parsing, CSV-injection-safe exports,
and no eval/exec/network/file-write on user input. The app is a stateless
calculator — nothing you enter is stored.

## Status

- **Live (public beta):** all four tabs deployed at the link above.
- **Not yet built:** a decay-over-time graph, a decay-in-storage solver
  (how long until this drops below a limit?), censored `<MDA` values,
  sum-of-fractions vs regulatory limits, Mode C (solve age + initial split
  together — needs a design session first), multi-parent attribution,
  relative-unit support in the age mode.

## Credits & references

**Decay engine** — built on
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

**Inverse-mode method** — the reverse mode transfers the pattern published
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

**Governing equations** — the decay-chain mathematics is Bateman's general
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

## License

[MIT](LICENSE) — the same license as the `radioactivedecay` and `pysoplot`
libraries this tool builds on.
