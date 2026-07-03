# I/O-Format & Feature Scan of Similar Tools — 2026-07-03

Roadmap step 2. Survey of what comparable tools ingest, output, and offer,
to drive feature decisions before the cleanup pass freezes the parser and
result contracts.

**Audience decision (user, 2026-07-03):** the tool is aimed squarely at
**waste-management professionals and health-physics professionals in the
UK** — not the public. Consequences: professional workflow fit (lab-report
conventions, decay-in-storage, limits comparison) outranks simplification;
the UI register is "instrument"; **no features aimed exclusively at other
countries' regulations** (e.g. no built-in US 10 CFR 61.55 tables —
generic machinery + user-supplied or UK/IAEA tables instead).

## Tools surveyed

| Tool | Audience | Notable I/O and features |
|---|---|---|
| Rad Pro Calculator (radprocalculator.com) | HP professionals | Decay **to/from a date** (not just an interval); **back-decay** ("knowing current activity, find original"); **timed decay** (how long until activity ≤ X); gamma point-source dose rate; specific-activity gram calculators; SI/US units both ways |
| Nucleonica Decay Engine++ (commercial) | Nuclear professionals | **Time mode and date mode**; **nuclide mixtures** as first-class saved objects; selectable output quantities in the grid; graphs with sliders; large-nuclide-set engine; decay trees |
| WISE Universal Decay Calculator | Waste/fuel-cycle | 1252 nuclides; **preset inventories** (nat-U/nat-Th in equilibrium, enriched/depleted U, spent fuel by burnup/cooling, HLW); time-series tables incl. ingrowth; **point-source air kerma at distance**; line/stacked-area charts, log axes; copy-paste output |
| HPS decay calculator, RP Alba calculators (UK) | HP / RPA | Simple decay + **decay-in-storage planning** (time to threshold, half-lives elapsed, fraction remaining) |
| ORNL Radiological Toolbox (NRC) | HP | Dose coefficients (ICRP 68/72), external dose-rate coefficients (FGR-12), gamma constants, interaction coefficients — the "data handbook" end of the spectrum |
| Genie-2000 / GammaVision report chain | Gamma-spec labs | CSV/Excel-compatible exports; per-nuclide **activity + propagated uncertainty + MDA**; "less-than" (censored) entries for non-detects; **reference/count date** on every report; parent/daughter decay correction |

## Findings → the pre-identified hypotheses, confirmed/adjusted

1. **Date-mode input is the professional norm** (Rad Pro, Nucleonica both
   lead with it). Lab certificates and gamma reports state a *reference
   date*; professionals want "decay from cert date to today/date X," not
   "enter an elapsed time." Confirmed, and cheap. Applies to all three
   tabs (forward target date; reverse/age reference dates).
2. **σ-convention (1σ/2σ/95%) selector** — confirmed by Genie reporting
   conventions (propagated uncertainties quoted at varying k). Cheap
   insurance against silently mis-scaled MC intervals.
3. **Censored values (`< MDA`)** — confirmed: every gamma-spec report
   carries non-detects as less-than values. Parse-and-flag is cheap;
   statistically honest treatment (upper-limit priors) is medium work.
4. **File upload (CSV/XLSX)** — confirmed; Genie exports are
   Excel-compatible CSV. Column layouts vary → a small column-mapping
   step beats hard-coding any one lab's format. Real sample files from
   the user's own instruments would pin this down.
5. **N42.42/SPE spectral files** — no evidence the peer *calculator*
   tools ingest them (they live one step earlier in the toolchain).
   Decision: **out of scope** unless the user's own workflow emits N42
   analysis-results sections; revisit only on demand. (Resolves the open
   non-goal question.)
6. **Presets/mixture libraries** (WISE, Nucleonica) — natural-series
   equilibrium inventories etc. as one-click starting points; mixtures
   saved/reloaded as files.
7. **Decay-in-storage** (RP Alba, HPS, every university RSO page) — the
   single most common professional decay *task* not yet covered: time
   until activity ≤ threshold (or ≥ N half-lives), for release/disposal
   planning. Mode A's machinery already contains the solve.
8. **Limits comparison / sum-of-fractions** — the universal waste-side
   operation (classification, acceptance, transport, out-of-scope).
   Machinery is country-neutral (per-nuclide limit table → fractions →
   sum); tables differ by regime. UK-relevant presets: UK LLW definition
   (≤4 GBq/te alpha, ≤12 GBq/te beta/gamma), IAEA transport A1/A2
   (apply in the UK via ADR), EPR out-of-scope/exemption values.
   User-supplied CSV tables cover everything else (site WAC, permits).
9. **Gamma dose rate at distance** (Rad Pro, WISE air kerma) — the main
   *physics* extension the peers have and we don't. Needs a per-nuclide
   gamma-constant dataset (e.g. ORNL's ICRP-107-derived constants, or
   Delacroix handbook) with provenance/licensing care — the
   `radioactivedecay` library does **not** expose emission energies, so
   this is a new data dependency, validated against published constants.
10. **Decay heat** (Nucleonica) — same story as dose rate (needs
    energy-per-decay data not exposed by the library); niche below
    spent-fuel scale. Deferred unless the user wants it.
11. **Dose coefficients (Sv/Bq)** — Radiological-Toolbox territory; big
    liability-adjacent tables; out of scope for this tool.
12. **Output side** — peers offer copy-paste tables, chart toggles, log
    axes. We already have copy-for-Excel; download buttons (CSV/JSON) and
    the deferred decay-curve graph close the gap. Nucleonica's
    "selectable output quantities" is a nice-to-have.

## Feature options and effort (presented to user 2026-07-03)

See the option menu in the session summary / roadmap memory. Effort
ratings assume the existing architecture (shared parser/engine, tabs).

## Layer-5 note

The scan surfaced no ready-made published worked example to reproduce
(revalidation recommendation R2). Best candidates remain a literature
Pu-241/Am-241 dating or a decay-in-storage worked example from an
IAEA/HPA publication — keep open.
