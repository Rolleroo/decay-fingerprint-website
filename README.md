# Decay Fingerprint Tool

A Streamlit app that takes a pasted list of nuclides + amounts (a
"fingerprint"), decays it forward to a chosen target time, and shows the
resulting composition as a table -- including progeny that grow in along
the way.

Forward decay only for now (see [Status](#status)). All decay physics, unit
conversions, and nuclide data come from the published, validated
[`radioactivedecay`](https://github.com/radioactivedecay/radioactivedecay)
library (ICRP-107 data) -- this app is a thin, tested wrapper around it.

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

1. Paste `nuclide, value` lines, one per line (comma, tab, or space
   separated -- pasting straight out of a spreadsheet works).
2. Pick the unit the values are in: Activity, Specific activity, Mass,
   Amount, or a Relative fraction/percent.
3. Set how far forward in time to decay, and click **Decay**.
4. The table shows every nuclide present at that time -- inputs and
   in-grown progeny -- sorted by quantity. Optionally filter out
   short-lived nuclides by half-life, and copy the table as
   tab-separated text for pasting into Excel.

## Project layout

```
app/
  parsing.py        # nuclide normalization + validation
  conversions.py     # unit <-> canonical base (scaling + per-nuclide)
  engine.py          # thin wrapper over radioactivedecay: decay + filtering
  ui.py               # Streamlit front end
tests/                # parsing/conversions/engine/validation suite
```

The parser, conversion layer, and decay engine are kept independent of the
UI so they're directly reusable (and testable) without Streamlit.

## Tests

```bash
pytest
```

## Status

- **Done:** forward decay to a single target time, full unit support
  (activity/specific activity/mass/amount/relative), half-life filtering,
  Excel-friendly export.
- **Not yet built:** a decay-over-time graph (deliberately deferred),
  reverse/age-dating mode (inverse Bateman), deployment.

## Credits

Built on [`radioactivedecay`](https://github.com/radioactivedecay/radioactivedecay),
created by Alex Malins and Thom Lemoine, with Ian Cullen and other
contributors (MIT License, © 2020-2024 Japan Atomic Energy Agency &
contributors).

If you use results from this tool in research, please cite the project's
own paper, per their request:

> Alex Malins & Thom Lemoine, *radioactivedecay: A Python package for
> radioactive decay calculations*. Journal of Open Source Software, 7 (71),
> 3318 (2022). DOI: [10.21105/joss.03318](https://doi.org/10.21105/joss.03318).

Nuclear data:

- Decay data: ICRP, 2008. *Nuclear Decay Data for Dosimetric Calculations*,
  ICRP Publication 107, Ann. ICRP 38(3). © 2008 A. Endo and K.F. Eckerman.
- Atomic mass data: W.J. Huang et al. 2021, *AME2020 (I)*, Chinese Phys. C
  45 030002; Meng Wang et al. 2021, *AME2020 (II)*, Chinese Phys. C 45
  030003; F.G. Kondev et al. 2021, *NUBASE2020*, Chinese Phys. C 45 030001.
  Source: [AMDC](https://www-nds.iaea.org/amdc/).
