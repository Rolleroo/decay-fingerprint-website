"""Generate a synthetic present-day measurement of U-235 + its next five
daughters (actinium series: Th-231, Pa-231, Ac-227, Th-227, Ra-223).

Same pattern as the round-trip validation tests (tests/test_reverse.py):
choose a ground-truth t=0 composition, decay it forward with the trusted
library, then -- unlike the tests -- perturb each value with realistic
measurement noise and quote the per-line uncertainty in the paste format
the reverse tab accepts. Because the dataset is synthetic, the ground
truth is known exactly, so Mode B's reconstruction can be judged.

Scenario: a freshly reprocessed U-235 sample with imperfect daughter
removal, measured again after a known interval.

Two files are written:

- u235_synthetic_2day.txt  -- age 2 days. Everything is inside the
  resolvability gate; expect mostly green flags, a MARGINAL Th-231
  (~1.9 half-lives of reach-back amplifies its noise), and Fr-223
  auto-pruned as a negligible mid-chain intermediate.
- u235_synthetic_50yr.txt  -- age 50 years. Th-231/Th-227/Ra-223 are
  thousands of half-lives back: their t=0 amounts are genuinely
  unknowable, so the tool must refuse them (gate) and taint the chain --
  the honest-refusal behaviour, on purpose.

Run:  .venv/Scripts/python.exe examples/generate_u235_dataset.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import radioactivedecay as rd

HERE = pathlib.Path(__file__).parent
DAY_S = 86400.0
YEAR_S = 86400.0 * 365.25

# Ground truth at t=0, in Bq: mostly-purified U-235 with residual daughters.
TRUTH_T0_BQ = {
    "U-235": 5.0e4,
    "Th-231": 2.0e4,
    "Pa-231": 8.0e2,
    "Ac-227": 1.2e3,
    "Th-227": 9.0e2,
    "Ra-223": 1.1e3,
}

# Quoted 1-sigma measurement uncertainty per nuclide (typical alpha/gamma
# spectrometry spread: worst for the weak Pa-231 line).
QUOTED_SIGMA = {
    "U-235": 0.02,
    "Th-231": 0.03,
    "Pa-231": 0.08,
    "Ac-227": 0.06,
    "Th-227": 0.04,
    "Ra-223": 0.04,
}

SEED = 20260702


def make_dataset(age_s: float, rng: np.random.Generator) -> tuple[list[str], dict[str, float]]:
    decayed = rd.Inventory(TRUTH_T0_BQ, "Bq").decay(age_s, "s")
    today_bq = {str(k): float(v) for k, v in decayed.activities("Bq").items()}

    lines = []
    for nuclide in TRUTH_T0_BQ:  # measure only the six of interest
        sigma = QUOTED_SIGMA[nuclide]
        measured = today_bq[nuclide] * (1.0 + sigma * rng.standard_normal())
        lines.append(f"{nuclide}, {measured:.6g}, {sigma * 100:g}%")
    return lines, today_bq


def main() -> None:
    rng = np.random.default_rng(SEED)

    for name, age_s, label in [
        ("u235_synthetic_2day.txt", 2 * DAY_S, "2 days"),
        ("u235_synthetic_50yr.txt", 50 * YEAR_S, "50 years"),
    ]:
        lines, _ = make_dataset(age_s, rng)
        path = HERE / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n--- {name}  (known age: {label}; paste into the Reverse tab, unit = Bq) ---")
        print("\n".join(lines))

    print("\n--- ground truth at t=0 (Bq) for judging the reconstruction ---")
    for nuclide, bq in TRUTH_T0_BQ.items():
        print(f"{nuclide}, {bq:g}")


if __name__ == "__main__":
    main()
