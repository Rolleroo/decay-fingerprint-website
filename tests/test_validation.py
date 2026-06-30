"""End-to-end input-validation checks (spec Sec 6.5).

Single-layer checks (does the parser flag a bad nuclide; does the
fraction-sum check fire) already live in test_parsing.py and
test_conversions.py. This module exercises the full
parse -> canonicalize -> decay pipeline together, to confirm a problem
caught at one layer actually blocks the pipeline before reaching the next,
and that a realistic multi-row paste makes it through end to end.
"""

import pytest

from app.conversions import ValidationError, canonicalize
from app.engine import auto_time_grid_s, run_time_series
from app.parsing import parse_paste


def test_unrecognized_nuclide_blocks_before_reaching_the_engine():
    result = parse_paste("Cs-137, 10\nNotANuclide, 5")
    assert not result.ok
    # The caller must check result.ok before calling canonicalize -- with
    # only the valid entries, the pipeline should still proceed for those.
    canon = canonicalize(result.entries, "Bq")
    assert canon.contents == {"Cs-137": 10.0}


def test_bad_fraction_sum_blocks_before_reaching_the_engine():
    result = parse_paste("Cs-137, 10\nCo-60, 10")  # sums to 20, not 1 or 100
    assert result.ok
    with pytest.raises(ValidationError):
        canonicalize(result.entries, "activity fraction")


def test_duplicate_nuclide_blocks_with_a_clear_message():
    result = parse_paste("Cs-137, 10\nCs-137, 5\nCo-60, 3")
    assert not result.ok
    assert any("Duplicate" in e.message and "Cs-137" in e.message for e in result.errors)
    # Only the unaffected line should survive for downstream use.
    assert [e.nuclide for e in result.entries] == ["Co-60"]


def test_empty_paste_does_not_crash_the_pipeline():
    result = parse_paste("   \n\n  ")
    assert result.ok
    assert result.entries == []
    # Calling canonicalize on an empty entry list should not raise for an
    # absolute unit -- it just yields an empty inventory.
    canon = canonicalize(result.entries, "Bq")
    assert canon.contents == {}


def test_realistic_multirow_paste_runs_end_to_end():
    paste = "\n".join(
        [
            "Cs-137, 3.7e9",
            "Co-60, 1.2e8",
            "Sr-90, 5.0e7",
            "Mo-99, 2.0e6",
            "Tc-99m, 1.0e5",
            "  cs134 , 4.4e6",  # casing + whitespace tolerance
            "I-131, 8.8e8",
            "Am-241, 3.1e5",
            "U-238, 9.99e3",
            "Pu-239, 7.7e4",
        ]
    )
    result = parse_paste(paste)
    assert result.ok, result.errors
    assert len(result.entries) == 10

    canon = canonicalize(result.entries, "Bq")
    times = auto_time_grid_s([e.nuclide for e in result.entries])
    series = run_time_series(canon, times)

    # every pasted nuclide, plus at least its direct progeny, should show up
    for e in result.entries:
        assert e.nuclide in series.nuclides
    assert len(series.nuclides) > len(result.entries)  # progeny grew in


def test_negative_value_never_reaches_canonicalize():
    result = parse_paste("Cs-137, -1")
    assert not result.ok
    assert result.entries == []


def test_zero_value_flows_through_to_a_harmless_zero_quantity():
    result = parse_paste("Cs-137, 0\nCo-60, 100")
    assert result.ok
    canon = canonicalize(result.entries, "Bq")
    times = auto_time_grid_s([e.nuclide for e in result.entries])
    series = run_time_series(canon, times)
    assert all(v == 0.0 for v in series.activities_bq["Cs-137"])
