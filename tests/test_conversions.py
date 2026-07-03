import math

import pytest
import radioactivedecay as rd

from app.conversions import ValidationError, canonicalize, scale_to_display
from app.parsing import parse_paste


def entries_of(text):
    result = parse_paste(text)
    assert result.ok, result.errors
    return result.entries


# --- pure scaling: passed straight through to the library, untouched ---


def test_activity_unit_passthrough_is_pure_scaling():
    canon = canonicalize(entries_of("Cs-137, 1"), "Ci")
    assert canon.library_unit == "Ci"
    assert canon.kind == "activity"
    assert canon.contents == {"Cs-137": 1.0}


def test_known_anchor_one_ci_equals_3p7e10_bq():
    canon = canonicalize(entries_of("Co-60, 1"), "Ci")
    inv = rd.Inventory(canon.contents, units=canon.library_unit)
    assert inv.activities("Bq")["Co-60"] == pytest.approx(3.7e10, rel=0, abs=1e-3)


def test_known_anchor_co60_specific_activity_of_pure_gram():
    # 1 g of pure Co-60 -> ~4.18e13 Bq, a published constant. Exercises the
    # per-nuclide Bq<->mass path (spec Sec 6.1), distinct from the
    # "specific activity" *input mode* which is bulk-sample activity
    # concentration and is tested separately below.
    canon = canonicalize(entries_of("Co-60, 1"), "g")
    inv = rd.Inventory(canon.contents, units=canon.library_unit)
    assert inv.activities("Bq")["Co-60"] == pytest.approx(4.18e13, rel=2e-3)


def test_round_trip_unit_to_base_and_back():
    canon = canonicalize(entries_of("Cs-137, 250"), "kBq")
    inv = rd.Inventory(canon.contents, units=canon.library_unit)
    assert inv.activities("kBq")["Cs-137"] == pytest.approx(250.0)


def test_cross_dimension_round_trip_two_different_half_lives():
    # Bq -> atoms -> Bq round trip for nuclides with very different
    # half-lives catches any place a *global* lambda was used instead of
    # per-nuclide (spec Sec 6.1).
    for nuclide in ["Co-60", "U-238"]:
        canon = canonicalize(entries_of(f"{nuclide}, 1000"), "Bq")
        inv = rd.Inventory(canon.contents, units=canon.library_unit)
        atoms = inv.numbers()[nuclide]
        inv2 = rd.Inventory({nuclide: atoms}, units="num")
        assert inv2.activities("Bq")[nuclide] == pytest.approx(1000.0, rel=1e-6)


def test_scaling_isolation_mass_units_never_touch_lambda():
    # g -> mg should be an exact x1000 factor identical for every nuclide,
    # regardless of half-life -- proof no decay constant entered the path.
    canon_long = canonicalize(entries_of("U-238, 2"), "g")
    canon_short = canonicalize(entries_of("Co-60, 2"), "g")
    for canon in (canon_long, canon_short):
        inv = rd.Inventory(canon.contents, units=canon.library_unit)
        assert inv.masses("mg")[next(iter(canon.contents))] == pytest.approx(2000.0)


# --- specific activity input mode: bulk-sample activity concentration ---


def test_specific_activity_is_treated_as_activity_numerically():
    # Per design decision: Bq/g and Bq/kg represent activity concentration
    # in the bulk sample, not the nuclide's own intrinsic specific
    # activity. The bulk mass is an unstated constant that doesn't decay,
    # so the entered number is carried through as Bq unchanged.
    canon = canonicalize(entries_of("Cs-137, 1200"), "Bq/kg")
    assert canon.library_unit == "Bq"
    assert canon.kind == "specific_activity"
    assert canon.contents == {"Cs-137": 1200.0}


def test_specific_activity_bq_per_g_and_bq_per_kg_are_not_rescaled_against_each_other():
    # These are different bulk-mass denominators, not a unit conversion --
    # 1200 Bq/kg is not divided/multiplied by 1000 to compare to Bq/g.
    canon_g = canonicalize(entries_of("Cs-137, 5"), "Bq/g")
    canon_kg = canonicalize(entries_of("Cs-137, 5"), "Bq/kg")
    assert canon_g.contents == canon_kg.contents == {"Cs-137": 5.0}


# --- relative/fraction modes ---


def test_activity_fraction_sum_check_fires_in_fraction_mode():
    with pytest.raises(ValidationError):
        canonicalize(entries_of("Cs-137, 0.5\nCo-60, 0.2"), "activity fraction", frac_as_percent=False)


def test_activity_fraction_sum_check_fires_in_percent_mode():
    with pytest.raises(ValidationError):
        canonicalize(entries_of("Cs-137, 50\nCo-60, 20"), "activity fraction", frac_as_percent=True)


def test_sum_check_does_not_fire_in_absolute_modes():
    # An absolute-mode "sum" of e.g. 0.7 Bq total is meaningless and must
    # never be flagged (spec Sec 6.5).
    canon = canonicalize(entries_of("Cs-137, 0.5\nCo-60, 0.2"), "Bq")
    assert canon.contents == {"Cs-137": 0.5, "Co-60": 0.2}


def test_fraction_within_tolerance_is_normalized_to_exact_one():
    canon = canonicalize(entries_of("Cs-137, 0.503\nCo-60, 0.5"), "activity fraction")
    assert sum(canon.contents.values()) == pytest.approx(1.0)


def test_percent_mode_normalizes_against_100():
    canon = canonicalize(entries_of("Cs-137, 60\nCo-60, 40"), "activity fraction", frac_as_percent=True)
    assert canon.contents["Cs-137"] == pytest.approx(0.6)
    assert canon.contents["Co-60"] == pytest.approx(0.4)


def test_mass_fraction_and_mole_fraction_anchor_on_distinct_base_units():
    mass_canon = canonicalize(entries_of("Cs-137, 1"), "mass fraction")
    mole_canon = canonicalize(entries_of("Cs-137, 1"), "mole fraction")
    assert mass_canon.library_unit == "g"
    assert mole_canon.library_unit == "mol"


def test_zero_sum_fraction_is_rejected():
    with pytest.raises(ValidationError):
        canonicalize(entries_of("Cs-137, 0\nCo-60, 0"), "activity fraction")


# --- display scaling (engine base unit -> user's chosen display unit) ---


def test_scale_to_display_round_trip():
    for base_unit, display_unit, value in [
        ("Bq", "Ci", 3.7e10),
        ("g", "mg", 2.5),
        ("mol", "mmol", 0.004),
    ]:
        out = scale_to_display(value, base_unit, display_unit)
        back = scale_to_display(out, display_unit, base_unit)
        assert back == pytest.approx(value, rel=1e-9)


def test_scale_to_display_known_anchor():
    assert scale_to_display(3.7e10, "Bq", "Ci") == pytest.approx(1.0)


def test_scale_to_display_passes_through_atoms_and_fractions_unscaled():
    assert scale_to_display(42.0, "num", "num") == 42.0
    assert scale_to_display(0.5, "fraction", "fraction") == 0.5


# --- professional units added 2026-07-03 (dpm, TBq, nCi, pCi, ug, tonne) ---


def test_dpm_input_is_one_sixtieth_of_bq_per_second():
    # 1 dpm = 1 disintegration/min = 1/60 Bq.
    canon = canonicalize(entries_of("Co-60, 60"), "dpm")
    inv = rd.Inventory(canon.contents, units=canon.library_unit)
    assert inv.activities("Bq")["Co-60"] == pytest.approx(1.0, rel=1e-9)


def test_new_activity_units_round_trip_through_bq():
    for label, value in [("TBq", 2.0), ("nCi", 500.0), ("pCi", 1e4), ("dpm", 12345.0)]:
        canon = canonicalize(entries_of(f"Cs-137, {value}"), label)
        inv = rd.Inventory(canon.contents, units=canon.library_unit)
        bq = inv.activities("Bq")["Cs-137"]
        assert scale_to_display(bq, "Bq", label) == pytest.approx(value, rel=1e-9)


def test_tonne_and_microgram_mass_units():
    canon_t = canonicalize(entries_of("U-238, 3"), "t")
    assert canon_t.library_unit == "t"
    # 1 t = 1e6 g; display back from grams must recover the tonne value.
    assert scale_to_display(3.0e6, "g", "t") == pytest.approx(3.0, rel=1e-12)
    assert scale_to_display(5.0e-6, "g", "µg") == pytest.approx(5.0, rel=1e-9)


def test_microsign_labels_display_without_error():
    # Regression: the 'µ' in the dropdown labels is the micro sign (U+00B5),
    # not the library's 'u'/'μ' -- scale_to_display must resolve the label
    # to the library unit or the converter raises ValueError.
    assert scale_to_display(3.7e10, "Bq", "µCi") == pytest.approx(1.0e6, rel=1e-6)
    assert scale_to_display(1.0, "g", "µg") == pytest.approx(1.0e6, rel=1e-9)
