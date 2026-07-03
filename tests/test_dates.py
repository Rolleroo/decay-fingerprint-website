"""Date-mode helper tests (2026-07-03)."""

from datetime import date

import pytest

from app.dates import (
    SECONDS_PER_DAY,
    DateError,
    date_from_age,
    forward_interval_seconds,
    interval_seconds,
)

YEAR_S = 86400.0 * 365.25


def test_interval_seconds_one_day():
    assert interval_seconds(date(2026, 1, 1), date(2026, 1, 2)) == SECONDS_PER_DAY


def test_interval_seconds_is_signed():
    assert interval_seconds(date(2026, 1, 2), date(2026, 1, 1)) == -SECONDS_PER_DAY


def test_interval_spans_leap_day():
    # 2024 is a leap year: Feb has 29 days, so Jan 1 -> Mar 1 is 60 days.
    assert interval_seconds(date(2024, 1, 1), date(2024, 3, 1)) == 60 * SECONDS_PER_DAY


def test_forward_interval_rejects_reversed_dates():
    with pytest.raises(DateError):
        forward_interval_seconds(date(2026, 6, 1), date(2026, 1, 1))


def test_forward_interval_rejects_equal_dates():
    with pytest.raises(DateError):
        forward_interval_seconds(date(2026, 1, 1), date(2026, 1, 1))


def test_forward_interval_ok_message_names_the_field():
    try:
        forward_interval_seconds(date(2026, 6, 1), date(2026, 1, 1), what="measurement date")
    except DateError as exc:
        assert "measurement date" in str(exc)
    else:
        raise AssertionError("expected DateError")


def test_date_from_age_round_trips_a_whole_year():
    reference = date(2026, 7, 3)
    origin = date_from_age(reference, 365 * SECONDS_PER_DAY)
    assert origin == date(2025, 7, 3)


def test_date_from_age_matches_a_known_interval():
    # 30 years back from a fixed date, to the nearest day.
    reference = date(2026, 7, 3)
    origin = date_from_age(reference, 30 * YEAR_S)
    assert (reference - origin).days == round(30 * YEAR_S / SECONDS_PER_DAY)
