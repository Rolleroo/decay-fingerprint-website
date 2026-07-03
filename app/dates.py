"""Date-mode helpers: turn a pair of dates into the elapsed interval the
engine works in (seconds), and turn a solved age back into a date.

Kept separate and pure so it is testable without the UI. Professionals
work from certificate / reference dates, not bare intervals (Rad Pro,
Nucleonica both lead with date mode), so every tab that takes a time
interval can offer "by date" as an alternative that computes the same
seconds the elapsed-time path already feeds downstream.

Dates are ``datetime.date`` (calendar days, no clock time or timezone):
decay intervals for this tool are days-to-years, where sub-day precision
and DST are irrelevant. One day is treated as exactly 86400 s.
"""

from __future__ import annotations

from datetime import date, timedelta

SECONDS_PER_DAY = 86400.0


class DateError(Exception):
    """Raised when a date pair cannot form a valid forward interval."""


def interval_seconds(start: date, end: date) -> float:
    """Elapsed seconds from ``start`` to ``end`` (negative if end precedes
    start). Callers that require a forward interval should use
    ``forward_interval_seconds`` instead."""
    return (end - start).days * SECONDS_PER_DAY


def forward_interval_seconds(start: date, end: date, *, what: str = "target date") -> float:
    """Elapsed seconds, requiring ``end`` to be on or after ``start``.

    Used where a *forward* interval is mandatory (decay to a later date;
    a measurement date after the origin date). Raises ``DateError`` with a
    professional-facing message if the dates are the wrong way round or
    equal, since a zero/negative decay interval is almost always a
    data-entry slip rather than an intended input.
    """
    seconds = interval_seconds(start, end)
    if seconds <= 0:
        raise DateError(
            f"The {what} ({end.isoformat()}) must be after the reference date "
            f"({start.isoformat()}); got a non-positive interval."
        )
    return seconds


def date_from_age(reference: date, age_s: float) -> date:
    """The date ``age_s`` seconds before ``reference`` -- i.e. turn a solved
    age (Mode A) into an implied origin/production date given the
    measurement date. Rounded to the nearest whole day."""
    days = round(age_s / SECONDS_PER_DAY)
    return reference - timedelta(days=days)
