"""Calendar operations used by the example billing application."""

from __future__ import annotations

import calendar
from datetime import date


def next_calendar_day(value: date) -> date:
    """Return the calendar day immediately following ``value``.

    This deliberately small implementation makes the fixture deterministic and
    keeps the interesting failure at a month/year boundary.
    """

    days_in_month = calendar.monthrange(value.year, value.month)[1]
    if value.day < days_in_month:
        return value.replace(day=value.day + 1)

    return value.replace(day=1)
