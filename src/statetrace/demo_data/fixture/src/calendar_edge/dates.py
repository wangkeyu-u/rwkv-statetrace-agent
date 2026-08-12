"""Calendar operations with one intentional teaching defect."""

import calendar
from datetime import date


def next_calendar_day(value: date) -> date:
    """Return the day after ``value`` (intentionally wrong at month end)."""

    days_in_month = calendar.monthrange(value.year, value.month)[1]
    if value.day < days_in_month:
        return value.replace(day=value.day + 1)

    return value.replace(day=1)
