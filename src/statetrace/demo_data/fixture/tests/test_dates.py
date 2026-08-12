from datetime import date

from calendar_edge import next_calendar_day


def test_advances_within_a_month() -> None:
    assert next_calendar_day(date(2026, 8, 11)) == date(2026, 8, 12)


def test_advances_from_month_end() -> None:
    assert next_calendar_day(date(2026, 8, 31)) == date(2026, 9, 1)


def test_advances_from_year_end() -> None:
    assert next_calendar_day(date(2026, 12, 31)) == date(2027, 1, 1)


def test_handles_leap_day() -> None:
    assert next_calendar_day(date(2024, 2, 29)) == date(2024, 3, 1)
