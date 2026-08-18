"""Structural look-ahead prevention.

The claim "my backtest has no look-ahead bias" is worth nothing when it rests on
the author's discipline, because look-ahead is not a mistake you make once -- it
is a mistake you make at 11pm six months in, in a helper function, and never
notice. So the guard here is structural: the data access object physically
refuses to return rows dated after the cursor. A strategy CANNOT see the future,
because there is no method that returns it.

Three doors look-ahead comes through, and what closes each:

  1. Reading a bar you could not have had yet (today's close at today's open).
     Closed by: PointInTimeView.history() slicing at the cursor, exclusive of
     any row after it.
  2. Executing at the price that generated the signal. Closed by: signals
     computed at t are filled at t+1's open, enforced in engine/backtest.py, not
     left to the strategy.
  3. Restated or survivorship-filtered data -- a universe assembled today,
     applied to 2015. Closed by: nothing here. This one is a data-sourcing
     problem, it is NOT solved in this repo, and docs/BIAS_AUDIT.md says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class LookAheadError(AssertionError):
    """Raised when code asks for data it could not have had at the cursor."""


class PointInTimeView:
    """A read-only window onto a price panel, clamped to `cursor`.

    There is deliberately no `df` property, no `.raw`, and no way to widen the
    window from inside a strategy. The only escape is `_unsafe_full_frame`,
    which exists so the leak test can plant a leak on purpose, and whose name is
    the documentation.
    """

    __slots__ = ("_frame", "_cursor", "_i")

    def __init__(self, frame: pd.DataFrame, cursor: pd.Timestamp):
        if not frame.index.is_monotonic_increasing:
            raise ValueError("price frame must be sorted by timestamp")
        self._frame = frame
        self._cursor = cursor
        self._i = int(frame.index.searchsorted(cursor, side="right"))

    @property
    def cursor(self) -> pd.Timestamp:
        return self._cursor

    def history(self, column: str, lookback: int | None = None) -> np.ndarray:
        """Values up to and including the cursor. Never beyond it."""
        s = self._frame[column].to_numpy()[: self._i]
        return s if lookback is None else s[-lookback:]

    def last(self, column: str) -> float:
        h = self.history(column, 1)
        if not len(h):
            raise LookAheadError("no history at or before {}".format(self._cursor))
        return float(h[0])

    def at(self, column: str, when: pd.Timestamp) -> float:
        if when > self._cursor:
            raise LookAheadError(
                "asked for {} at {}, cursor is {} -- that value does not exist yet"
                .format(column, when, self._cursor))
        return float(self._frame.loc[:when, column].iloc[-1])

    def _unsafe_full_frame(self) -> pd.DataFrame:
        """ONLY for the planted-leak test. Using this in a strategy is the bug
        the harness is built to catch."""
        return self._frame


def load_panel(csv_or_frame) -> pd.DataFrame:
    df = csv_or_frame if isinstance(csv_or_frame, pd.DataFrame) else pd.read_csv(
        csv_or_frame, parse_dates=["date"], index_col="date")
    df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError("duplicate timestamps in price panel")
    return df
