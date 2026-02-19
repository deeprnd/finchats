"""Data access layer for canonical quarterly fundamentals."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "ticker",
    "quarter",
    "report_date",
    "revenue",
    "gross_profit",
    "operating_income",
    "ebitda",
    "free_cash_flow",
    "gross_margin",
    "operating_margin",
    "fcf_margin",
    "pe_ttm",
    "p_fcf_ttm",
}


class FactsStore:
    """Loads and exposes the canonical facts table as a typed pandas DataFrame."""

    def __init__(self, facts_path: Path) -> None:
        self.facts_path = facts_path
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """Load facts table from disk and normalize date/sorting columns."""
        if self._df is None:
            df = pd.read_parquet(self.facts_path)
            missing = REQUIRED_COLUMNS.difference(df.columns)
            if missing:
                raise ValueError(f"facts table missing required columns: {sorted(missing)}")
            df = df.copy()
            df["report_date"] = pd.to_datetime(df["report_date"])  # deterministic ordering
            df["ticker"] = df["ticker"].astype(str).str.upper()
            df = df.sort_values(["ticker", "report_date"]).reset_index(drop=True)
            self._df = df
        return self._df.copy()
