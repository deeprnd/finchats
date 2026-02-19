"""Runtime configuration for the financial Q&A agent."""

from pathlib import Path

FACTS_PATH = Path("data/facts.parquet")
ALLOWED_TICKERS: set[str] = set()
MAX_LAST_N_QUARTERS = 12
SUPPORTED_OUTPUT_FORMATS = {"brief", "table", "bullets"}
