"""Deterministic financial operations over the quarterly facts table."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .schemas import ExecutionError


def _resolve_quarters(df: pd.DataFrame, timeframe_kind: str, n_quarters: int | None, quarter: str | None) -> list[str]:
    quarters = (
        df[["quarter", "report_date"]]
        .drop_duplicates()
        .sort_values("report_date")
        .reset_index(drop=True)
    )
    if quarters.empty:
        raise ExecutionError("No quarter data available")
    if timeframe_kind == "LATEST":
        return [str(quarters.iloc[-1]["quarter"])]
    if timeframe_kind == "LAST_N_QUARTERS":
        if not n_quarters:
            raise ExecutionError("LAST_N_QUARTERS requires n_quarters")
        return quarters.tail(n_quarters)["quarter"].astype(str).tolist()
    if timeframe_kind == "SPECIFIC_QUARTER":
        if not quarter:
            raise ExecutionError("SPECIFIC_QUARTER requires quarter")
        if quarter not in quarters["quarter"].astype(str).tolist():
            raise ExecutionError(f"Quarter {quarter} not found")
        return [quarter]
    raise ExecutionError(f"Unsupported timeframe: {timeframe_kind}")


def _subset(df: pd.DataFrame, tickers: Iterable[str], quarters: Iterable[str], columns: list[str]) -> pd.DataFrame:
    base_cols = ["ticker", "quarter", "report_date"]
    mask = df["quarter"].isin(list(quarters))
    ticker_list = [t.upper() for t in tickers]
    if ticker_list:
        mask &= df["ticker"].isin(ticker_list)
    out = df.loc[mask, base_cols + columns].sort_values(["quarter", "ticker"]).reset_index(drop=True)
    if out.empty:
        raise ExecutionError("No rows matched the query")
    return out


def get_metrics(df: pd.DataFrame, tickers: list[str], metrics: list[str], timeframe_kind: str, n_quarters: int | None = None, quarter: str | None = None) -> pd.DataFrame:
    quarters = _resolve_quarters(df, timeframe_kind, n_quarters, quarter)
    return _subset(df, tickers, quarters, metrics)


def compare_metrics(df: pd.DataFrame, tickers: list[str], metrics: list[str], timeframe_kind: str, n_quarters: int | None = None, quarter: str | None = None) -> pd.DataFrame:
    return get_metrics(df, tickers, metrics, timeframe_kind, n_quarters, quarter)


def trend_metric(df: pd.DataFrame, tickers: list[str], metrics: list[str], n_quarters: int) -> pd.DataFrame:
    return get_metrics(df, tickers, metrics, "LAST_N_QUARTERS", n_quarters=n_quarters)


def change_metric(df: pd.DataFrame, tickers: list[str], metrics: list[str], basis: str = "qoq") -> pd.DataFrame:
    if basis != "qoq":
        raise ExecutionError("Only QoQ change is supported in MVP")
    base = get_metrics(df, tickers, metrics, "LAST_N_QUARTERS", n_quarters=2)
    rows: list[dict[str, object]] = []
    for ticker, group in base.sort_values("report_date").groupby("ticker"):
        if len(group) < 2:
            continue
        prev_row = group.iloc[-2]
        curr_row = group.iloc[-1]
        row: dict[str, object] = {
            "ticker": ticker,
            "quarter": str(curr_row["quarter"]),
            "prior_quarter": str(prev_row["quarter"]),
        }
        for metric in metrics:
            row[f"{metric}_delta"] = float(curr_row[metric]) - float(prev_row[metric])
        rows.append(row)
    if not rows:
        raise ExecutionError("Insufficient rows to compute change")
    return pd.DataFrame(rows)


def rank_tickers(df: pd.DataFrame, tickers: list[str], metric: str, timeframe_kind: str, n_quarters: int | None = None, quarter: str | None = None) -> pd.DataFrame:
    rows = get_metrics(df, tickers, [metric], timeframe_kind, n_quarters=n_quarters, quarter=quarter)
    latest = rows.sort_values("report_date").groupby("ticker", as_index=False).tail(1)
    ranked = latest.sort_values(metric, ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked[metric].rank(method="dense", ascending=False).astype(int)
    return ranked[["rank", "ticker", "quarter", metric]]
