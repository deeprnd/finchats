"""Thin tool wrappers around deterministic operations."""

from __future__ import annotations

from .facts_store import FactsStore
from .operations import change_metric, compare_metrics, get_metrics, rank_tickers, trend_metric
from .schemas import QueryPlan


class FinancialTools:
    """Provides operation-specific methods that execute against canonical facts data."""

    def __init__(self, store: FactsStore) -> None:
        self.store = store

    def execute(self, plan: QueryPlan) -> list[dict[str, object]]:
        df = self.store.load()
        metrics = [m.value for m in plan.metrics]

        if plan.operation.value == "GET":
            out = get_metrics(df, plan.tickers, metrics, plan.timeframe.kind.value, plan.timeframe.n_quarters, plan.timeframe.quarter)
        elif plan.operation.value == "COMPARE":
            out = compare_metrics(df, plan.tickers, metrics, plan.timeframe.kind.value, plan.timeframe.n_quarters, plan.timeframe.quarter)
        elif plan.operation.value == "TREND":
            out = trend_metric(df, plan.tickers, metrics, n_quarters=plan.timeframe.n_quarters or 4)
        elif plan.operation.value == "CHANGE":
            out = change_metric(df, plan.tickers, metrics, basis=plan.change_basis)
        elif plan.operation.value == "RANK":
            out = rank_tickers(df, plan.tickers, metrics[0], plan.timeframe.kind.value, plan.timeframe.n_quarters, plan.timeframe.quarter)
        else:
            raise ValueError(f"Unsupported operation in MVP: {plan.operation.value}")

        return out.to_dict(orient="records")
