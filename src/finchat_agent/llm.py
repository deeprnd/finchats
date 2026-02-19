"""LLM interfaces; includes an offline mock planner and renderer."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import MetricEnum, OperationEnum, QueryPlan, Timeframe, TimeframeType


METRIC_SYNONYMS: dict[MetricEnum, set[str]] = {
    MetricEnum.REVENUE: {"revenue", "sales"},
    MetricEnum.GROSS_PROFIT: {"gross_profit", "gross profit"},
    MetricEnum.OPERATING_INCOME: {"operating_income", "operating income"},
    MetricEnum.EBITDA: {"ebitda"},
    MetricEnum.FREE_CASH_FLOW: {"free_cash_flow", "free cash flow", "fcf"},
    MetricEnum.GROSS_MARGIN: {"gross_margin", "gross margin"},
    MetricEnum.OPERATING_MARGIN: {"operating_margin", "operating margin"},
    MetricEnum.FCF_MARGIN: {"fcf_margin", "fcf margin"},
    MetricEnum.PE_TTM: {"pe_ttm", "p/e", "pe"},
    MetricEnum.P_FCF_TTM: {"p_fcf_ttm", "p/fcf", "p to fcf"},
}


@dataclass
class MockLLM:
    """Offline mock planner that maps natural language to a structured query plan."""

    def _extract_tickers(self, query: str) -> list[str]:
        tickers = [token.strip(" ,.?") for token in query.split() if token.isupper() and 1 < len(token) <= 6]
        return sorted(set(tickers))

    def _extract_metrics(self, lowered: str) -> list[MetricEnum]:
        metrics: list[MetricEnum] = []
        for metric, aliases in METRIC_SYNONYMS.items():
            if any(alias in lowered for alias in aliases):
                metrics.append(metric)
        if not metrics:
            metrics = [MetricEnum.REVENUE]
        return metrics

    def _infer_operation(self, lowered: str) -> OperationEnum:
        op_keywords: dict[OperationEnum, set[str]] = {
            OperationEnum.COMPARE: {"compare", "vs", "versus", "relative"},
            OperationEnum.TREND: {"trend", "over", "history", "across"},
            OperationEnum.CHANGE: {"change", "delta", "increase", "decrease", "qoq", "yoy"},
            OperationEnum.RANK: {"rank", "top", "highest", "lowest"},
            OperationEnum.FILTER: {"filter", "above", "below", "where"},
            OperationEnum.GET: {"show", "get", "what", "latest"},
        }
        scores = {op: 0 for op in op_keywords}
        tokens = set(lowered.replace("?", " ").replace(",", " ").split())
        for op, words in op_keywords.items():
            scores[op] = sum(1 for word in words if word in tokens or f" {word} " in f" {lowered} ")
        return max(scores, key=scores.get)

    def _infer_timeframe(self, lowered: str) -> Timeframe:
        if "latest" in lowered:
            return Timeframe(kind=TimeframeType.LATEST)
        if "last" in lowered and "quarter" in lowered:
            n = 4
            for token in lowered.split():
                if token.isdigit():
                    n = int(token)
                    break
            return Timeframe(kind=TimeframeType.LAST_N_QUARTERS, n_quarters=n)
        return Timeframe(kind=TimeframeType.LATEST)

    def plan(self, query: str) -> QueryPlan:
        lowered = query.lower()
        return QueryPlan(
            operation=self._infer_operation(lowered),
            tickers=self._extract_tickers(query),
            metrics=self._extract_metrics(lowered),
            timeframe=self._infer_timeframe(lowered),
            output_format="brief",
        )

    def polish(self, query: str, rows: list[dict[str, object]], timeframe_label: str) -> str:
        if not rows:
            return "No matching financial rows were found."
        header = f"Query: {query}\nTimeframe: {timeframe_label}\n"
        lines = [", ".join(f"{k}={v}" for k, v in row.items()) for row in rows]
        return header + "\n".join(lines)
