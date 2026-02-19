from __future__ import annotations

from pathlib import Path

import pandas as pd

from finchat_agent.facts_store import FactsStore
from finchat_agent.graph import build_graph, score_answer_fields
from finchat_agent.llm import MockLLM
from finchat_agent.schemas import OperationEnum


def _write_facts(path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "ticker": "NVDA",
                "quarter": "2024Q3",
                "report_date": "2024-10-31",
                "revenue": 100.0,
                "gross_profit": 60.0,
                "operating_income": 40.0,
                "ebitda": 45.0,
                "free_cash_flow": 35.0,
                "gross_margin": 0.60,
                "operating_margin": 0.40,
                "fcf_margin": 0.35,
                "pe_ttm": 35.0,
                "p_fcf_ttm": 28.0,
            },
            {
                "ticker": "NVDA",
                "quarter": "2024Q4",
                "report_date": "2025-01-31",
                "revenue": 120.0,
                "gross_profit": 75.0,
                "operating_income": 50.0,
                "ebitda": 55.0,
                "free_cash_flow": 42.0,
                "gross_margin": 0.625,
                "operating_margin": 0.417,
                "fcf_margin": 0.35,
                "pe_ttm": 38.0,
                "p_fcf_ttm": 30.0,
            },
            {
                "ticker": "AMD",
                "quarter": "2024Q4",
                "report_date": "2025-01-31",
                "revenue": 80.0,
                "gross_profit": 38.4,
                "operating_income": 16.0,
                "ebitda": 20.0,
                "free_cash_flow": 12.0,
                "gross_margin": 0.48,
                "operating_margin": 0.20,
                "fcf_margin": 0.15,
                "pe_ttm": 29.0,
                "p_fcf_ttm": 24.0,
            },
        ]
    )
    df.to_parquet(path, index=False)


def test_compare_and_trend_queries(tmp_path: Path) -> None:
    facts_path = tmp_path / "facts.parquet"
    _write_facts(facts_path)

    app = build_graph(FactsStore(facts_path), MockLLM())

    compare_state = app.invoke(
        {"user_query": "Compare NVDA and AMD gross_margin using the latest quarter"}
    )
    compare_plan = compare_state["plan"]
    assert compare_plan.operation == OperationEnum.COMPARE
    assert compare_plan.timeframe.kind.value == "LATEST"
    compare_rows = compare_state["result_rows"]
    assert {r["ticker"] for r in compare_rows} == {"NVDA", "AMD"}
    assert {r["quarter"] for r in compare_rows} == {"2024Q4"}
    row_by_ticker = {r["ticker"]: r for r in compare_rows}
    assert row_by_ticker["NVDA"]["gross_margin"] == 0.625
    assert row_by_ticker["AMD"]["gross_margin"] == 0.48
    assert "NVDA" in compare_state["answer_text"]
    assert "AMD" in compare_state["answer_text"]
    assert "gross_margin" in compare_state["answer_text"]
    assert "2024Q4" in compare_state["answer_text"]

    trend_state = app.invoke(
        {"user_query": "Show NVDA free_cash_flow trend over the last 2 quarters"}
    )
    trend_plan = trend_state["plan"]
    assert trend_plan.operation == OperationEnum.TREND
    assert trend_plan.timeframe.kind.value == "LAST_N_QUARTERS"
    assert trend_plan.timeframe.n_quarters == 2
    trend_rows = trend_state["result_rows"]
    assert [r["quarter"] for r in trend_rows] == ["2024Q3", "2024Q4"]
    assert [r["free_cash_flow"] for r in trend_rows] == [35.0, 42.0]
    assert "NVDA" in trend_state["answer_text"]
    assert "free_cash_flow" in trend_state["answer_text"]
    assert "2024Q3" in trend_state["answer_text"]
    assert "2024Q4" in trend_state["answer_text"]


def test_evaluation_hook_fields() -> None:
    answer = "NVDA AMD 2024Q4 gross_margin operating_margin fcf_margin"
    scores = score_answer_fields(
        answer_text=answer,
        required_tickers=["NVDA", "AMD"],
        required_quarters=["2024Q4"],
        required_margins=["gross_margin", "operating_margin", "fcf_margin"],
    )
    assert scores == {
        "tickers_present": True,
        "quarters_present": True,
        "margins_present": True,
    }
