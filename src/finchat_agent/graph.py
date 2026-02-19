"""LangGraph orchestration for plan -> validate -> route -> execute -> render."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import ALLOWED_TICKERS, MAX_LAST_N_QUARTERS, SUPPORTED_OUTPUT_FORMATS
from .facts_store import FactsStore
from .llm import MockLLM
from .schemas import OperationEnum, PlanValidationError, QueryPlan, TimeframeType
from .tools import FinancialTools
from .tracing import invocation_config, traceable


class GraphState(TypedDict, total=False):
    user_query: str
    plan: QueryPlan
    result_rows: list[dict[str, Any]]
    answer_text: str
    evaluation: dict[str, Any]


def _timeframe_label(plan: QueryPlan) -> str:
    if plan.timeframe.kind == TimeframeType.LATEST:
        return "LATEST"
    if plan.timeframe.kind == TimeframeType.SPECIFIC_QUARTER:
        return plan.timeframe.quarter or "SPECIFIC_QUARTER"
    return f"LAST_{plan.timeframe.n_quarters}_QUARTERS"


def validate_plan(plan: QueryPlan) -> None:
    if plan.output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise PlanValidationError(f"Unsupported output format: {plan.output_format}")
    if ALLOWED_TICKERS and any(t not in ALLOWED_TICKERS for t in plan.tickers):
        raise PlanValidationError("Query contains disallowed ticker")
    if plan.timeframe.kind == TimeframeType.LAST_N_QUARTERS:
        n = plan.timeframe.n_quarters or 0
        if n < 1 or n > MAX_LAST_N_QUARTERS:
            raise PlanValidationError(f"LAST_N_QUARTERS must be in [1, {MAX_LAST_N_QUARTERS}]")
    if plan.operation == OperationEnum.RANK and len(plan.metrics) != 1:
        raise PlanValidationError("RANK requires exactly one metric")


def score_answer_fields(
    answer_text: str,
    required_tickers: list[str],
    required_quarters: list[str],
    required_margins: list[str],
) -> dict[str, bool]:
    """Evaluation hook to score whether required tickers/quarters/margins appear in text."""
    lower = answer_text.lower()
    return {
        "tickers_present": all(t.lower() in lower for t in required_tickers),
        "quarters_present": all(q.lower() in lower for q in required_quarters),
        "margins_present": all(m.lower() in lower for m in required_margins),
    }


def evaluate_answer(plan: QueryPlan, answer_text: str, rows: list[dict[str, Any]]) -> dict[str, bool]:
    quarters = sorted({str(row.get("quarter")) for row in rows if row.get("quarter") is not None})
    required_margins = ["gross_margin", "operating_margin", "fcf_margin"]
    field_scores = score_answer_fields(answer_text, plan.tickers, quarters, required_margins)
    metric_ok = all(m.value.lower() in answer_text.lower() for m in plan.metrics)
    timeframe_ok = _timeframe_label(plan).lower().replace("_", "")[:6] in answer_text.lower().replace("_", "")
    return {
        **field_scores,
        "metrics_present": metric_ok,
        "timeframe_stated": timeframe_ok,
    }


def build_graph(store: FactsStore, llm: MockLLM):
    tools = FinancialTools(store)

    @traceable("plan_node")
    def plan_node(state: GraphState) -> GraphState:
        return {"plan": llm.plan(state["user_query"])}

    @traceable("validate_node")
    def validate_node(state: GraphState) -> GraphState:
        validate_plan(state["plan"])
        return {}

    @traceable("route_node")
    def route_node(state: GraphState) -> GraphState:
        return {}

    @traceable("execute_node")
    def execute_generic_node(state: GraphState) -> GraphState:
        rows = tools.execute(state["plan"])
        return {"result_rows": rows}

    @traceable("render_node")
    def render_node(state: GraphState) -> GraphState:
        label = _timeframe_label(state["plan"])
        answer_text = llm.polish(state["user_query"], state.get("result_rows", []), label)
        return {
            "answer_text": answer_text,
            "evaluation": evaluate_answer(state["plan"], answer_text, state.get("result_rows", [])),
        }

    graph = StateGraph(GraphState)
    graph.add_node("plan", plan_node)
    graph.add_node("validate", validate_node)
    graph.add_node("route", route_node)
    graph.add_node("exec_get", execute_generic_node)
    graph.add_node("exec_compare", execute_generic_node)
    graph.add_node("exec_trend", execute_generic_node)
    graph.add_node("exec_change", execute_generic_node)
    graph.add_node("exec_rank", execute_generic_node)
    graph.add_node("render", render_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "validate")
    graph.add_edge("validate", "route")

    graph.add_conditional_edges(
        "route",
        lambda s: s["plan"].operation.value,
        {
            "GET": "exec_get",
            "COMPARE": "exec_compare",
            "TREND": "exec_trend",
            "CHANGE": "exec_change",
            "RANK": "exec_rank",
        },
    )
    for node in ["exec_get", "exec_compare", "exec_trend", "exec_change", "exec_rank"]:
        graph.add_edge(node, "render")
    graph.add_edge("render", END)
    return graph.compile()


def run_query(user_query: str, facts_path: str = "data/facts.parquet") -> GraphState:
    store = FactsStore(Path(facts_path))
    llm = MockLLM()
    app = build_graph(store, llm)
    config = invocation_config(user_query)
    if config:
        return app.invoke({"user_query": user_query}, config=config)
    return app.invoke({"user_query": user_query})
