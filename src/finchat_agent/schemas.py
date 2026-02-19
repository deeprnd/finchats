"""Typed schemas for plans, state, and domain errors."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class OperationEnum(str, Enum):
    GET = "GET"
    COMPARE = "COMPARE"
    TREND = "TREND"
    CHANGE = "CHANGE"
    RANK = "RANK"
    FILTER = "FILTER"


class MetricEnum(str, Enum):
    REVENUE = "revenue"
    GROSS_PROFIT = "gross_profit"
    OPERATING_INCOME = "operating_income"
    EBITDA = "ebitda"
    FREE_CASH_FLOW = "free_cash_flow"
    GROSS_MARGIN = "gross_margin"
    OPERATING_MARGIN = "operating_margin"
    FCF_MARGIN = "fcf_margin"
    PE_TTM = "pe_ttm"
    P_FCF_TTM = "p_fcf_ttm"


class TimeframeType(str, Enum):
    LATEST = "LATEST"
    LAST_N_QUARTERS = "LAST_N_QUARTERS"
    SPECIFIC_QUARTER = "SPECIFIC_QUARTER"


class Timeframe(BaseModel):
    kind: TimeframeType
    n_quarters: int | None = None
    quarter: str | None = None

    @field_validator("n_quarters")
    @classmethod
    def validate_n_quarters(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("n_quarters must be positive")
        return value


class QueryPlan(BaseModel):
    operation: OperationEnum
    tickers: list[str] = Field(default_factory=list)
    metrics: list[MetricEnum]
    timeframe: Timeframe
    output_format: Literal["brief", "table", "bullets"] = "brief"
    change_basis: Literal["qoq", "yoy"] = "qoq"


class PlanValidationError(Exception):
    """Raised when a generated query plan violates policy constraints."""


class ExecutionError(Exception):
    """Raised when execution fails due to missing facts or unsupported requests."""


class AgentState(BaseModel):
    user_query: str
    plan: QueryPlan | None = None
    result_rows: list[dict[str, Any]] = Field(default_factory=list)
    answer_text: str = ""
    evaluation: dict[str, Any] = Field(default_factory=dict)
