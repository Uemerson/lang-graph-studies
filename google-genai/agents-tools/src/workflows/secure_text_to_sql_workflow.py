"""
Secure Text-to-SQL with LangGraph and Google Gemini LLM
------------------------------------------------------

Key principles:
- LLM is used ONLY for intent extraction
- All validation is hard-coded
- SQL generation is fully deterministic
- Enumerations strictly control the attack surface
"""

import os
from typing import Dict, Literal, Optional, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

# =============================
# Semantic Model (Source of Truth)
# =============================

SEMANTIC_MODEL = {
    "table": "sales_summary",
    "metrics": {
        "revenue": "SUM(total_amount)",
        "orders": "COUNT(order_id)",
    },
    "dimensions": {
        "country": "country",
        "month": "DATE_TRUNC(order_date, MONTH)",
        "product": "product_name",
    },
    "time_ranges": {
        "last_month": (
            "order_date >= DATE_TRUNC(CURRENT_DATE - INTERVAL 1 MONTH, MONTH) "
            "AND order_date < DATE_TRUNC(CURRENT_DATE, MONTH)"
        ),
        "this_month": ("order_date >= DATE_TRUNC(CURRENT_DATE, MONTH)"),
        "last_3_months": (
            "order_date >= DATE_TRUNC(CURRENT_DATE - INTERVAL 3 MONTH, MONTH)"
        ),
    },
}


# =============================
# LangGraph State
# =============================


class QueryState(TypedDict, total=False):
    """State for the text-to-SQL workflow."""

    question: str
    intent: Dict
    sql: str


# =============================
# Intent Schema (LLM Output)
# =============================


class Intent(BaseModel):
    """Schema for structured output parsing."""

    metric: Literal["revenue", "orders"] = Field(
        description="The requested metric. Use one of: revenue, orders."
    )

    dimension: Literal["country", "month", "product"] = Field(
        description=(
            "The aggregation dimension. Use country, month, " "or product."
        )
    )

    time_range: Literal["last_month", "this_month", "last_3_months"] = Field(
        description=(
            "The time range. Always return one of: "
            "last_month, this_month, last_3_months (snake_case)."
        )
    )

    country: Optional[str] = Field(
        default=None,
        description=(
            "Optional country filter. Only include if explicitly mentioned."
        ),
    )


# =============================
# LLM Setup
# =============================

model_name = os.getenv("MODEL", "gemini-1.5-pro")
api_key = os.getenv("GOOGLE_API_KEY", "")

model = ChatGoogleGenerativeAI(
    model=model_name,
    api_key=api_key,
    temperature=0,
)

intent_structured_model = model.with_structured_output(
    schema=Intent.model_json_schema(),
    method="json_schema",
    include_raw=True,
)


# =============================
# Node 1 — Interpret Question
# =============================


def interpret_question(state: QueryState) -> Dict:
    """
    Uses the LLM ONLY to map the user question into a constrained intent.
    """
    response = intent_structured_model.invoke(state["question"])
    return {"intent": response["parsed"]}


# =============================
# Node 2 — Hard Validation
# =============================


def validate_intent(state: QueryState) -> QueryState:
    """
    Never trust the LLM output blindly.
    """
    intent = state["intent"]

    if intent["metric"] not in SEMANTIC_MODEL["metrics"]:
        raise ValueError("Invalid metric")

    if intent["dimension"] not in SEMANTIC_MODEL["dimensions"]:
        raise ValueError("Invalid dimension")

    if intent["time_range"] not in SEMANTIC_MODEL["time_ranges"]:
        raise ValueError("Invalid time range")

    return state


# =============================
# Node 3 — Deterministic SQL Builder
# =============================


def build_sql(state: QueryState) -> Dict:
    """
    Builds SQL deterministically.
    No LLM involvement here.
    """
    intent = state["intent"]

    metric_sql = SEMANTIC_MODEL["metrics"][intent["metric"]]
    dimension_sql = SEMANTIC_MODEL["dimensions"][intent["dimension"]]
    time_filter = SEMANTIC_MODEL["time_ranges"][intent["time_range"]]

    where_clauses = [time_filter]

    if intent.get("country"):
        where_clauses.append("country = %(country)s")

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT
            {dimension_sql} AS dimension,
            {metric_sql} AS value
        FROM {SEMANTIC_MODEL["table"]}
        WHERE {where_sql}
        GROUP BY 1
        ORDER BY value DESC
        LIMIT 100
    """.strip()

    return {"sql": sql}


# =============================
# Node 4 — Final Response
# =============================


def final_response(state: QueryState) -> Dict:
    """
    Final node: returns safe SQL only.
    """
    return {
        "sql": state["sql"],
        "message": "Query generated using a safe, controlled approach.",
    }


# =============================
# Build and Run LangGraph
# =============================


def secure_text_to_sql_workflow():
    """Builds and runs the secure text-to-SQL LangGraph workflow."""

    graph = StateGraph(QueryState)

    graph.add_node("interpret", interpret_question)
    graph.add_node("validate", validate_intent)
    graph.add_node("build_sql", build_sql)
    graph.add_node("final", final_response)

    graph.set_entry_point("interpret")

    graph.add_edge("interpret", "validate")
    graph.add_edge("validate", "build_sql")
    graph.add_edge("build_sql", "final")

    app = graph.compile()

    questions = [
        "What was the revenue by country last month?",
        # "How many orders by product this month?",
        # "Show revenue by month for the last 3 months",
        # "Top 5 countries by revenue last month",
        # "Revenue by product in Brazil last month",
    ]

    for q in questions:
        print("\nQUESTION:", q)
        result = app.invoke({"question": q})
        print(result["sql"])


if __name__ == "__main__":
    secure_text_to_sql_workflow()
