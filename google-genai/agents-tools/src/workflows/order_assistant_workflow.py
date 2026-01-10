"""Order assistant agent using LangGraph + Google Gemini."""

import os
from typing import Dict, List, Literal

from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

# ======================================================
# LLM configuration
# ======================================================
model = os.getenv("MODEL", "gemini-1.5-pro")
api_key = os.getenv("GOOGLE_API_KEY", "")

llm = ChatGoogleGenerativeAI(
    model=model,
    api_key=api_key,
    temperature=0.2,
)


# ======================================================
# Helper to save graph
# ======================================================
def save_graph(graph: CompiledStateGraph, path: str = "graph.png") -> None:
    """Saves the workflow graph as a Mermaid PNG."""

    png = graph.get_graph().draw_mermaid_png()
    with open(path, "wb") as f:
        f.write(png)


# ======================================================
# Tools (realistic domain)
# ======================================================
@tool
def get_order_status(order_id: str) -> str:
    """Returns the status of an order."""

    fake_db = {
        "123": "Order shipped",
        "456": "Order processing",
        "789": "Order delivered",
    }
    return fake_db.get(order_id, "Order not found")


@tool
def calculate_order_total(items: List[Dict[str, float]]) -> float:
    """
    Calculates the total value of an order.

    Args:
        items: List of items with price and quantity
    """
    total = 0.0
    for item in items:
        total += item["price"] * item["quantity"]
    return total


@tool
def apply_discount(total: float, coupon: str) -> float:
    """Applies a discount coupon to the order total."""
    if coupon == "PROMO10":
        return total * 0.9
    if coupon == "PROMO20":
        return total * 0.8
    return total


# ======================================================
# Tool binding
# ======================================================
tools = [get_order_status, calculate_order_total, apply_discount]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)


# ======================================================
# Nodes
# ======================================================
def llm_call(state: MessagesState):
    """LLM decides what to do next."""

    system_prompt = SystemMessage(
        content=(
            "You are an e-commerce assistant. "
            "You help users by checking order status, "
            "calculating order totals, and applying discounts. "
            "Use tools whenever necessary."
        )
    )

    response = llm_with_tools.invoke([system_prompt] + state["messages"])

    return {"messages": [response]}


def tool_node(state: MessagesState):
    """Executes tool calls made by the LLM."""

    results = []

    last_message = state["messages"][-1]

    for tool_call in last_message.tool_calls:
        tool_func = tools_by_name[tool_call["name"]]
        observation = tool_func.invoke(tool_call["args"])

        results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": results}


# ======================================================
# Conditional routing
# ======================================================
def should_continue(state: MessagesState) -> Literal["tool_node", "end"]:
    """
    Decide whether to continue to the tool node or end
    based on whether the LLM made a tool call.
    """

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tool_node"

    return END


# ======================================================
# Build and run
# ======================================================
def order_agent_workflow():
    """Builds and runs the order assistant workflow."""

    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("llm_call", llm_call)
    graph_builder.add_node("tool_node", tool_node)

    graph_builder.add_edge(START, "llm_call")
    graph_builder.add_conditional_edges(
        "llm_call", should_continue, ["tool_node", END]
    )
    graph_builder.add_edge("tool_node", "llm_call")

    agent = graph_builder.compile()

    save_graph(agent, "order_agent_graph.png")

    # Example invocation
    messages = [HumanMessage(content=("What is the status of order 123?"))]

    result = agent.invoke({"messages": messages})

    for msg in result["messages"]:
        msg.pretty_print()
