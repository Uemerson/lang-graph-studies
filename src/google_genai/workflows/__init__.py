"""Workflows module initialization."""

from .math_workflow import math_workflow
from .order_assistant_workflow import order_agent_workflow
from .secure_text_to_sql_workflow import secure_text_to_sql_workflow
from .structured_output import structured_output_workflow

__all__ = [
    "math_workflow",
    "order_agent_workflow",
    "secure_text_to_sql_workflow",
    "structured_output_workflow",
]
