"""
Multi LLM Main Entry Point
"""

import argparse

from src.google_genai.workflows import (
    math_workflow,
    order_assistant_workflow,
    secure_text_to_sql_workflow,
    structured_output_workflow,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi LLM")

    parser.add_argument(
        "--llm",
        required=True,
        choices=["google", "aws", "openai"],
        help="LLM provider to use",
    )

    parser.add_argument(
        "--workflow",
        required=True,
        choices=[
            "math_workflow",
            "order_assistant_workflow",
            "secure_text_to_sql_workflow",
            "structured_output_workflow",
        ],
        help="Workflow to execute",
    )

    args = parser.parse_args()

    if args.llm == "google":
        if args.workflow == "math_workflow":
            math_workflow()
        elif args.workflow == "order_assistant_workflow":
            order_assistant_workflow()
        elif args.workflow == "secure_text_to_sql_workflow":
            secure_text_to_sql_workflow()
        elif args.workflow == "structured_output_workflow":
            structured_output_workflow()
