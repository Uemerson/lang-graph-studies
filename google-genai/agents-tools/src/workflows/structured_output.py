"""Workflow demonstrating structured output parsing with Google Gemini."""

import os
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel


class Feedback(BaseModel):
    """Schema for structured output parsing."""

    sentiment: Literal["positive", "neutral", "negative"]
    summary: str


model_name = os.getenv("MODEL", "gemini-1.5-pro")
api_key = os.getenv("GOOGLE_API_KEY", "")

model = ChatGoogleGenerativeAI(
    model=model_name,
    api_key=api_key,
    temperature=0,
)
structured_model = model.with_structured_output(
    schema=Feedback.model_json_schema(), method="json_schema"
)


def structured_output_workflow():
    """Workflow demonstrating structured output parsing."""
    response = structured_model.invoke("The new UI is great!")
    sentiment = response["sentiment"]  # "positive"
    summary = response["summary"]  # "The user expresses positive..."

    print("response:", response)
    print(sentiment, summary)
