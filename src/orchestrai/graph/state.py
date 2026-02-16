from typing import Annotated, Literal, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.orchestrai.schemas.models import CalendarEvent, EmailContent


class GraphState(TypedDict, total=False):
    """Message-first shared state for the LangGraph multi-agent workflow."""

    messages: Annotated[list[AnyMessage], add_messages]
    service_provider: Literal["gmail", "outlook"]

    orchestrator_decision: Literal[
        "send_email",
        "create_calendar_event",
        "both",
        "ask_for_clarification",
        "none",
    ]
    pending_tasks: list[Literal["email", "calendar"]]
    staged_tasks: list[Literal["email", "calendar"]]
    awaiting_confirmation: bool

    email_details: Optional[EmailContent]
    email_draft: dict
    calendar_details: Optional[CalendarEvent]
    clarification_message: Optional[str]
    recipient_disambiguation: dict
    pending_contact_capture: dict
    final_response: str
