from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.orchestrai.graph.nodes import (
    create_calendar_event_node,
    final_response_node,
    orchestrator_node,
    send_email_node,
)
from src.orchestrai.graph.state import GraphState

CHECKPOINTER = MemorySaver()


def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("send_email", send_email_node)
    workflow.add_node("create_calendar_event", create_calendar_event_node)
    workflow.add_node("final_response", final_response_node)

    workflow.set_entry_point("orchestrator")

    def decide_after_orchestrator(
        state: GraphState,
    ) -> Literal["send_email", "create_calendar_event", "final_response"]:
        tasks = state.get("pending_tasks", [])
        if "email" in tasks:
            return "send_email"
        if "calendar" in tasks:
            return "create_calendar_event"
        return "final_response"

    def after_email(
        state: GraphState,
    ) -> Literal["create_calendar_event", "final_response"]:
        tasks = state.get("pending_tasks", [])
        if "calendar" in tasks:
            return "create_calendar_event"
        return "final_response"

    workflow.add_conditional_edges(
        "orchestrator",
        decide_after_orchestrator,
        {
            "send_email": "send_email",
            "create_calendar_event": "create_calendar_event",
            "final_response": "final_response",
        },
    )
    workflow.add_conditional_edges(
        "send_email",
        after_email,
        {
            "create_calendar_event": "create_calendar_event",
            "final_response": "final_response",
        },
    )
    workflow.add_edge("create_calendar_event", "final_response")
    workflow.add_edge("final_response", END)

    return workflow.compile(checkpointer=CHECKPOINTER)
