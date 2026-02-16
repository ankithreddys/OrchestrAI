from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.orchestrai.llm.client import get_structured_chat_model
from src.orchestrai.schemas.models import OrchestratorOutput


def _orchestrator_model():
    return get_structured_chat_model(OrchestratorOutput)


def orchestrator_agent_func(conversation_messages: list[BaseMessage]) -> OrchestratorOutput:
    """Plan the next action from recent conversation context."""
    model = _orchestrator_model()
    system_prompt = (
        "You orchestrate an assistant for email and calendar tasks.\n"
        "Use previous messages for context and slot-filling.\n"
        "Return one action: send_email, create_calendar_event, both, ask_for_clarification, or none.\n"
        "If details are missing, use ask_for_clarification with a short prompt.\n"
        "For calendar events, output ISO 8601 timestamps."
    )

    contextual: list[BaseMessage] = [
        m for m in conversation_messages[-14:] if isinstance(m, (HumanMessage, AIMessage))
    ]
    try:
        return model.invoke([SystemMessage(content=system_prompt), *contextual])
    except Exception:
        # Graceful fallback when provider/model returns malformed structured output.
        return OrchestratorOutput(
            action="ask_for_clarification",
            clarification_message=(
                "I could not reliably parse your request. "
                "Please provide recipient, subject, and body in one message."
            ),
        )
