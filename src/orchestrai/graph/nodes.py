import logging
import warnings

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langsmith import traceable

log = logging.getLogger("orchestrai.nodes")

# Suppress Pydantic v2 serialization warning triggered when LangChain's AIMessage
# carries a structured-output Pydantic model in its `parsed` field during
# LangSmith tracing / LangGraph checkpointing.  This is cosmetic and does not
# affect functionality.
warnings.filterwarnings(
    "ignore",
    message="Expected `none`",
    category=UserWarning,
    module=r"pydantic.*",
)

from src.orchestrai.agents.orchestrator import orchestrator_agent_func
from src.orchestrai.graph.state import GraphState
from src.orchestrai.llm.client import get_structured_chat_model
from src.orchestrai.providers.calendar import create_calendar_event
from src.orchestrai.providers.email import send_email
from src.orchestrai.schemas.models import (
    AgentResponse,
    CalendarEvent,
    ContactCaptureDetails,
    EmailContent,
    EmailDraftAutofill,
    EmailDraftUpdate,
    ContactLookupIntent,
    MeetingIntent,
    PendingStateRoute,
)
from src.orchestrai.services.contact_directory import is_valid_email, load_contacts, save_contact, search_contacts

EMAIL_SLOT_MODEL = get_structured_chat_model(EmailDraftUpdate, temperature=0)
MEETING_INTENT_MODEL = get_structured_chat_model(MeetingIntent, temperature=0)
EMAIL_AUTOFILL_MODEL = get_structured_chat_model(EmailDraftAutofill, temperature=0.2)
CONTACT_CAPTURE_MODEL = get_structured_chat_model(ContactCaptureDetails, temperature=0)
CONTACT_LOOKUP_MODEL = get_structured_chat_model(ContactLookupIntent, temperature=0)
PENDING_STATE_ROUTER_MODEL = get_structured_chat_model(PendingStateRoute, temperature=0)
RESPONSE_GEN_MODEL = get_structured_chat_model(AgentResponse, temperature=0.4)


@traceable(name="agent.response_generator", run_type="chain")
def _generate_response(situation: str, context: list | None = None, data: dict | None = None) -> str:
    """Generate a natural user-facing response via LLM — replaces every hardcoded string."""
    log.debug("_generate_response  situation=%r  data=%s", situation[:80], data)
    data_section = f"\nData (use these values naturally in your reply): {data}" if data else ""
    prompt = (
        "You are the response-generation agent for OrchestrAI, a multi-agent email & calendar assistant.\n"
        "Produce a concise, friendly, and natural response for the situation below.\n"
        "Rules:\n"
        "- Weave ALL data values (names, emails, dates, etc.) naturally into your sentences — never omit or alter them.\n"
        "- NEVER expose raw JSON, dicts, lists, or key-value pairs like {'key': 'value'} in the response.\n"
        "- NEVER include technical field names like 'query', 'collected', 'still_needed' in the response.\n"
        "- Do NOT invent information the user did not supply.\n"
        "- Keep responses brief (1-3 sentences) unless presenting structured details.\n"
        "- Use a warm, professional tone.\n"
        f"\nSituation: {situation}{data_section}"
    )
    msgs: list = [SystemMessage(content=prompt)]
    if context:
        msgs.extend(context[-6:])
    try:
        result = RESPONSE_GEN_MODEL.invoke(msgs)
        if result.message and result.message.strip():
            return result.message.strip()
    except Exception:
        pass
    return situation


def _latest_human_content(state: GraphState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _context_messages(state: GraphState):
    return [m for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))]


def _extract_email_slots(state: GraphState) -> EmailDraftUpdate:
    context = _context_messages(state)
    existing = state.get("email_draft", {}) or {}
    prompt = (
        "You are an email slot-filling agent.\n"
        "Extract ONLY information the user has EXPLICITLY provided:\n"
        "- intent (send_email/not_email/unknown)\n"
        "- to (recipient names/emails the user stated)\n"
        "- subject (topic or subject line the user mentioned)\n"
        "- body (ONLY if the user wrote actual email content / message text)\n\n"
        "CRITICAL RULES FOR BODY:\n"
        "- NEVER generate, fabricate, or invent body text.\n"
        "- Phrases like 'about X', 'regarding Y', 'updates on Z' describe the TOPIC → put in subject, NOT body.\n"
        "- Body must ONLY contain text the user explicitly dictated as the email message.\n"
        "- If user has not provided explicit message content, body MUST be null.\n\n"
        "Keep existing values unless user updates them.\n"
        f"Existing draft: {existing}"
    )
    try:
        return EMAIL_SLOT_MODEL.invoke([SystemMessage(content=prompt), *context[-14:]])
    except Exception:
        return EmailDraftUpdate(intent="unknown")


def _merge_email_draft(existing: dict, update: EmailDraftUpdate) -> dict:
    draft = dict(existing or {})

    def _is_placeholder_recipient(value: str) -> bool:
        token = (value or "").strip().lower()
        if not token:
            return True
        if token.startswith("<") and token.endswith(">"):
            return True
        return token in {
            "person_name_or_email",
            "<person_name_or_email>",
            "recipient_name_or_email",
            "<recipient_name_or_email>",
            "recipient",
            "<recipient>",
        }

    if update.to:
        cleaned_to = [t.strip() for t in update.to if str(t).strip() and not _is_placeholder_recipient(str(t))]
        # Do not overwrite a good recipient with placeholder outputs.
        if cleaned_to:
            draft["to"] = cleaned_to
    if update.subject and update.subject.strip():
        draft["subject"] = update.subject.strip()
    if update.body and update.body.strip():
        draft["body"] = update.body.strip()
    return draft


def _autofill_email_draft(state: GraphState, draft: dict) -> dict:
    """Infer missing subject from conversation IF the user mentioned a topic.

    Only fills subject when there is a clear topic cue in the conversation
    (e.g. 'about the project', 'regarding the meeting').
    Never fabricates a subject or body from nothing.
    """
    if draft.get("subject"):
        return draft

    context = _context_messages(state)
    prompt = (
        "You are an email drafting assistant.\n"
        "Your ONLY job: check if the user mentioned a specific topic/subject for the email.\n"
        "If they did (e.g. 'about X', 'regarding Y'), generate a short subject line from it.\n"
        "If they did NOT mention any topic, return subject as null. Do NOT invent one.\n"
        "NEVER generate a body.\n"
        f"Current draft: {draft}"
    )
    try:
        gen = EMAIL_AUTOFILL_MODEL.invoke([SystemMessage(content=prompt), *context[-14:]])
        if gen.subject and gen.subject.strip():
            draft["subject"] = gen.subject.strip()
    except Exception:
        pass
    return draft


def _polish_email_draft(state: GraphState, draft: dict) -> dict:
    """Rewrite subject and body into polished, professional email text.

    Called once all three slots (to, subject, body) are filled so the user
    sees an optimised version in the confirmation step.
    """
    context = _context_messages(state)
    prompt = (
        "You are an expert email editor.\n"
        "Rewrite the subject and body below into a clear, professional, and well-structured email.\n"
        "Preserve ALL factual details, names, dates, and intent from the original.\n"
        "Do NOT add information the user did not mention.\n"
        "Keep the tone professional yet natural (not overly formal).\n"
        "Return the improved subject and body.\n"
        f"Current draft: {draft}"
    )
    try:
        polished = EMAIL_AUTOFILL_MODEL.invoke([SystemMessage(content=prompt), *context[-14:]])
        if polished.subject and polished.subject.strip():
            draft["subject"] = polished.subject.strip()
        if polished.body and polished.body.strip():
            draft["body"] = polished.body.strip()
    except Exception:
        pass
    return draft


def _extract_meeting_intent(state: GraphState, email_details: EmailContent | None) -> MeetingIntent:
    if not email_details:
        return MeetingIntent(create_calendar_event=False)

    context = _context_messages(state)
    prompt = (
        "Decide whether the user is asking for a calendar event in addition to an email.\n"
        "If meeting/time/date cues are present, set create_calendar_event=true and fill structured fields.\n"
        "Use ISO 8601 datetimes for start_time/end_time. If end_time is unknown, set it to start+1 hour.\n"
        f"Resolved email recipient(s): {email_details.to}\n"
        f"Email subject: {email_details.subject}\n"
        f"Email body: {email_details.body}"
    )
    try:
        return MEETING_INTENT_MODEL.invoke([SystemMessage(content=prompt), *context[-14:]])
    except Exception:
        return MeetingIntent(create_calendar_event=False)


def _resolve_recipients(email_details: EmailContent):
    def _is_placeholder_recipient(value: str) -> bool:
        token = (value or "").strip().lower()
        if not token:
            return True
        if token.startswith("<") and token.endswith(">"):
            return True
        return token in {
            "person_name_or_email",
            "<person_name_or_email>",
            "recipient_name_or_email",
            "<recipient_name_or_email>",
            "recipient",
            "<recipient>",
        }

    resolved: list[str] = []
    for raw in email_details.to:
        token = (raw or "").strip()
        if not token:
            continue
        if _is_placeholder_recipient(token):
            continue
        if is_valid_email(token):
            resolved.append(token)
            continue
        matches = search_contacts(token)
        if len(matches) == 0:
            return [], None, {"query": token}
        if len(matches) > 1:
            options = [f"{c.name} <{c.email}>" for c in matches[:5]]
            return [], _generate_response(
                f"Multiple contacts matched '{token}'. Present the options and ask the user to pick one.",
                data={"matches": options},
            ), None
        resolved.append(matches[0].email)
    if not resolved:
        return [], _generate_response("No recipient could be resolved. Ask the user to provide a recipient email or name."), None
    return resolved, None, None


def _parse_contact_details(user_text: str) -> tuple[dict | None, str | None]:
    text = (user_text or "").strip()
    if not text:
        return None, _generate_response(
            "User provided no text for contact details. Ask them to provide first name, last name, email, and phone number."
        )

    first_name = ""
    last_name = ""
    email = ""
    phone = ""

    try:
        parsed = CONTACT_CAPTURE_MODEL.invoke(
            [
                SystemMessage(
                    content=(
                        "Extract contact details from user text into first_name, last_name, email, and phone.\n"
                        "Understand natural language and shorthand labels (fname/lname/phno).\n"
                        "Never output label words as names. Example: if user writes 'fname Amogh', first_name must be 'Amogh'.\n"
                        "Return values only if they are real person details."
                    )
                ),
                HumanMessage(content=text),
            ]
        )
        first_name = (parsed.first_name or "").strip()
        last_name = (parsed.last_name or "").strip()
        email = (str(parsed.email) if parsed.email else "").strip()
        phone = (parsed.phone or "").strip()
    except Exception:
        pass

    # Repair obvious extraction mistakes using email local-part.
    label_words = {"fname", "lname", "email", "mail", "phone", "phno", "number", "mobile"}
    if first_name.lower() in label_words:
        first_name = ""
    if last_name.lower() in label_words:
        last_name = ""

    if email and ("@" in email) and (not first_name or not last_name):
        local_part = email.split("@", 1)[0]
        tokens = [t for t in local_part.replace("_", ".").replace("-", ".").split(".") if t]
        if len(tokens) >= 2:
            if not first_name:
                first_name = tokens[0].strip().title()
            if not last_name:
                last_name = tokens[1].strip().title()
        elif len(tokens) == 1 and not first_name:
            first_name = tokens[0].strip().title()

    if not first_name or not last_name or not email or not phone:
        collected = {k: v for k, v in [("first_name", first_name), ("last_name", last_name), ("email", email), ("phone", phone)] if v}
        still_needed = [k for k, v in [("first_name", first_name), ("last_name", last_name), ("email", email), ("phone", phone)] if not v]
        return None, _generate_response(
            "Some contact details are missing. Tell the user which fields are still needed and give a brief example format.",
            data={"collected": collected, "still_needed": still_needed},
        )

    return {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "phone": phone,
    }, None


def _extract_contact_lookup_intent(state: GraphState) -> ContactLookupIntent:
    latest = _latest_human_content(state)
    if not latest:
        return ContactLookupIntent(intent="other")

    prompt = (
        "Identify if user asks to retrieve/show/contact details of a person.\n"
        "Return intent=lookup_contact only for contact lookup requests.\n"
        "Return query as the person name/email being requested."
    )
    try:
        return CONTACT_LOOKUP_MODEL.invoke([SystemMessage(content=prompt), HumanMessage(content=latest)])
    except Exception:
        return ContactLookupIntent(intent="other")


def _lookup_contact_details(query: str) -> tuple[str, bool]:
    q = (query or "").strip()
    if not q:
        return _generate_response("User asked for contact details but didn't specify who. Ask them to provide a name or email."), True

    if is_valid_email(q):
        contacts = [c for c in load_contacts() if c.email.lower() == q.lower()]
    else:
        contacts = search_contacts(q)

    if not contacts:
        return _generate_response(
            f"No contact found for '{q}'. Inform the user.",
            data={"query": q},
        ), False

    if len(contacts) > 1:
        details = [
            {"name": f"{c.first_name} {c.last_name}", "email": c.email, "phone": c.phone or "N/A"}
            for c in contacts[:5]
        ]
        return _generate_response(
            f"Multiple contacts found for '{q}'. Display all their details and let the user know.",
            data={"contacts": details},
        ), True

    c = contacts[0]
    return _generate_response(
        "Display this contact's full details to the user.",
        data={"first_name": c.first_name, "last_name": c.last_name, "email": c.email, "phone": c.phone or "N/A"},
    ), True


@traceable(name="agent.pending_state_router", run_type="chain")
def _route_pending_state(user_text: str, state_type: str, source: str | None = None) -> PendingStateRoute:
    log.info("_route_pending_state  state_type=%s  source=%s  text=%r", state_type, source, user_text[:80] if user_text else "")
    text = (user_text or "").strip()
    if not text:
        return PendingStateRoute(action="other")

    source_hint = source or "email_flow"
    prompt = (
        "You are a routing agent for pending conversation state.\n"
        "Classify the user's message into EXACTLY ONE action.\n\n"
        "AVAILABLE ACTIONS:\n"
        "  confirm     – user agrees / approves / wants to proceed\n"
        "  cancel      – user declines / wants to stop\n"
        "  pause       – user sends a greeting or small-talk while something is pending\n"
        "  greet       – user greets (ONLY in general_conversation state)\n"
        "  create      – user wants to create a new contact (contact_creation_confirmation only)\n"
        "  provide_details – user supplies contact details directly (contact_creation_confirmation only)\n"
        "  alternate_recipient – user supplies a different recipient (contact_creation_confirmation only)\n"
        "  other       – anything that does not fit the above\n\n"
        "STATE-SPECIFIC RULES:\n"
        "  execution_confirmation → ONLY confirm / cancel / pause / other\n"
        "  contact_creation_confirmation → create / cancel / pause / alternate_recipient / provide_details / other\n"
        "  general_conversation → greet / other\n\n"
        "CRITICAL EXAMPLES (follow these exactly):\n"
        "  state=execution_confirmation, input='confirm'       → confirm\n"
        "  state=execution_confirmation, input='yes'           → confirm\n"
        "  state=execution_confirmation, input='send'          → confirm\n"
        "  state=execution_confirmation, input='go ahead'      → confirm\n"
        "  state=execution_confirmation, input='ok'            → confirm\n"
        "  state=execution_confirmation, input='sure'          → confirm\n"
        "  state=execution_confirmation, input='do it'         → confirm\n"
        "  state=execution_confirmation, input='cancel'        → cancel\n"
        "  state=execution_confirmation, input='no'            → cancel\n"
        "  state=execution_confirmation, input='stop'          → cancel\n"
        "  state=execution_confirmation, input='hello'         → pause\n"
        "  state=contact_creation_confirmation, input='yes'    → create\n"
        "  state=contact_creation_confirmation, input='hello'  → pause\n"
        "  state=general_conversation, input='hello'           → greet\n"
        "  state=general_conversation, input='hi, draft email' → other\n\n"
        f"Current state: {state_type}\n"
        f"Conversation source: {source_hint}"
    )
    try:
        result = PENDING_STATE_ROUTER_MODEL.invoke([SystemMessage(content=prompt), HumanMessage(content=text)])
        log.info("_route_pending_state → action=%s", result.action)
        return result
    except Exception:
        log.exception("_route_pending_state LLM call failed")
        return PendingStateRoute(action="other")


def _handle_contact_capture_details(state: GraphState, capture: dict, user_message: str):
    log.info("_handle_contact_capture_details  capture=%s  user=%r", capture, user_message[:80])
    parsed, parse_error = _parse_contact_details(user_message)
    if parse_error:
        return {"pending_tasks": [], "messages": [AIMessage(content=parse_error)]}

    try:
        saved = save_contact(
            first_name=parsed["first_name"],
            last_name=parsed["last_name"],
            email=parsed["email"],
            phone=parsed["phone"],
        )
    except Exception as exc:
        return {"pending_tasks": [], "messages": [AIMessage(content=_generate_response(
            "Failed to save a new contact. Inform the user about the error.",
            data={"error": str(exc)},
        ))]}

    if capture.get("source") == "lookup_only":
        return {
            "pending_tasks": [],
            "pending_contact_capture": {},
            "messages": [
                AIMessage(
                    content=_generate_response(
                        "Contact was saved successfully. Show the user the saved details.",
                        data={"name": saved.name, "email": saved.email, "phone": saved.phone},
                    )
                )
            ],
        }

    query = str(capture.get("query", "")).strip().lower()
    draft = dict(state.get("email_draft", {}) or {})
    recipients = list(draft.get("to", []))
    replaced = False
    updated_to: list[str] = []
    for r in recipients:
        token = str(r).strip().lower()
        if not replaced and token == query:
            updated_to.append(saved.email)
            replaced = True
        else:
            updated_to.append(r)
    if not replaced:
        updated_to.append(saved.email)
    draft["to"] = updated_to
    response = _build_email_flow_response(state, draft)
    msgs = response.get("messages", [])
    response["messages"] = [
        AIMessage(content=_generate_response(
            "Contact saved successfully. Briefly confirm to the user.",
            data={"name": saved.name, "email": saved.email},
        )),
        *msgs,
    ]
    response["pending_contact_capture"] = {}
    return response


def _build_email_flow_response(state: GraphState, merged: dict):
    log.info("_build_email_flow_response  draft=%s", {k: v for k, v in merged.items() if v})
    # ── Early recipient validation ──
    # Check the contact directory as soon as a recipient name is available,
    # BEFORE asking for subject / body.  This avoids collecting all slots
    # only to discover the recipient doesn't exist.
    recipients = merged.get("to") or []
    for raw in recipients:
        token = (raw or "").strip()
        if not token or is_valid_email(token):
            continue
        matches = search_contacts(token)
        if len(matches) == 0:
            return {
                "email_draft": merged,
                "pending_contact_capture": {"query": token, "awaiting_create_confirmation": True},
                "staged_tasks": [],
                "pending_tasks": [],
                "messages": [
                    AIMessage(content=_generate_response(
                        f"Recipient '{token}' was not found in contacts. Ask the user if they want to create a new contact or provide a different recipient.",
                        context=_context_messages(state),
                        data={"query": token},
                    ))
                ],
            }
        if len(matches) > 1:
            options = [f"{c.name} <{c.email}>" for c in matches[:5]]
            return {
                "email_draft": merged,
                "pending_tasks": [],
                "messages": [AIMessage(content=_generate_response(
                    f"Multiple contacts matched '{token}'. Present the options and ask the user to pick one.",
                    data={"matches": options},
                ))],
            }

    missing = [k for k in ("to", "subject", "body") if not merged.get(k)]
    if missing:
        filled = {k: v for k, v in merged.items() if v and k in ("to", "subject", "body")}
        return {
            "email_draft": merged,
            "pending_tasks": [],
            "messages": [AIMessage(content=_generate_response(
                "Collecting email details. Some fields are filled, others are still needed. Acknowledge what's been provided and ask for the remaining fields.",
                context=_context_messages(state),
                data={"collected": filled, "still_needed": missing},
            ))],
        }

    # Polish subject + body before presenting for confirmation
    merged = _polish_email_draft(state, merged)

    # Resolve recipients — at this point they are already validated above,
    # so this mainly converts names to email addresses.
    candidate = EmailContent(to=merged["to"], subject=merged["subject"], body=merged["body"])
    resolved, err, capture_needed = _resolve_recipients(candidate)
    if capture_needed:
        q = str(capture_needed.get("query", "")).strip()
        return {
            "email_draft": merged,
            "pending_contact_capture": {"query": q, "awaiting_create_confirmation": True},
            "staged_tasks": [],
            "pending_tasks": [],
            "messages": [
                AIMessage(content=_generate_response(
                    f"Recipient '{q}' was not found in contacts. Ask the user if they want to create a new contact for this person.",
                    context=_context_messages(state),
                    data={"query": q},
                ))
            ],
        }
    if err:
        return {"email_draft": merged, "staged_tasks": [], "pending_tasks": [], "messages": [AIMessage(content=err)]}

    final_email = EmailContent(to=resolved, subject=candidate.subject, body=candidate.body)
    updates: GraphState = {
        "email_draft": {},
        "pending_contact_capture": {},
        "email_details": final_email,
        "staged_tasks": ["email"],
        "pending_tasks": [],
        "awaiting_confirmation": True,
        "orchestrator_decision": "send_email",
    }
    meeting = _extract_meeting_intent(state, final_email)
    if meeting.create_calendar_event and meeting.title and meeting.start_time and meeting.end_time:
        updates["calendar_details"] = CalendarEvent(
            title=meeting.title,
            start_time=meeting.start_time,
            end_time=meeting.end_time,
            attendees=meeting.attendees or final_email.to,
            location=meeting.location,
            description=meeting.description or final_email.body,
        )
        updates["staged_tasks"] = ["email", "calendar"]
        updates["orchestrator_decision"] = "both"
    updates["messages"] = [AIMessage(content=_summary(updates, context=_context_messages(state)))]
    return updates


def _summary(updates: dict, context: list | None = None) -> str:
    data = {}
    if updates.get("email_details"):
        e = updates["email_details"]
        data["email"] = {"to": ", ".join(e.to), "subject": e.subject, "body": e.body}
    if updates.get("calendar_details"):
        c = updates["calendar_details"]
        data["calendar"] = {"title": c.title, "start": c.start_time.isoformat(), "end": c.end_time.isoformat()}
    return _generate_response(
        "Present the complete email and/or calendar event details for user confirmation. "
        "Show every field clearly. Ask the user to reply confirm to execute or cancel to abort.",
        context=context,
        data=data,
    )


@traceable(name="node.orchestrator", run_type="chain")
def orchestrator_node(state: GraphState):
    user_message = _latest_human_content(state)
    log.info("──── orchestrator_node  user=%r  awaiting_conf=%s  pending_capture=%s  draft=%s",
             user_message[:80] if user_message else "",
             state.get("awaiting_confirmation"),
             bool(state.get("pending_contact_capture")),
             bool(state.get("email_draft")))
    if not user_message:
        return {"messages": [AIMessage(content=_generate_response(
            "User sent an empty message. Ask them to provide a request for email or calendar tasks.",
            context=_context_messages(state),
        ))], "pending_tasks": []}

    if state.get("awaiting_confirmation"):
        decision = _route_pending_state(user_message, state_type="execution_confirmation")
        if decision.action == "confirm":
            return {"awaiting_confirmation": False, "pending_tasks": state.get("staged_tasks", [])}
        if decision.action == "cancel":
            return {
                "awaiting_confirmation": False,
                "staged_tasks": [],
                "pending_tasks": [],
                "messages": [AIMessage(content=_generate_response(
                    "User cancelled the pending email/calendar action. Acknowledge the cancellation.",
                    context=_context_messages(state),
                ))],
            }
        if decision.action == "pause":
            return {
                "pending_tasks": [],
                "messages": [AIMessage(content=_generate_response(
                    "User sent an ambiguous message while awaiting confirmation. Gently remind them to confirm or cancel.",
                    context=_context_messages(state),
                ))],
            }
        return {"pending_tasks": [], "messages": [AIMessage(content=_generate_response(
            "User's response doesn't clearly indicate confirm or cancel. Ask them to confirm or cancel the pending action.",
            context=_context_messages(state),
        ))]}

    if state.get("pending_contact_capture"):
        capture = state.get("pending_contact_capture", {}) or {}

        if capture.get("awaiting_create_confirmation"):
            decision = _route_pending_state(
                user_message,
                state_type="contact_creation_confirmation",
                source=capture.get("source"),
            )

            if decision.action == "provide_details":
                capture_next = dict(capture)
                capture_next["awaiting_create_confirmation"] = False
                return _handle_contact_capture_details(state, capture_next, user_message)

            if decision.action == "create":
                q = str(capture.get("query", "")).strip()
                return {
                    "pending_tasks": [],
                    "pending_contact_capture": {
                        "query": q,
                        "awaiting_create_confirmation": False,
                        "source": capture.get("source"),
                    },
                    "messages": [
                        AIMessage(content=_generate_response(
                            "User confirmed they want to create a new contact. Ask them to provide first name, last name, email, and phone number.",
                            context=_context_messages(state),
                        ))
                    ],
                }

            if decision.action == "cancel":
                source = capture.get("source")
                return {
                    "pending_tasks": [],
                    "staged_tasks": [],
                    "pending_contact_capture": {},
                    "messages": [AIMessage(content=_generate_response(
                        "User declined to create a new contact."
                        + (" Offer to continue the email with a different recipient." if source != "lookup_only" else ""),
                        context=_context_messages(state),
                    ))],
                }

            if capture.get("source") != "lookup_only" and decision.action == "alternate_recipient":
                candidate = (decision.recipient_candidate or user_message).strip()
                if candidate:
                    if is_valid_email(candidate):
                        draft = dict(state.get("email_draft", {}) or {})
                        draft["to"] = [candidate]
                        response = _build_email_flow_response(state, draft)
                        response["pending_contact_capture"] = {}
                        return response

                    matches = search_contacts(candidate)
                    if len(matches) == 1:
                        draft = dict(state.get("email_draft", {}) or {})
                        draft["to"] = [matches[0].email]
                        response = _build_email_flow_response(state, draft)
                        response["pending_contact_capture"] = {}
                        return response

            if decision.action == "pause":
                return {
                    "pending_tasks": [],
                    "staged_tasks": [],
                    "pending_contact_capture": {},
                    "messages": [
                        AIMessage(content=_generate_response(
                            "User sent a greeting while in contact creation flow. Pause the flow, greet them, and offer help with email or calendar.",
                            context=_context_messages(state),
                        ))
                    ],
                }

            # action="other" — user switched to a completely different request.
            # Clear the stale capture state and fall through to normal routing
            # so the new message is processed fresh (e.g. contact lookup, new email, etc.).
            log.info("pending_contact_capture: action=other, clearing capture and re-routing")

        else:
            # Not awaiting_create_confirmation — handle as contact detail input.
            return _handle_contact_capture_details(state, capture, user_message)

    # LLM-based conversational routing — catches greetings, small talk, etc.
    conversational_route = _route_pending_state(user_message, state_type="general_conversation")
    if conversational_route.action == "greet":
        return {
            "pending_tasks": [],
            "pending_contact_capture": {},
            "messages": [
                AIMessage(content=_generate_response(
                    "User greeted the assistant. Respond warmly and offer help with email and calendar tasks.",
                    context=_context_messages(state),
                ))
            ],
        }

    # Child slot-filling agent: handles multi-turn email drafting.
    existing_draft = state.get("email_draft", {}) or {}
    slot_update = _extract_email_slots(state)
    if slot_update.intent == "send_email" or existing_draft:
        merged = _merge_email_draft(existing_draft, slot_update)
        merged = _autofill_email_draft(state, merged)
        return _build_email_flow_response(state, merged)

    lookup = _extract_contact_lookup_intent(state)
    if lookup.intent == "lookup_contact":
        query = (lookup.query or "").strip()
        details, found = _lookup_contact_details(query)
        if not found and query:
            return {
                "pending_tasks": [],
                "pending_contact_capture": {
                    "query": query,
                    "awaiting_create_confirmation": True,
                    "source": "lookup_only",
                },
                "messages": [
                    AIMessage(content=_generate_response(
                        f"Contact '{query}' was not found. Inform the user and ask if they want to create a new contact for this person.",
                        context=_context_messages(state),
                        data={"query": query},
                    ))
                ],
            }
        return {"pending_tasks": [], "messages": [AIMessage(content=details)]}

    plan = orchestrator_agent_func(_context_messages(state))
    pending_tasks: list[str] = []
    if plan.action in ("send_email", "both") and plan.email_details:
        pending_tasks.append("email")
    if plan.action in ("create_calendar_event", "both") and plan.calendar_details:
        pending_tasks.append("calendar")

    updates: GraphState = {
        "orchestrator_decision": plan.action,
        "email_details": plan.email_details,
        "email_draft": {},
        "calendar_details": plan.calendar_details,
        "pending_tasks": [],
        "staged_tasks": pending_tasks,
        "awaiting_confirmation": False,
    }

    if plan.action == "ask_for_clarification":
        msg = plan.clarification_message or _generate_response(
            "Could not understand the user's request. Ask them to clarify.",
            context=_context_messages(state),
        )
        updates["messages"] = [AIMessage(content=msg)]
        return updates
    if plan.action == "none":
        updates["messages"] = [AIMessage(content=_generate_response(
            "The user's message is not about email or calendar. Let them know what you can help with.",
            context=_context_messages(state),
        ))]
        return updates

    if "email" in pending_tasks and plan.email_details:
        resolved, err, capture_needed = _resolve_recipients(plan.email_details)
        if capture_needed:
            q = str(capture_needed.get("query", "")).strip()
            updates["pending_contact_capture"] = {"query": q, "awaiting_create_confirmation": True}
            updates["staged_tasks"] = []
            updates["messages"] = [
                AIMessage(content=_generate_response(
                    f"Recipient '{q}' was not found in contacts. Ask the user if they want to create a new contact for this person.",
                    context=_context_messages(state),
                    data={"query": q},
                ))
            ]
            return updates
        if err:
            updates["staged_tasks"] = []
            updates["messages"] = [AIMessage(content=err)]
            return updates
        updates["email_details"] = EmailContent(
            to=resolved,
            subject=plan.email_details.subject,
            body=plan.email_details.body,
        )

    updates["awaiting_confirmation"] = True
    updates["messages"] = [AIMessage(content=_summary(updates, context=_context_messages(state)))]
    return updates


@traceable(name="node.send_email", run_type="tool")
def send_email_node(state: GraphState):
    log.info("send_email_node  enter")
    details = state.get("email_details")
    if not details:
        return {"messages": [AIMessage(content=_generate_response(
            "Email details are missing and the email cannot be sent. Inform the user.",
            context=_context_messages(state),
        ))]}
    provider = state.get("service_provider", "gmail")
    try:
        result = send_email(provider, details)
        msg = _generate_response(
            "Email was sent successfully. Inform the user.",
            context=_context_messages(state),
            data={"result": result, "recipients": details.to, "subject": details.subject},
        )
    except Exception as exc:
        msg = _generate_response(
            "Email sending failed. Inform the user about the error.",
            context=_context_messages(state),
            data={"error": str(exc)},
        )
    pending = [t for t in state.get("pending_tasks", []) if t != "email"]
    return {"messages": [AIMessage(content=msg)], "pending_tasks": pending}


@traceable(name="node.create_calendar_event", run_type="tool")
def create_calendar_event_node(state: GraphState):
    log.info("create_calendar_event_node  enter")
    details = state.get("calendar_details")
    if not details:
        return {"messages": [AIMessage(content=_generate_response(
            "Calendar event details are missing and the event cannot be created. Inform the user.",
            context=_context_messages(state),
        ))]}
    provider = state.get("service_provider", "gmail")
    try:
        result = create_calendar_event(provider, details)
        msg = _generate_response(
            "Calendar event was created successfully. Inform the user.",
            context=_context_messages(state),
            data={"result": result, "title": details.title, "start": details.start_time.isoformat()},
        )
    except Exception as exc:
        msg = _generate_response(
            "Calendar event creation failed. Inform the user about the error.",
            context=_context_messages(state),
            data={"error": str(exc)},
        )
    pending = [t for t in state.get("pending_tasks", []) if t != "calendar"]
    return {"messages": [AIMessage(content=msg)], "pending_tasks": pending}


@traceable(name="node.final_response", run_type="chain")
def final_response_node(state: GraphState):
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage) and message.content:
            return {"final_response": str(message.content)}
    return {"final_response": _generate_response(
        "Processing is complete but there is no specific result to show. Let the user know.",
        context=_context_messages(state),
    )}
