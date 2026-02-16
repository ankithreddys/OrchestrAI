from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.orchestrai.agents.orchestrator import orchestrator_agent_func
from src.orchestrai.graph.state import GraphState
from src.orchestrai.llm.client import get_structured_chat_model
from src.orchestrai.providers.calendar import create_calendar_event
from src.orchestrai.providers.email import send_email
from src.orchestrai.schemas.models import (
    CalendarEvent,
    ContactCaptureDetails,
    EmailContent,
    EmailDraftAutofill,
    EmailDraftUpdate,
    ContactLookupIntent,
    MeetingIntent,
)
from src.orchestrai.services.contact_directory import is_valid_email, load_contacts, save_contact, search_contacts

EMAIL_SLOT_MODEL = get_structured_chat_model(EmailDraftUpdate, temperature=0)
MEETING_INTENT_MODEL = get_structured_chat_model(MeetingIntent, temperature=0)
EMAIL_AUTOFILL_MODEL = get_structured_chat_model(EmailDraftAutofill, temperature=0.2)
CONTACT_CAPTURE_MODEL = get_structured_chat_model(ContactCaptureDetails, temperature=0)
CONTACT_LOOKUP_MODEL = get_structured_chat_model(ContactLookupIntent, temperature=0)


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
        "Use the conversation and existing draft to extract:\n"
        "- intent (send_email/not_email/unknown)\n"
        "- to (recipient names/emails)\n"
        "- subject\n"
        "- body\n"
        "Keep existing values unless user updates them.\n"
        "If user gives email body text, put it in body.\n"
        "If user only provides recipient email/name, fill only to.\n"
        f"Existing draft: {existing}"
    )
    try:
        return EMAIL_SLOT_MODEL.invoke([SystemMessage(content=prompt), *context[-14:]])
    except Exception:
        return EmailDraftUpdate(intent="unknown")


def _merge_email_draft(existing: dict, update: EmailDraftUpdate) -> dict:
    draft = dict(existing or {})
    if update.to:
        draft["to"] = [t.strip() for t in update.to if str(t).strip()]
    if update.subject and update.subject.strip():
        draft["subject"] = update.subject.strip()
    if update.body and update.body.strip():
        draft["body"] = update.body.strip()
    return draft


def _autofill_email_draft(state: GraphState, draft: dict) -> dict:
    """Generate missing subject/body from conversation context when possible."""
    missing = [k for k in ("subject", "body") if not draft.get(k)]
    if not missing:
        return draft

    context = _context_messages(state)
    prompt = (
        "You are an email drafting assistant.\n"
        "Generate missing email subject/body based on the conversation.\n"
        "Keep concise, professional tone unless user style is explicitly casual.\n"
        "If body is generated, include the core details user mentioned (who/what/when/where).\n"
        f"Current draft: {draft}\n"
        f"Missing fields: {missing}"
    )
    try:
        gen = EMAIL_AUTOFILL_MODEL.invoke([SystemMessage(content=prompt), *context[-14:]])
        if missing and "subject" in missing and gen.subject and gen.subject.strip():
            draft["subject"] = gen.subject.strip()
        if missing and "body" in missing and gen.body and gen.body.strip():
            draft["body"] = gen.body.strip()
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
    resolved: list[str] = []
    for raw in email_details.to:
        token = (raw or "").strip()
        if not token:
            continue
        if is_valid_email(token):
            resolved.append(token)
            continue
        matches = search_contacts(token)
        if len(matches) == 0:
            return [], None, {"query": token}
        if len(matches) > 1:
            options = "\n".join([f"- {c.name} <{c.email}>" for c in matches[:5]])
            return [], f"I found multiple matches for '{token}'. Please choose one:\n{options}", None
        resolved.append(matches[0].email)
    if not resolved:
        return [], "Please provide at least one recipient.", None
    return resolved, None, None


def _parse_contact_details(user_text: str) -> tuple[dict | None, str | None]:
    text = (user_text or "").strip()
    if not text:
        return None, "Please provide first name, last name, email, and phone."

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
        return None, (
            "I need contact details to save this person.\n"
            "Please share: first name, last name, email, phone.\n"
            "Example: Amogh Padakanti, amogh@ufl.edu, +1 352-555-1234"
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


def _lookup_contact_details(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "Please tell me whose contact details you need."

    if is_valid_email(q):
        contacts = [c for c in load_contacts() if c.email.lower() == q.lower()]
    else:
        contacts = search_contacts(q)

    if not contacts:
        return f"I could not find contact details for '{q}'."

    if len(contacts) > 1:
        options = "\n".join(
            [
                f"- {c.first_name} {c.last_name} <{c.email}> | phone: {c.phone or 'N/A'}"
                for c in contacts[:5]
            ]
        )
        return f"I found multiple contacts for '{q}':\n{options}"

    c = contacts[0]
    return (
        "Contact details:\n"
        f"First Name: {c.first_name}\n"
        f"Last Name: {c.last_name}\n"
        f"Email: {c.email}\n"
        f"Phone: {c.phone or 'N/A'}"
    )


def _build_email_flow_response(state: GraphState, merged: dict):
    missing = [k for k in ("to", "subject", "body") if not merged.get(k)]
    if missing:
        prompts = {"to": "recipient email/name", "subject": "subject line", "body": "email message body"}
        ask = ", ".join(prompts[m] for m in missing)
        return {
            "email_draft": merged,
            "pending_tasks": [],
            "messages": [AIMessage(content=f"Got it. Please provide: {ask}.")],
        }

    candidate = EmailContent(to=merged["to"], subject=merged["subject"], body=merged["body"])
    resolved, err, capture_needed = _resolve_recipients(candidate)
    if capture_needed:
        q = str(capture_needed.get("query", "")).strip()
        return {
            "email_draft": merged,
            "pending_contact_capture": capture_needed,
            "staged_tasks": [],
            "pending_tasks": [],
            "messages": [
                AIMessage(
                    content=(
                        f"I could not find '{q}' in contacts.\n"
                        "Please provide first name, last name, email, and phone so I can save it."
                    )
                )
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
    updates["messages"] = [AIMessage(content=_summary(updates))]
    return updates


def _summary(state: GraphState) -> str:
    lines = ["Please confirm before I execute:"]
    if state.get("email_details"):
        e = state["email_details"]
        lines.extend(["", f"Email To: {', '.join(e.to)}", f"Subject: {e.subject}", f"Body: {e.body}"])
    if state.get("calendar_details"):
        c = state["calendar_details"]
        lines.extend(["", f"Event: {c.title}", f"Start: {c.start_time.isoformat()}", f"End: {c.end_time.isoformat()}"])
    lines.append("")
    lines.append("Reply `confirm` to execute or `cancel`.")
    return "\n".join(lines)


def orchestrator_node(state: GraphState):
    user_message = _latest_human_content(state)
    if not user_message:
        return {"messages": [AIMessage(content="Please provide a request.")], "pending_tasks": []}

    if state.get("awaiting_confirmation"):
        norm = user_message.strip().lower()
        if norm in {"confirm", "yes", "y", "send", "go ahead"}:
            return {"awaiting_confirmation": False, "pending_tasks": state.get("staged_tasks", [])}
        if norm in {"cancel", "no", "n", "stop"}:
            return {
                "awaiting_confirmation": False,
                "staged_tasks": [],
                "pending_tasks": [],
                "messages": [AIMessage(content="Cancelled. I did not execute any action.")],
            }
        return {"pending_tasks": [], "messages": [AIMessage(content="Reply `confirm` or `cancel`.")]}

    if state.get("pending_contact_capture"):
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
            return {"pending_tasks": [], "messages": [AIMessage(content=f"Could not save contact: {exc}")]}

        capture = state.get("pending_contact_capture", {}) or {}
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
        # Prepend save acknowledgement.
        msgs = response.get("messages", [])
        response["messages"] = [AIMessage(content=f"Saved contact: {saved.name} <{saved.email}>"), *msgs]
        response["pending_contact_capture"] = {}
        return response

    lookup = _extract_contact_lookup_intent(state)
    if lookup.intent == "lookup_contact":
        details = _lookup_contact_details(lookup.query or "")
        return {"pending_tasks": [], "messages": [AIMessage(content=details)]}

    # Child slot-filling agent: handles multi-turn email drafting.
    existing_draft = state.get("email_draft", {}) or {}
    slot_update = _extract_email_slots(state)
    if slot_update.intent == "send_email" or existing_draft:
        merged = _merge_email_draft(existing_draft, slot_update)
        merged = _autofill_email_draft(state, merged)
        return _build_email_flow_response(state, merged)

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
        msg = plan.clarification_message or "Please clarify your request."
        updates["messages"] = [AIMessage(content=msg)]
        return updates
    if plan.action == "none":
        updates["messages"] = [AIMessage(content="Ask me to send an email or create a calendar event.")]
        return updates

    if "email" in pending_tasks and plan.email_details:
        resolved, err, capture_needed = _resolve_recipients(plan.email_details)
        if capture_needed:
            q = str(capture_needed.get("query", "")).strip()
            updates["pending_contact_capture"] = capture_needed
            updates["staged_tasks"] = []
            updates["messages"] = [
                AIMessage(
                    content=(
                        f"I could not find '{q}' in contacts.\n"
                        "Please provide first name, last name, email, and phone so I can save it."
                    )
                )
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
    updates["messages"] = [AIMessage(content=_summary(updates))]
    return updates


def send_email_node(state: GraphState):
    details = state.get("email_details")
    if not details:
        return {"messages": [AIMessage(content="Email details are missing.")]}
    provider = state.get("service_provider", "gmail")
    try:
        msg = send_email(provider, details)
    except Exception as exc:
        msg = f"Email failed: {exc}"
    pending = [t for t in state.get("pending_tasks", []) if t != "email"]
    return {"messages": [AIMessage(content=msg)], "pending_tasks": pending}


def create_calendar_event_node(state: GraphState):
    details = state.get("calendar_details")
    if not details:
        return {"messages": [AIMessage(content="Calendar details are missing.")]}
    provider = state.get("service_provider", "gmail")
    try:
        msg = create_calendar_event(provider, details)
    except Exception as exc:
        msg = f"Calendar failed: {exc}"
    pending = [t for t in state.get("pending_tasks", []) if t != "calendar"]
    return {"messages": [AIMessage(content=msg)], "pending_tasks": pending}


def final_response_node(state: GraphState):
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage) and message.content:
            return {"final_response": str(message.content)}
    return {"final_response": "Request processed."}
