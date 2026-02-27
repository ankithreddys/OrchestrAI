# OrchestrAI

**OrchestrAI** is a fully LLM-driven, multi-agent AI assistant built on **LangGraph** and **LangChain** that helps users send emails and create calendar events through natural conversation. Every user-facing response, every classification decision, and every routing choice is made by specialized LLM agents — there are zero hardcoded strings or keyword-based rules in the system.

Users interact via a **Gradio** chat interface (text or voice) embedded inside a **FastAPI** web application with optional Google OAuth sign-in.

---

## What It Does

- **Send emails** via Gmail or Outlook through natural multi-turn conversation
- **Create calendar events** on Google Calendar or Outlook
- **Auto-detect meeting intent** — if the email mentions a time/date, the system automatically stages a calendar event alongside the email
- **Manage contacts** — look up, create, and store contacts in a local directory with fuzzy name matching
- **Voice input** — speak requests via microphone; audio is transcribed with Google Speech Recognition
- **Confirm before executing** — always shows a summary and waits for explicit user confirmation

---

## How It Works

OrchestrAI is built as a **stateful graph** of specialized LLM agents that collaborate to handle a user's request. There are no hardcoded response templates or keyword-matching rules — every decision is made by an LLM agent with a structured output schema.

### Architecture Overview

```
User (text/voice)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  FastAPI + Gradio Web App (main.py)                 │
│  - Google OAuth (optional)                          │
│  - Speech-to-text transcription                     │
│  - Session management with thread_id                │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  LangGraph State Machine (builder.py)               │
│                                                     │
│  orchestrator ──► send_email ──► create_calendar    │
│       │               │               │             │
│       └───────────────┴───────────────┘             │
│                       │                             │
│                       ▼                             │
│                 final_response                      │
└─────────────────────────────────────────────────────┘
```

### The Agent Team

OrchestrAI uses **7 specialized LLM agents**, each with its own Pydantic output schema and system prompt:

| Agent | Schema | Purpose |
|-------|--------|---------|
| **Email Slot-Filler** | `EmailDraftUpdate` | Extracts recipient, subject, and body from conversation across multiple turns |
| **Email Autofill** | `EmailDraftAutofill` | Infers a subject line from context if user mentioned a topic but didn't provide one |
| **Email Polisher** | `EmailDraftAutofill` | Rewrites subject + body into professional email text before confirmation |
| **Meeting Intent Detector** | `MeetingIntent` | Detects time/date cues in email context and stages a calendar event |
| **Pending State Router** | `PendingStateRoute` | Classifies user input during pending states (confirm, cancel, greeting, provide details, etc.) |
| **Contact Lookup** | `ContactLookupIntent` | Identifies when user wants to look up a contact's details |
| **Response Generator** | `AgentResponse` | Produces every user-facing message — ensures natural, context-aware language with no raw data leakage |

In addition, the **Orchestrator Agent** (`OrchestratorOutput`) serves as the fallback planner when the conversation doesn't match slot-filling or contact patterns.

### Step-by-Step Flow

Here is what happens for a typical email request like *"send an email to Pranay about the project update"*:

1. **User sends message** → Gradio captures text (or transcribes audio) and calls the LangGraph with a `thread_id`

2. **Orchestrator node** receives the message and checks state flags in order:
   - `awaiting_confirmation` → routes to confirm/cancel via **Pending State Router**
   - `pending_contact_capture` → routes to contact creation flow
   - Otherwise → proceeds to classification

3. **Conversational routing** — the **Pending State Router** (in `general_conversation` mode) checks if the message is a greeting/small-talk. If so, responds warmly. If not, continues.

4. **Slot-filling** — the **Email Slot-Filler** agent extracts `{to: "Pranay", subject: "Project Update", body: null}` from the conversation context

5. **Early recipient validation** — before asking for remaining fields, the system checks if "Pranay" exists in the contact directory:
   - **Found (1 match)** → resolves to email address, continues
   - **Not found** → immediately asks user to create a new contact or provide a different recipient (does NOT collect subject/body first)
   - **Multiple matches** → shows options and asks user to pick one

6. **Missing fields** — if subject or body is still missing, the **Response Generator** asks the user for the remaining fields. This repeats across turns.

7. **Autofill** — once recipient is set, the **Email Autofill** agent checks if a subject can be inferred from context (e.g., "about the project" → subject line). It never fabricates a body.

8. **Polish** — once all three fields are filled, the **Email Polisher** rewrites the subject and body into professional email text

9. **Meeting detection** — the **Meeting Intent Detector** checks for time/date cues. If found, it stages a calendar event alongside the email.

10. **Confirmation** — the **Response Generator** presents a clear summary:
    ```
    Email:
    To: pranay@example.com
    Subject: Project Update Discussion
    Body: Hi Pranay, I'd like to discuss the latest project updates...

    Please reply confirm to send or cancel to abort.
    ```

11. **Execution** — on "confirm", the graph routes to `send_email` node (and optionally `create_calendar_event`), which calls the Gmail/Outlook API

12. **Final response** — the last AI message is returned to the user

### Contact Management

When a recipient name doesn't match any contact:

1. System asks: *"Pranay wasn't found in your contacts. Would you like to create a new contact or use a different recipient?"*
2. User says "yes" → system asks for first name, last name, email, and phone
3. User provides details → contact is saved to `data/contacts.json` and the email flow resumes with the new email address
4. User says "no" → system offers to continue with a different recipient

Contact lookup also works independently: *"get details of Amogh"* returns the stored contact info.

### Thread Memory

Each chat session has a unique `thread_id`. The LangGraph `MemorySaver` checkpointer maintains full conversation state per thread, enabling multi-turn interactions within a session.

---

## Project Structure

```text
.
├── main.py                          # FastAPI + Gradio app, OAuth, entry point
├── requirements.txt
├── langgraph.json                   # LangGraph Studio configuration
├── credentials.json                 # Google OAuth client file (manual setup)
├── .env                             # Environment variables (create manually)
├── data/
│   ├── contacts.json                # Runtime contacts (auto-created)
│   └── contacts.example.json        # Example contacts file
├── logs/
│   └── orchestrai.log               # Application logs
├── evals/
│   ├── goldens.json                 # End-to-end eval test cases
│   ├── agent_goldens.json           # Agent-level eval test cases
│   ├── test_orchestrai_deepeval.py  # Full-flow DeepEval suite
│   └── test_agents_deepeval.py      # Agent-level DeepEval suite
└── src/orchestrai/
    ├── agents/
    │   └── orchestrator.py          # Fallback orchestrator agent
    ├── graph/
    │   ├── builder.py               # LangGraph state machine definition
    │   ├── nodes.py                 # All graph nodes + LLM agents
    │   └── state.py                 # GraphState TypedDict definition
    ├── llm/
    │   └── client.py                # Centralized LLM client configuration
    ├── providers/
    │   ├── email.py                 # Gmail + Outlook email sending
    │   └── calendar.py              # Google Calendar + Outlook event creation
    ├── schemas/
    │   └── models.py                # Pydantic schemas for all LLM agents
    └── services/
        └── contact_directory.py     # Contact CRUD + fuzzy search
```

---

## Setup

### 1. Create virtual environment

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

#### LLM Configuration (required)

```env
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0
```

The LLM client is compatible with any OpenAI-compatible API endpoint.

Backward-compatible aliases are also supported:

```env
GATOR_API_KEY=...
GATOR_BASE_URL=...
ORCH_MODEL=...
```

#### LangSmith Tracing (optional, recommended)

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=OrchestrAI
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

#### Google Sign-in Auth (optional)

```env
AUTH_ENABLED=true
APP_BASE_URL=http://127.0.0.1:8080
APP_SESSION_SECRET=replace_with_a_long_random_secret
GOOGLE_CLIENT_ID=your_google_oauth_client_id
GOOGLE_CLIENT_SECRET=your_google_oauth_client_secret
```

Google OAuth console: add Authorized redirect URI `http://127.0.0.1:8080/auth/google/callback`

#### Contact Configuration (optional)

```env
CONTACTS_FILE_PATH=data/contacts.json
RECIPIENT_MATCH_THRESHOLD=0.7
```

#### Outlook (only if using Outlook provider)

```env
OUTLOOK_CLIENT_ID=...
OUTLOOK_CLIENT_SECRET=...
```

### 4. Google OAuth Setup (for Gmail / Google Calendar)

1. Create a project in [Google Cloud Console](https://console.cloud.google.com/)
2. Enable **Gmail API** and **Google Calendar API**
3. Create OAuth 2.0 credentials (Desktop app type)
4. Download the client JSON and save it as `credentials.json` in the project root
5. On first email send or calendar event creation, a browser window opens for OAuth consent
6. After consent, token files are created automatically:
   - `gmail_token.pickle`
   - `google_calendar_token.pickle`

If tokens expire, the system automatically attempts refresh. If refresh fails, the stale token is deleted and re-authentication is triggered.

---

## Run

```bash
python main.py
```

The app starts at `http://127.0.0.1:8080`.

| URL | Description |
|-----|-------------|
| `http://127.0.0.1:8080` | Landing page (redirects to `/app` or sign-in) |
| `http://127.0.0.1:8080/app` | Gradio chat interface |
| `http://127.0.0.1:8080/app-shell` | Authenticated wrapper (when auth enabled) |
| `http://127.0.0.1:8080/health` | Health check endpoint |

---

## LangGraph Studio

Run the local LangGraph dev server for visual graph inspection:

```bash
langgraph dev
```

Opens at `http://127.0.0.1:2024/studio`. Configuration is in `langgraph.json` with entrypoint `src/orchestrai/graph/builder.py:build_graph`.

---

## Logging

Application logs are written to both the console and `logs/orchestrai.log`. Key events logged:

- Every graph invocation (input, thread ID, response)
- Orchestrator node entry (user message, state flags)
- Pending state routing decisions (state type, action result)
- Email flow progress (draft state, recipient validation)
- Contact capture steps
- Email/calendar node execution
- OAuth token refresh attempts and failures

---

## Observability with LangSmith

When LangSmith tracing is enabled, every user turn generates a detailed trace with nested spans:

- **Root span**: `orchestrai_request` — full request lifecycle
- **Node spans**: `node.orchestrator`, `node.send_email`, `node.create_calendar_event`, `node.final_response`
- **Agent spans**: `agent.response_generator`, `agent.pending_state_router` — internal LLM decisions

View traces in the [LangSmith dashboard](https://smith.langchain.com/) under your configured project name.

---

## Evaluation with DeepEval

### End-to-end evals

```bash
deepeval test run evals/test_orchestrai_deepeval.py
```

Test cases are defined in `evals/goldens.json`.

### Agent-level evals

```bash
deepeval test run evals/test_agents_deepeval.py
```

Test cases in `evals/agent_goldens.json` evaluate the orchestrator agent's decision quality.

### Required env vars for evals

```powershell
$env:LLM_API_KEY="..."
$env:LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
$env:LLM_MODEL="gpt-4o-mini"
$env:OPENAI_API_KEY="..."   # DeepEval judge key
```

---

## Contact File Format

`data/contacts.json`:

```json
[
  {
    "first_name": "Amogh",
    "last_name": "P",
    "email": "amoghp@ufl.edu",
    "phone": "352********"
  }
]
```

The `name` field is supported for backward compatibility. Fuzzy matching uses first name, last name, full name, and email local-part.

---

## Example Conversations

### Simple email

```
User: Send an email to Amogh
Assistant: I found Amogh Padakanti (amoghp@ufl.edu). Could you provide the subject and body?
User: Subject is project update, body is let's sync up tomorrow at 2pm
Assistant: Here are the details:
  To: amoghp@ufl.edu
  Subject: Project Update
  Body: Hi Amogh, Let's sync up tomorrow at 2:00 PM...
  Please confirm to send or cancel.
User: confirm
Assistant: Email sent successfully to amoghp@ufl.edu!
```

### Email + calendar auto-detection

```
User: Email Vishal about lunch tomorrow at 3pm at Broward
Assistant: [Presents email + calendar event for confirmation]
User: confirm
Assistant: Email sent! Calendar event "Lunch Tomorrow at 3 PM" created.
```

### Unknown recipient

```
User: Send email to Pranay
Assistant: I couldn't find Pranay in your contacts. Would you like to create a new contact or provide a different recipient?
User: Yes, create the contact
Assistant: Please provide first name, last name, email, and phone number.
User: Pranay P, pranayp@ufl.edu, 352*******
Assistant: Contact saved! The email will go to pranayp@ufl.edu. What's the subject and body?
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `credentials.json` not found | Place your Google OAuth client file in the project root |
| Google Calendar 403 `accessNotConfigured` | Enable Google Calendar API in your Google Cloud project |
| Gmail/Calendar hangs on send | Token expired — delete `gmail_token.pickle` or `google_calendar_token.pickle` and re-authenticate |
| `invalid_grant: Token has been expired` | Delete the corresponding `.pickle` file; the system will re-trigger OAuth |
| Fuzzy contact matching wrong person | Adjust `RECIPIENT_MATCH_THRESHOLD` in `.env` (higher = stricter) |
| `PydanticSerializationUnexpectedValue` warning | Cosmetic; suppressed automatically in the codebase |
