# OrchestrAI

OrchestrAI is a LangGraph + LangChain assistant that can:

- draft and send emails (Gmail or Outlook)
- create calendar events (Google Calendar or Outlook)
- collect and reuse contact details from a local contact directory
- run multi-turn conversations with confirmation before execution

## Current Features

- **Multi-turn email drafting**
  - User can provide recipient/subject/body across multiple turns.
  - Missing `subject` and `body` are auto-drafted from context when possible.
- **Contact resolution**
  - Recipients can be names or emails.
  - Name lookup uses fuzzy matching (configurable threshold).
  - If a contact is missing, assistant asks for first name, last name, email, phone and stores it.
- **Contact lookup**
  - Prompts like "get details of amogh" return stored contact details.
- **Auto email + calendar pairing**
  - If meeting intent is detected in email context, assistant stages both email and event.
- **Execution confirmation**
  - Assistant shows a summary and waits for `confirm` or `cancel`.
- **Thread memory**
  - Each chat session uses a `thread_id`; memory persists per thread while app runs.

## Project Structure

```text
.
├── main.py
├── requirements.txt
├── credentials.json                 # Google OAuth client file (manual setup)
├── data/
│   ├── contacts.json                # runtime contacts
│   └── contacts.example.json
└── src/orchestrai/
    ├── agents/
    │   └── orchestrator.py
    ├── graph/
    │   ├── builder.py
    │   ├── nodes.py
    │   └── state.py
    ├── llm/
    │   └── client.py
    ├── providers/
    │   ├── email.py
    │   └── calendar.py
    ├── schemas/
    │   └── models.py
    └── services/
        ├── contact_directory.py
        └── __init__.py
```

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Environment Variables

Create `.env` in project root.

### LLM config (required)

```env
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0
```

Backward-compatible aliases are also supported:

```env
GATOR_API_KEY=...
GATOR_BASE_URL=...
ORCH_MODEL=...
```

### Contact config (optional)

```env
CONTACTS_FILE_PATH=data/contacts.json
RECIPIENT_MATCH_THRESHOLD=0.7
```

### Outlook config (only if using Outlook provider)

```env
OUTLOOK_CLIENT_ID=...
OUTLOOK_CLIENT_SECRET=...
```

## Google OAuth Requirements

For Gmail and Google Calendar providers:

- You must place `credentials.json` in project root.
- First successful auth generates token files:
  - `gmail_token.pickle`
  - `google_calendar_token.pickle`

If Calendar API is disabled in Google Cloud, event creation fails with HTTP 403. Enable:

- Gmail API
- Google Calendar API

in the same Google project used by `credentials.json`.

## Run

```bash
python main.py
```

App starts on:

- `http://127.0.0.1:8080`

## Contact File Format

`data/contacts.json` uses:

```json
[
  {
    "first_name": "Amogh",
    "last_name": "Padakanti",
    "name": "Amogh Padakanti",
    "email": "amoghpadakanti@ufl.edu",
    "phone": "3527570959"
  }
]
```

Notes:

- `name` is kept for compatibility.
- Matching uses `first_name`, `last_name`, full name, and email local-part.

## Typical Conversation Flow

1. User asks to send email or create event.
2. Assistant fills missing details across turns.
3. Unknown contact -> asks for first/last/email/phone and stores it.
4. Assistant shows execution summary.
5. User replies `confirm`.
6. Assistant executes provider actions.

## Troubleshooting

- **`credentials.json` not found**
  - Put OAuth file at `C:/PROJECTS/OrchestrAI/credentials.json`.
- **Google Calendar 403 `accessNotConfigured`**
  - Enable Google Calendar API in your Google Cloud project and retry after a few minutes.
- **Contact lookup weird match**
  - Check `data/contacts.json` values and fuzzy threshold in `.env`.
