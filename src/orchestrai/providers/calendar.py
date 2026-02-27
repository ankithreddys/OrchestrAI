import logging
import os
import pickle

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from O365 import Account, MSGraphProtocol

from src.orchestrai.schemas.models import CalendarEvent

log = logging.getLogger("orchestrai.calendar")


def create_calendar_event(service_provider: str, calendar_details: CalendarEvent) -> str:
    provider = (service_provider or "").lower()
    if provider == "outlook":
        _create_outlook_event(calendar_details)
        return f"Calendar event '{calendar_details.title}' created in Outlook."
    if provider == "gmail":
        _create_gmail_event(calendar_details)
        return f"Calendar event '{calendar_details.title}' created in Google Calendar."
    raise ValueError("Invalid service provider. Choose 'Outlook' or 'Gmail'.")


def _create_outlook_event(calendar_details: CalendarEvent):
    credentials = (os.getenv("OUTLOOK_CLIENT_ID"), os.getenv("OUTLOOK_CLIENT_SECRET"))
    account = Account(credentials, protocol=MSGraphProtocol())
    if not account.authenticate(scopes=["basic", "calendar_all", "offline_access"]):
        raise Exception("Outlook authentication failed.")

    schedule = account.schedule()
    calendar = schedule.get_default_calendar()
    new_event = calendar.new_event()
    new_event.set_subject(calendar_details.title)
    new_event.set_start(calendar_details.start_time)
    new_event.set_end(calendar_details.end_time)
    if calendar_details.attendees:
        attendees = [{"address": a} for a in calendar_details.attendees]
        new_event.add_attendees(attendees)
    if calendar_details.location:
        new_event.set_location(calendar_details.location)
    if calendar_details.description:
        new_event.set_body(calendar_details.description)
    new_event.save()


def _create_gmail_event(calendar_details: CalendarEvent):
    scopes = ["https://www.googleapis.com/auth/calendar.events"]
    token_file = "google_calendar_token.pickle"
    credentials_file = "credentials.json"
    creds = None

    if os.path.exists(token_file):
        try:
            with open(token_file, "rb") as token:
                creds = pickle.load(token)
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            log.info("Calendar token expired, attempting refresh…")
            try:
                creds.refresh(Request())
                log.info("Calendar token refreshed successfully.")
            except Exception as exc:
                log.warning("Calendar token refresh failed: %s — deleting stale token.", exc)
                creds = None
                try:
                    os.remove(token_file)
                except OSError:
                    pass

        if not creds or not creds.valid:
            if not os.path.exists(credentials_file):
                raise FileNotFoundError(
                    "Missing credentials.json in project root. "
                    "Place Google OAuth client file at C:/PROJECTS/OrchestrAI/credentials.json"
                )
            log.info("Starting Calendar OAuth browser flow (run_local_server)…")
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, scopes)
            creds = flow.run_local_server(port=0)
            log.info("Calendar OAuth flow completed.")

        with open(token_file, "wb") as token:
            pickle.dump(creds, token)

    service = build("calendar", "v3", credentials=creds)
    event = {
        "summary": calendar_details.title,
        "location": calendar_details.location,
        "description": calendar_details.description,
        "start": {"dateTime": calendar_details.start_time.isoformat(), "timeZone": "America/New_York"},
        "end": {"dateTime": calendar_details.end_time.isoformat(), "timeZone": "America/New_York"},
        "attendees": [{"email": a} for a in (calendar_details.attendees or [])],
    }
    service.events().insert(calendarId="primary", body=event).execute()
