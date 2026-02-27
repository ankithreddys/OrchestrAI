import base64
import logging
import os
import pickle
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from O365 import Account, MSGraphProtocol

from src.orchestrai.schemas.models import EmailContent

log = logging.getLogger("orchestrai.email")


def send_email(service_provider: str, email_details: EmailContent) -> str:
    provider = (service_provider or "").lower()
    if provider == "outlook":
        _send_outlook_email(email_details)
        return f"Email to {', '.join(email_details.to)} sent via Outlook."
    if provider == "gmail":
        _send_gmail_email(email_details)
        return f"Email to {', '.join(email_details.to)} sent via Gmail."
    raise ValueError("Invalid service provider. Choose 'Outlook' or 'Gmail'.")


def _send_outlook_email(email_details: EmailContent):
    credentials = (os.getenv("OUTLOOK_CLIENT_ID"), os.getenv("OUTLOOK_CLIENT_SECRET"))
    account = Account(credentials, protocol=MSGraphProtocol())
    if not account.authenticate(scopes=["basic", "message_all", "offline_access"]):
        raise Exception("Outlook authentication failed.")

    mailbox = account.mailbox()
    new_email = mailbox.new_message()
    new_email.to.add(email_details.to)
    new_email.subject = email_details.subject
    new_email.body = email_details.body
    new_email.send()


def _send_gmail_email(email_details: EmailContent):
    scopes = ["https://www.googleapis.com/auth/gmail.send"]
    token_file = "gmail_token.pickle"
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
            log.info("Gmail token expired, attempting refresh…")
            try:
                creds.refresh(Request())
                log.info("Gmail token refreshed successfully.")
            except Exception as exc:
                log.warning("Gmail token refresh failed: %s — deleting stale token.", exc)
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
            log.info("Starting Gmail OAuth browser flow (run_local_server)…")
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, scopes)
            creds = flow.run_local_server(port=0)
            log.info("Gmail OAuth flow completed.")

        with open(token_file, "wb") as token:
            pickle.dump(creds, token)

    service = build("gmail", "v1", credentials=creds)
    message_text = MIMEText(email_details.body)
    message_text["to"] = ", ".join(email_details.to)
    message_text["subject"] = email_details.subject
    raw_message = base64.urlsafe_b64encode(message_text.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw_message}).execute()
