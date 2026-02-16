from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, EmailStr, Field


class EmailContent(BaseModel):
    to: List[str] = Field(
        ...,
        description="Recipient identifiers. Can be full emails or names to resolve via contacts directory.",
    )
    subject: str = Field(..., description="Email subject line.")
    body: str = Field(..., description="Email body content.")


class CalendarEvent(BaseModel):
    title: str = Field(..., description="Calendar event title.")
    start_time: datetime = Field(..., description="Start date/time (ISO 8601).")
    end_time: datetime = Field(..., description="End date/time (ISO 8601).")
    attendees: List[EmailStr] = Field(..., description="Attendee email addresses.")
    location: Optional[str] = Field(None, description="Event location.")
    description: Optional[str] = Field(None, description="Event description.")


class OrchestratorOutput(BaseModel):
    action: Literal["send_email", "create_calendar_event", "both", "ask_for_clarification", "none"]
    email_details: Optional[EmailContent] = None
    calendar_details: Optional[CalendarEvent] = None
    clarification_message: Optional[str] = None


class ContactCaptureDetails(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None


class EmailDraftUpdate(BaseModel):
    """Child-agent extraction for multi-turn email slot filling."""

    intent: Literal["send_email", "not_email", "unknown"] = "unknown"
    to: Optional[List[str]] = None
    subject: Optional[str] = None
    body: Optional[str] = None


class MeetingIntent(BaseModel):
    create_calendar_event: bool = False
    title: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    attendees: Optional[List[EmailStr]] = None
    location: Optional[str] = None
    description: Optional[str] = None


class EmailDraftAutofill(BaseModel):
    """Autofill missing email draft fields from conversation context."""

    subject: Optional[str] = None
    body: Optional[str] = None


class ContactLookupIntent(BaseModel):
    intent: Literal["lookup_contact", "other"] = "other"
    query: Optional[str] = None
