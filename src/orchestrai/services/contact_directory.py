import csv
import json
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher


EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass
class Contact:
    first_name: str
    last_name: str
    email: str
    phone: str = ""

    @property
    def name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()


def _normalize(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum() or ch.isspace()).strip()


def is_valid_email(value: str) -> bool:
    return bool(EMAIL_REGEX.match((value or "").strip()))


def load_contacts() -> list[Contact]:
    """Load contacts from JSON or CSV file configured by CONTACTS_FILE_PATH."""
    path = os.getenv("CONTACTS_FILE_PATH", "data/contacts.json")
    if not os.path.exists(path):
        return []

    _, ext = os.path.splitext(path.lower())
    if ext == ".csv":
        return _load_csv_contacts(path)
    return _load_json_contacts(path)


def _load_json_contacts(path: str) -> list[Contact]:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            raw = fp.read().strip()
            if not raw:
                return []
            data = json.loads(raw)
    except Exception:
        # Gracefully handle malformed/empty contacts file.
        return []

    contacts: list[Contact] = []
    if isinstance(data, list):
        for row in data:
            first_name = str((row or {}).get("first_name", "")).strip()
            last_name = str((row or {}).get("last_name", "")).strip()
            legacy_name = str((row or {}).get("name", "")).strip()
            if not first_name and not last_name and legacy_name:
                tokens = legacy_name.split()
                if len(tokens) >= 2:
                    first_name = tokens[0]
                    last_name = " ".join(tokens[1:])
                elif len(tokens) == 1:
                    first_name = tokens[0]
                    last_name = ""
            email = str((row or {}).get("email", "")).strip()
            phone = str((row or {}).get("phone", "")).strip()
            if (first_name or last_name) and is_valid_email(email):
                contacts.append(Contact(first_name=first_name, last_name=last_name, email=email, phone=phone))
    return contacts


def _load_csv_contacts(path: str) -> list[Contact]:
    contacts: list[Contact] = []
    with open(path, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            first_name = str((row or {}).get("first_name", "")).strip()
            last_name = str((row or {}).get("last_name", "")).strip()
            legacy_name = str((row or {}).get("name", "")).strip()
            if not first_name and not last_name and legacy_name:
                tokens = legacy_name.split()
                if len(tokens) >= 2:
                    first_name = tokens[0]
                    last_name = " ".join(tokens[1:])
                elif len(tokens) == 1:
                    first_name = tokens[0]
                    last_name = ""
            email = str((row or {}).get("email", "")).strip()
            phone = str((row or {}).get("phone", "")).strip()
            if (first_name or last_name) and is_valid_email(email):
                contacts.append(Contact(first_name=first_name, last_name=last_name, email=email, phone=phone))
    return contacts


def _score(query: str, candidate: str) -> float:
    qn = _normalize(query)
    cn = _normalize(candidate)
    if not qn or not cn:
        return 0.0

    # Use order-aware fuzzy similarity. Unique-letter overlap caused false positives
    # (e.g. "ankith" matching "padakanti" due to shared letters).
    return SequenceMatcher(None, qn, cn).ratio()


def search_contacts(query: str, threshold: float | None = None) -> list[Contact]:
    """Fuzzy-search contacts by name/email local part with configurable threshold."""
    normalized_query = _normalize(query)
    if not normalized_query:
        return []

    min_score = threshold
    if min_score is None:
        try:
            min_score = float(os.getenv("RECIPIENT_MATCH_THRESHOLD", "0.7"))
        except ValueError:
            min_score = 0.7

    contacts = load_contacts()

    # Prefer exact field matches first to avoid noisy fuzzy disambiguation.
    exact_matches: list[Contact] = []
    for contact in contacts:
        local_part = contact.email.split("@", 1)[0]
        fields = (
            _normalize(contact.first_name),
            _normalize(contact.last_name),
            _normalize(contact.name),
            _normalize(local_part),
        )
        if normalized_query in fields:
            exact_matches.append(contact)
    if exact_matches:
        return exact_matches

    ranked: list[tuple[float, Contact]] = []
    for contact in contacts:
        local_part = contact.email.split("@", 1)[0]
        score = max(
            _score(query, contact.name),
            _score(query, contact.first_name),
            _score(query, contact.last_name),
            _score(query, local_part),
        )
        if score >= min_score:
            ranked.append((score, contact))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked]


def save_contact(first_name: str, last_name: str, email: str, phone: str = "") -> Contact:
    """Persist a new contact to configured contacts file (JSON/CSV)."""
    first = (first_name or "").strip()
    last = (last_name or "").strip()
    full_name = f"{first} {last}".strip()
    mail = (email or "").strip()
    phone_value = (phone or "").strip()

    if not full_name:
        raise ValueError("Contact name is required.")
    if not is_valid_email(mail):
        raise ValueError("A valid email is required.")

    path = os.getenv("CONTACTS_FILE_PATH", "data/contacts.json")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    _, ext = os.path.splitext(path.lower())
    if ext == ".csv":
        _save_contact_csv(path, first, last, mail, phone_value)
    else:
        _save_contact_json(path, first, last, mail, phone_value)

    return Contact(first_name=first, last_name=last, email=mail, phone=phone_value)


def _save_contact_json(path: str, first_name: str, last_name: str, email: str, phone: str):
    rows: list[dict] = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, list):
                    rows = [row for row in data if isinstance(row, dict)]
        except Exception:
            rows = []

    updated = False
    for row in rows:
        row_email = str(row.get("email", "")).strip().lower()
        if row_email == email.lower():
            row["first_name"] = first_name
            row["last_name"] = last_name
            row["name"] = f"{first_name} {last_name}".strip()
            row["email"] = email
            row["phone"] = phone
            updated = True
            break

    if not updated:
        rows.append(
            {
                "first_name": first_name,
                "last_name": last_name,
                "name": f"{first_name} {last_name}".strip(),
                "email": email,
                "phone": phone,
            }
        )

    with open(path, "w", encoding="utf-8") as fp:
        json.dump(rows, fp, indent=2)


def _save_contact_csv(path: str, first_name: str, last_name: str, email: str, phone: str):
    existing: list[dict] = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                existing.append(
                    {
                        "first_name": str((row or {}).get("first_name", "")).strip(),
                        "last_name": str((row or {}).get("last_name", "")).strip(),
                        "name": str((row or {}).get("name", "")).strip(),
                        "email": str((row or {}).get("email", "")).strip(),
                        "phone": str((row or {}).get("phone", "")).strip(),
                    }
                )

    updated = False
    for row in existing:
        if row["email"].lower() == email.lower():
            row["first_name"] = first_name
            row["last_name"] = last_name
            row["name"] = f"{first_name} {last_name}".strip()
            row["phone"] = phone
            updated = True
            break
    if not updated:
        existing.append(
            {
                "first_name": first_name,
                "last_name": last_name,
                "name": f"{first_name} {last_name}".strip(),
                "email": email,
                "phone": phone,
            }
        )

    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["first_name", "last_name", "name", "email", "phone"])
        writer.writeheader()
        writer.writerows(existing)
