"""
Human escalation tool — logs an issue and returns a JCRC ticket ID.

Tool: escalate_to_human
Used when the agent cannot resolve an issue from policy documents alone and
a JCRC committee member needs to follow up.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel


class EscalationInput(BaseModel):
    issue_type: str
    description: str
    resident_query: str


class EscalationResult(BaseModel):
    ticket_id: str
    message: str
    contact: str


async def escalate_to_human(input: EscalationInput) -> EscalationResult:
    """
    Create a support ticket and return contact details for the JCRC.

    In production this would POST to a ticketing system (Notion, Airtable, etc.)
    or send an email to the JCRC inbox.  Currently it generates a UUID ticket ID
    and instructs the resident to follow up via email.
    """
    ticket_id = f"SH-{str(uuid.uuid4())[:8].upper()}"

    return EscalationResult(
        ticket_id=ticket_id,
        message=(
            f"Your issue has been logged as ticket {ticket_id}. "
            "A JCRC member will follow up within 2 working days."
        ),
        contact="jcrc@sheares.nus.edu.sg",
    )
