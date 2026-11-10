"""Send a complaint confirmation to the citizen's email address."""
import os
import smtplib
from email.message import EmailMessage


def smtp_configured() -> bool:
    return all(os.getenv(name) for name in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM"))


def send_complaint_email(*, to: str, name: str, title: str, tickets: list[dict]) -> tuple[bool, str | None]:
    if not smtp_configured():
        return False, "Email is not configured. Add SMTP_HOST, SMTP_USER, SMTP_PASSWORD, and SMTP_FROM to the .env file."

    base = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    lines = [
        f"Hello {name},",
        "",
        "Your public grievance was received and linked to this Google account.",
        f"Subject: {title}",
        "",
        "Tracking details:",
    ]
    for ticket in tickets:
        lines.append(
            f"- {ticket['reference_id']} · {ticket['category']} · {ticket['department']} · {base}/#track"
        )
    lines.extend([
        "",
        "Keep these reference IDs. You can track each ticket on the portal.",
        "",
        "CivicPulse Public Grievance Management System",
    ])

    message = EmailMessage()
    message["Subject"] = f"Complaint received — {tickets[0]['reference_id']}"
    message["From"] = os.getenv("SMTP_FROM")
    message["To"] = to
    message.set_content("\n".join(lines))

    host = os.getenv("SMTP_HOST", "")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "")
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=20) as server:
                server.login(user, password)
                server.send_message(message)
        else:
            with smtplib.SMTP(host, port, timeout=20) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(user, password)
                server.send_message(message)
    except Exception as exc:
        return False, f"The mail server rejected the confirmation email ({exc.__class__.__name__})."
    return True, None
