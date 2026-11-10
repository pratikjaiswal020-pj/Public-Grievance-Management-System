"""Twilio Verify integration for mobile OTP authentication."""
import os

import httpx


TWILIO_VERIFY_URL = "https://verify.twilio.com/v2/Services"


def sms_configured() -> bool:
    return all(os.getenv(name) for name in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_VERIFY_SERVICE_SID"))


def _request(path: str, data: dict[str, str]) -> dict:
    if not sms_configured():
        raise RuntimeError("Mobile sign-in is not configured.")
    service_sid = os.environ["TWILIO_VERIFY_SERVICE_SID"]
    with httpx.Client(timeout=15) as client:
        response = client.post(
            f"{TWILIO_VERIFY_URL}/{service_sid}/{path}",
            data=data,
            auth=(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"]),
        )
    if response.is_error:
        raise RuntimeError("Unable to send or verify the code. Please try again shortly.")
    return response.json()


def request_otp(phone: str) -> None:
    _request("Verifications", {"To": phone, "Channel": "sms"})


def verify_otp(phone: str, code: str) -> bool:
    return _request("VerificationCheck", {"To": phone, "Code": code}).get("status") == "approved"
