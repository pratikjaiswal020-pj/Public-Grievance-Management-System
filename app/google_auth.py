"""Google sign-in for citizen accounts."""
import os

from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv

load_dotenv()


def google_configured() -> bool:
    return bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"))


def redirect_uri() -> str:
    return os.getenv("GOOGLE_REDIRECT_URI", "http://127.0.0.1:8000/api/auth/google/callback")


def build_oauth() -> OAuth | None:
    if not google_configured():
        return None
    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    return oauth


oauth = build_oauth()
