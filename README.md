# Public Grievance Management System

A FastAPI and SQLite citizen portal that uses the included trained classifier to route public grievances.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` for the citizen portal and `http://127.0.0.1:8000/docs` for the API documentation.

## Authentication & Demo Roles

The portal supports **Hybrid Authentication**:
1. **Password Authentication & Self-Service Registration** (enabled by default for immediate local testing & deployment).
2. **Instant 1-Click Demo Logins** on `/login.html`:
   - **System Administrator:** `admin@grievance.gov` (Password: `Admin@1234` or shortcut username: `admin`)
   - **Nodal Officer:** `officer.verma@dept.gov.in` (Password: `Officer@1234` or shortcut username: `officer`)
   - **Citizen User:** `citizen@example.com` (Password: `Citizen@1234` or shortcut username: `citizen`)
3. **Optional Google OAuth & Mobile SMS OTP**:
   To enable Google Sign-In or Twilio SMS OTP, configure `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` or Twilio credentials in a `.env` file (see `.env.example`). When configured, Google Sign-In and SMS OTP activate automatically. Officers can also be provisioned with passwords directly from the Admin console.

## Included APIs

- `POST /api/predict` — predicted category, department, confidence, and priority
- `POST /api/complaints` — create a complaint and receive a tracking ID
- `GET /api/complaints/{reference_id}` — retrieve complaint details and timeline
- `POST /api/complaints/{reference_id}/attachments` — add JPG, PNG, or PDF evidence (maximum 5 MB)
- `POST /api/complaints/{reference_id}/feedback` — submit a 1–5 resolution rating
- `PATCH /api/complaints/{reference_id}/status` — update the status timeline (prepare with officer authentication before deployment)

SQLite data and uploaded files are created under `data/`, which is ignored by Git.
