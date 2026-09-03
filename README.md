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

## Included APIs

- `POST /api/predict` — predicted category, department, confidence, and priority
- `POST /api/complaints` — create a complaint and receive a tracking ID
- `GET /api/complaints/{reference_id}` — retrieve complaint details and timeline
- `POST /api/complaints/{reference_id}/attachments` — add JPG, PNG, or PDF evidence (maximum 5 MB)
- `POST /api/complaints/{reference_id}/feedback` — submit a 1–5 resolution rating
- `PATCH /api/complaints/{reference_id}/status` — update the status timeline (prepare with officer authentication before deployment)

SQLite data and uploaded files are created under `data/`, which is ignored by Git.
