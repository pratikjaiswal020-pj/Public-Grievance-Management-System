# Public Grievance Complaint Management System (PGCMS)
## Implementation Plan

---

## Overview

Build a full-featured **Public Grievance Complaint Management System** — a digital platform where citizens can file, track, and resolve complaints against public services/government departments. The system uses an **NLP/ML model to auto-categorize complaints**, routing them to the right department instantly.

**Tech Stack:**
- **Backend:** Python + FastAPI
- **Frontend:** HTML5 / CSS3 / Vanilla JavaScript (responsive, mobile-first)
- **Database:** SQLite (dev) → PostgreSQL (production)
- **ML/AI:** scikit-learn / transformers for NLP auto-categorization
- **Dataset:** Kaggle Grievance/Complaint dataset (text, sentiment, categories)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    FRONTEND (HTML/CSS/JS)                 │
│  Citizen Portal | Officer Dashboard | Admin Panel        │
└──────────────────┬───────────────────────────────────────┘
                   │ REST API (HTTP/JSON)
┌──────────────────▼───────────────────────────────────────┐
│                   FASTAPI BACKEND                         │
│  Auth │ Complaints │ Routing │ NLP Engine │ Analytics    │
└──────────────────┬───────────────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
   ┌────▼───┐ ┌────▼───┐ ┌───▼────┐
   │ SQLite │ │  ML    │ │ Static │
   │  /PG   │ │ Models │ │ Files  │
   └────────┘ └────────┘ └────────┘
```

---

## Project Structure

```
PUBLIC GRIEVANCE project/
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Configuration & env vars
│   ├── database.py                # DB connection, session management
│   ├── models/
│   │   ├── user.py               # User, Officer, Admin models
│   │   ├── complaint.py          # Complaint, Status, Timeline models
│   │   ├── department.py         # Department, Category models
│   │   └── analytics.py          # Analytics aggregation models
│   ├── schemas/
│   │   ├── complaint.py          # Pydantic request/response schemas
│   │   ├── user.py               # Auth schemas
│   │   └── analytics.py          # Dashboard schemas
│   ├── routers/
│   │   ├── auth.py               # Login, OTP, JWT endpoints
│   │   ├── complaints.py         # File, track, update complaints
│   │   ├── dashboard.py          # Officer/Admin dashboard endpoints
│   │   └── analytics.py          # Reports & analytics endpoints
│   ├── services/
│   │   ├── nlp_service.py        # ML auto-categorization engine
│   │   ├── routing_service.py    # Complaint routing to departments
│   │   ├── notification_service.py  # Email/SMS notifications
│   │   └── analytics_service.py  # Aggregation queries
│   ├── ml/
│   │   ├── train_model.py        # Training script (one-time)
│   │   ├── predict.py            # Inference wrapper
│   │   ├── models/               # Saved .pkl model files
│   │   └── data/                 # Kaggle dataset CSVs
│   └── requirements.txt
├── frontend/
│   ├── index.html                # Landing + Citizen Portal
│   ├── dashboard.html            # Officer/Admin Dashboard
│   ├── analytics.html            # Analytics & Reports page
│   ├── css/
│   │   ├── main.css             # Global design system & tokens
│   │   ├── components.css       # Reusable UI components
│   │   └── dashboard.css        # Dashboard-specific styles
│   └── js/
│       ├── api.js               # Fetch wrapper for backend calls
│       ├── citizen.js           # Citizen complaint flow logic
│       ├── officer.js           # Officer dashboard logic
│       ├── admin.js             # Admin panel logic
│       └── charts.js            # Analytics chart rendering
├── uploads/                      # Complaint attachments storage
└── README.md
```

---

## Phase 1 — ML Model & Data Pipeline

### Step 1: Dataset Processing & Model Training

**Dataset:** Kaggle Grievance/Complaint CSV with columns: `complaint_text`, `category`, `department`, `sentiment`

#### [NEW] `backend/ml/train_model.py`
- Load and clean Kaggle dataset
- Preprocess text (lowercase, remove stopwords, lemmatize)
- TF-IDF vectorizer → Multi-class classifier (Logistic Regression / LinearSVC)
- Train department categorization model + sentiment model
- Save models as `.pkl` files using `joblib`
- Evaluate accuracy (target: >80%)

#### [NEW] `backend/services/nlp_service.py`
- `predict_category(text: str)` → returns `{department, sub_category, confidence}`
- `predict_sentiment(text: str)` → returns `{sentiment, priority_score}`
- Used at complaint submission time

---

## Phase 2 — Backend (FastAPI)

### Step 2: Database Models

#### [NEW] `backend/database.py`
- SQLAlchemy async engine (SQLite dev, Postgres prod)
- Session factory, Base declarative class

#### [NEW] `backend/models/` — 5 core tables:

| Table | Key Columns |
|---|---|
| `users` | id, name, mobile, email, role (citizen/officer/admin), password_hash |
| `departments` | id, name, category_codes[], sla_days |
| `complaints` | id, ref_id (unique), title, description, category, dept_id, status, priority, citizen_id, officer_id, location, created_at |
| `status_timeline` | id, complaint_id, status, changed_by, remark, timestamp |
| `feedback` | id, complaint_id, rating (1–5), comment, submitted_at |

### Step 3: API Endpoints

#### [NEW] `backend/routers/auth.py`
- `POST /auth/register` — citizen registration (mobile + password)
- `POST /auth/login` — JWT token generation
- `GET /auth/me` — current user profile

#### [NEW] `backend/routers/complaints.py`
- `POST /complaints/` — File complaint → triggers NLP auto-categorize + routing
- `GET /complaints/{ref_id}` — Track complaint by reference ID (public)
- `GET /complaints/my` — List citizen's own complaints
- `PUT /complaints/{id}/status` — Officer updates status
- `POST /complaints/{id}/feedback` — Citizen rates resolution
- `POST /complaints/{id}/reopen` — Citizen reopens within 7 days

#### [NEW] `backend/routers/dashboard.py`
- `GET /dashboard/officer` — Officer's assigned queue (filtered by SLA urgency)
- `GET /dashboard/admin` — Admin view with SLA breach alerts
- `PUT /dashboard/complaints/{id}/assign` — Reassign complaint

#### [NEW] `backend/routers/analytics.py`
- `GET /analytics/overview` — KPI cards (total, resolved, pending, avg time)
- `GET /analytics/trends` — Monthly volume chart data
- `GET /analytics/categories` — Category-wise breakdown
- `GET /analytics/sla` — SLA compliance %

---

## Phase 3 — Frontend (HTML/CSS/JS)

### Step 4: Design System & Global Styles

#### [NEW] `frontend/css/main.css`
- CSS custom properties (dark glassmorphism theme: deep navy + electric blue + amber)
- Typography: Google Fonts (Inter)
- Responsive grid system
- Smooth micro-animations and hover effects

### Step 5: Citizen Portal (`index.html`)

**Sections:**
1. **Hero Banner** — Tagline + CTA buttons (File Complaint / Track Complaint)
2. **File Complaint Wizard** (3 steps):
   - Step 1: Personal details + location
   - Step 2: Complaint description (real-time NLP preview shows predicted category)
   - Step 3: Attachments + review + submit
3. **Track Complaint** — Enter reference ID → show animated timeline
4. **Stats Bar** — Total complaints filed, resolved, average resolution time
5. **Login/Register Modal**

### Step 6: Officer/Admin Dashboard (`dashboard.html`)

**Features:**
- Complaint queue table (sortable by SLA urgency, status, category)
- SLA breach warning badges (red/orange indicators)
- Status update modal (officer remarks, evidence upload)
- Reassignment panel (admin only)
- User management (admin only)

### Step 7: Analytics Dashboard (`analytics.html`)

**Charts (using Chart.js):**
- Line chart: Monthly complaint volume trends
- Doughnut chart: Category-wise breakdown
- Bar chart: Department-wise resolution times
- KPI cards: Total, Resolved, Pending, SLA Compliance %

---

## Phase 4 — Integration & Polish

### Step 8: NLP Live Preview (Smart UX Feature)
- As citizen types complaint description, JS calls `/complaints/predict` endpoint
- Returns predicted category in real-time
- Shows "We'll route this to: **Water Department**" suggestion badge

### Step 9: Unique Reference ID System
- Generate alphanumeric IDs: `GRV-2026-XXXXX`
- Used for anonymous tracking (no login needed)

### Step 10: SLA Engine
- `sla_days` configured per department (stored in DB)
- Background task (FastAPI `BackgroundTasks` / APScheduler) checks overdue complaints
- Auto-escalates to admin + flags in dashboard with red badge

---

## Data Flow: Filing a Complaint

```
Citizen fills form
       ↓
Frontend JS sends POST /complaints/
       ↓
FastAPI receives complaint text
       ↓
NLP Service: predict_category() + predict_sentiment()
       ↓
Routing Service: assigns to correct department/officer
       ↓
Database: saves complaint with ref_id "GRV-2026-XXXXX"
       ↓
Returns ref_id + confirmation to frontend
       ↓
Frontend shows success modal with tracking ID
```

---

## Proposed Changes (File by File)

### Backend Foundation

#### [NEW] `backend/requirements.txt`
```
fastapi, uvicorn, sqlalchemy, pydantic, python-jose, passlib,
scikit-learn, pandas, numpy, joblib, nltk, python-multipart, aiofiles, apscheduler
```

#### [NEW] `backend/main.py`
#### [NEW] `backend/config.py`
#### [NEW] `backend/database.py`
#### [NEW] `backend/models/user.py`
#### [NEW] `backend/models/complaint.py`
#### [NEW] `backend/models/department.py`
#### [NEW] `backend/schemas/complaint.py`
#### [NEW] `backend/schemas/user.py`
#### [NEW] `backend/routers/auth.py`
#### [NEW] `backend/routers/complaints.py`
#### [NEW] `backend/routers/dashboard.py`
#### [NEW] `backend/routers/analytics.py`
#### [NEW] `backend/services/nlp_service.py`
#### [NEW] `backend/services/routing_service.py`
#### [NEW] `backend/ml/train_model.py`
#### [NEW] `backend/ml/predict.py`

### Frontend

#### [NEW] `frontend/index.html`
#### [NEW] `frontend/dashboard.html`
#### [NEW] `frontend/analytics.html`
#### [NEW] `frontend/css/main.css`
#### [NEW] `frontend/css/components.css`
#### [NEW] `frontend/js/api.js`
#### [NEW] `frontend/js/citizen.js`
#### [NEW] `frontend/js/officer.js`
#### [NEW] `frontend/js/charts.js`

---

## Verification Plan

### Automated Tests
```bash
# Backend unit tests
pytest backend/tests/

# API endpoint tests
uvicorn backend.main:app --reload
# Then test via Swagger UI: http://localhost:8000/docs
```

### Manual Verification
1. File a complaint → verify NLP auto-categorizes correctly
2. Check reference ID is generated and tracking works without login
3. Login as officer → verify complaint appears in dashboard
4. Update status → verify timeline updates
5. Test SLA badge on overdue complaints
6. Verify analytics charts load with correct data

---

## Open Questions

> [!IMPORTANT]
> **Dataset location**: Please place your Kaggle grievance CSV dataset at `backend/ml/data/grievances.csv`. What are the column names in your downloaded dataset? (e.g., `complaint_text`, `category`, `department`?)

> [!IMPORTANT]
> **Database choice**: Should we use SQLite (simple, no setup) for development or jump straight to PostgreSQL?

> [!NOTE]
> **Languages**: The PRD mentions multilingual support (Hindi, Odia). Should we include multilingual UI in Phase 1 or keep it English-only for now?

> [!NOTE]
> **Notifications**: Should email/SMS notifications be real (requires API keys like Twilio/SendGrid) or simulated for now?

---

## Estimated Timeline

| Phase | Tasks | Duration |
|---|---|---|
| Phase 1 | Dataset prep + ML model training | 1–2 days |
| Phase 2 | FastAPI backend + database | 2–3 days |
| Phase 3 | Frontend (citizen + dashboard + analytics) | 2–3 days |
| Phase 4 | Integration, testing, polish | 1 day |
| **Total** | **Full MVP** | **~1 week** |
