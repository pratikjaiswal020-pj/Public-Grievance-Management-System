from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator
from sqlalchemy import func, inspect, or_, select, text
from sqlalchemy.orm import Session, selectinload

from .auth import (
    create_access_token,
    get_current_user,
    get_password_hash,
    require_officer,
    verify_password,
)
from .database import Base, DATA_DIR, engine, get_db
from .ml_service import predict
from .models import Attachment, Complaint, Feedback, StatusEvent, User


UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "application/pdf"}
MAX_UPLOAD_SIZE = 5 * 1024 * 1024

# Initialise the local SQLite schema. Use Alembic for deployed DBs.
Base.metadata.create_all(bind=engine)


def ensure_schema_upgrades() -> None:
    """Apply the small SQLite upgrades needed for linked multi-issue tickets."""
    columns = {column["name"] for column in inspect(engine).get_columns("complaints")}
    if "parent_reference_id" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE complaints ADD COLUMN parent_reference_id VARCHAR(32)"))


def seed_admin(db: Session) -> None:
    """Create a default admin account on first startup."""
    exists = db.scalar(select(User).where(User.role == "admin"))
    if not exists:
        admin = User(
            name="System Administrator",
            email="admin@grievance.gov",
            hashed_password=get_password_hash("Admin@1234"),
            role="admin",
        )
        db.add(admin)
        db.commit()


ensure_schema_upgrades()

# Seed the default admin using a one-off session.
from .database import SessionLocal as _SessionLocal
_seed_db = _SessionLocal()
try:
    seed_admin(_seed_db)
finally:
    _seed_db.close()


app = FastAPI(title="Public Grievance Management System", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Narrow to the deployed frontend origin in production.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────── Pydantic schemas ────────────────────────────

class PredictionRequest(BaseModel):
    text: str = Field(min_length=10, max_length=5000)


class ComplaintCreate(BaseModel):
    citizen_name: str = Field(min_length=2, max_length=120)
    citizen_email: EmailStr | None = None
    citizen_phone: str | None = Field(default=None, max_length=32)
    title: str = Field(min_length=5, max_length=180)
    description: str = Field(min_length=10, max_length=5000)
    location: str | None = Field(default=None, max_length=255)
    issue_categories: list[str] | None = Field(default=None, max_length=14)

    @field_validator("citizen_email", "citizen_phone", "location", mode="before")
    @classmethod
    def blank_optional_fields_are_none(cls, value):
        return None if isinstance(value, str) and not value.strip() else value


class FeedbackCreate(BaseModel):
    rating: int = Field(ge=1, le=5)
    comment: str | None = Field(default=None, max_length=2000)


class StatusUpdate(BaseModel):
    status: str = Field(pattern="^(Submitted|Assigned|In Progress|Resolved)$")
    remark: str = Field(min_length=3, max_length=2000)


class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: str | None = None
    username: str | None = None
    password: str

    @model_validator(mode="before")
    @classmethod
    def normalize_login_payload(cls, data):
        if isinstance(data, dict):
            identifier = data.get("email") or data.get("username")
            if identifier is not None:
                data["email"] = str(identifier).strip()
            if "password" in data and data["password"] is not None:
                data["password"] = str(data["password"])
        return data


# ──────────────────────────────── Helpers ────────────────────────────────────

def generate_reference_id(db: Session) -> str:
    while True:
        reference_id = f"GRV-{datetime.now():%Y}-{uuid4().hex[:6].upper()}"
        if not db.scalar(select(Complaint.id).where(Complaint.reference_id == reference_id)):
            return reference_id


def complaint_or_404(reference_id: str, db: Session) -> Complaint:
    statement = (
        select(Complaint)
        .where(Complaint.reference_id == reference_id.upper())
        .options(selectinload(Complaint.attachments), selectinload(Complaint.timeline), selectinload(Complaint.feedback))
    )
    complaint = db.scalar(statement)
    if not complaint:
        raise HTTPException(status_code=404, detail="No complaint was found with this reference ID.")
    return complaint


def serialize_complaint(complaint: Complaint, db: Session | None = None) -> dict:
    data = {
        "reference_id": complaint.reference_id,
        "citizen_name": complaint.citizen_name,
        "citizen_email": complaint.citizen_email,
        "citizen_phone": complaint.citizen_phone,
        "title": complaint.title,
        "description": complaint.description,
        "location": complaint.location,
        "category": complaint.category,
        "department": complaint.department,
        "confidence": complaint.confidence,
        "priority": complaint.priority,
        "status": complaint.status,
        "created_at": complaint.created_at.isoformat(),
        "updated_at": complaint.updated_at.isoformat(),
        "attachments": [
            {"name": item.original_name, "url": f"/uploads/{item.stored_name}", "uploaded_at": item.uploaded_at.isoformat()}
            for item in complaint.attachments
        ],
        "timeline": [
            {"status": event.status, "remark": event.remark, "created_at": event.created_at.isoformat()}
            for event in sorted(complaint.timeline, key=lambda item: item.created_at)
        ],
        "feedback": (
            {"rating": complaint.feedback.rating, "comment": complaint.feedback.comment, "created_at": complaint.feedback.created_at.isoformat()}
            if complaint.feedback else None
        ),
    }
    if db:
        group_reference = complaint.parent_reference_id or complaint.reference_id
        related = db.scalars(
            select(Complaint)
            .where((Complaint.reference_id == group_reference) | (Complaint.parent_reference_id == group_reference))
            .order_by(Complaint.created_at)
        ).all()
        data["group_reference_id"] = group_reference
        data["related_complaints"] = [
            {
                "reference_id": item.reference_id,
                "category": item.category,
                "department": item.department,
                "status": item.status,
                "is_current": item.reference_id == complaint.reference_id,
            }
            for item in related
        ]
    return data


# ──────────────────────────── Auth endpoints ──────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> dict:
    """Citizen self-registration. Officers/admins are created by admin only."""
    clean_email = payload.email.strip().lower()
    if "@" not in clean_email or "." not in clean_email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    existing = db.scalar(select(User).where(func.lower(User.email) == clean_email))
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    user = User(
        name=payload.name.strip(),
        email=clean_email,
        hashed_password=get_password_hash(payload.password),
        role="citizen",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(user.id, user.email, user.name, user.role)
    return {"access_token": token, "token_type": "bearer", "role": user.role, "name": user.name}


@app.post("/api/auth/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> dict:
    """Login for all roles. Accepts email, username, or 'admin'."""
    identifier = (payload.email or payload.username or "").strip().lower()
    if not identifier:
        raise HTTPException(status_code=400, detail="Email or username is required.")

    # Match by email (case-insensitive) or full name (case-insensitive)
    user = db.scalar(
        select(User).where(
            or_(
                func.lower(User.email) == identifier,
                func.lower(User.name) == identifier,
            )
        )
    )
    # Also support entering "admin" as shortcut for the admin account
    if not user and identifier in ("admin", "administrator", "admin@grievance"):
        user = db.scalar(select(User).where(User.role == "admin"))

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email/username or password.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated.")
    token = create_access_token(user.id, user.email, user.name, user.role)
    return {"access_token": token, "token_type": "bearer", "role": user.role, "name": user.name}


@app.get("/api/auth/me")
def me(current_user: User = Depends(get_current_user)) -> dict:
    return {"id": current_user.id, "name": current_user.name, "email": current_user.email, "role": current_user.role}


@app.post("/api/admin/officers", status_code=status.HTTP_201_CREATED)
def create_officer(payload: RegisterRequest, db: Session = Depends(get_db), _admin: User = Depends(require_officer)) -> dict:
    """Admin creates officer accounts."""
    existing = db.scalar(select(User).where(User.email == str(payload.email)))
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    officer = User(
        name=payload.name.strip(),
        email=str(payload.email),
        hashed_password=get_password_hash(payload.password),
        role="officer",
    )
    db.add(officer)
    db.commit()
    db.refresh(officer)
    return {"id": officer.id, "name": officer.name, "email": officer.email, "role": officer.role}


# ──────────────────────── Admin dashboard endpoints ──────────────────────────

@app.get("/api/admin/complaints")
def list_all_complaints(
    page: int = 1,
    per_page: int = 20,
    status_filter: str | None = None,
    priority_filter: str | None = None,
    search: str | None = None,
    db: Session = Depends(get_db),
    _officer: User = Depends(require_officer),
) -> dict:
    """Paginated list of all complaints for officer/admin dashboard."""
    from sqlalchemy import func, or_
    stmt = select(Complaint).options(
        selectinload(Complaint.timeline),
        selectinload(Complaint.attachments),
        selectinload(Complaint.feedback),
    ).order_by(Complaint.created_at.desc())

    filters = []
    if status_filter:
        filters.append(Complaint.status == status_filter)
    if priority_filter:
        filters.append(Complaint.priority == priority_filter)
    if search and search.strip():
        term = f"%{search.strip()}%"
        filters.append(
            or_(
                Complaint.reference_id.ilike(term),
                Complaint.citizen_name.ilike(term),
                Complaint.title.ilike(term),
                Complaint.category.ilike(term),
                Complaint.department.ilike(term),
                Complaint.location.ilike(term),
            )
        )

    if filters:
        stmt = stmt.where(*filters)

    count_stmt = select(func.count()).select_from(Complaint)
    if filters:
        count_stmt = count_stmt.where(*filters)
    total = db.scalar(count_stmt) or 0

    complaints = db.scalars(stmt.offset((page - 1) * per_page).limit(per_page)).all()
    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "complaints": [serialize_complaint(c) for c in complaints],
    }



@app.get("/api/admin/stats")
def admin_stats(db: Session = Depends(get_db), _officer: User = Depends(require_officer)) -> dict:
    """KPI summary for the admin dashboard."""
    from sqlalchemy import func
    total = db.scalar(select(func.count()).select_from(Complaint)) or 0
    resolved = db.scalar(select(func.count()).select_from(Complaint).where(Complaint.status == "Resolved")) or 0
    in_progress = db.scalar(select(func.count()).select_from(Complaint).where(Complaint.status == "In Progress")) or 0
    assigned = db.scalar(select(func.count()).select_from(Complaint).where(Complaint.status == "Assigned")) or 0
    submitted = db.scalar(select(func.count()).select_from(Complaint).where(Complaint.status == "Submitted")) or 0
    high_priority = db.scalar(select(func.count()).select_from(Complaint).where(Complaint.priority == "High")) or 0
    return {
        "total": total,
        "resolved": resolved,
        "in_progress": in_progress,
        "assigned": assigned,
        "submitted": submitted,
        "high_priority": high_priority,
    }


# ─────────────────────── Complaint endpoints (public) ────────────────────────

@app.post("/api/predict")
def classify_complaint(payload: PredictionRequest) -> dict:
    return predict(payload.text)


@app.post("/api/complaints", status_code=status.HTTP_201_CREATED)
def create_complaint(
    payload: ComplaintCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    result = predict(payload.description)
    issues_by_category = {issue["category"]: issue for issue in result["issues"]}
    if payload.issue_categories is None:
        selected_categories = list(issues_by_category)
    else:
        selected_categories = list(dict.fromkeys(payload.issue_categories))
        if not selected_categories:
            raise HTTPException(status_code=422, detail="Select at least one detected issue.")
        invalid_categories = [category for category in selected_categories if category not in issues_by_category]
        if invalid_categories:
            raise HTTPException(status_code=422, detail="Issue selections must match the detected service issues.")

    complaints: list[Complaint] = []
    parent_reference_id: str | None = None
    citizen_name = payload.citizen_name.strip() if payload.citizen_name else current_user.name
    citizen_email = str(payload.citizen_email) if payload.citizen_email else current_user.email

    for index, category in enumerate(selected_categories):
        issue = issues_by_category[category]
        complaint = Complaint(
            reference_id=generate_reference_id(db),
            parent_reference_id=parent_reference_id,
            citizen_name=citizen_name,
            citizen_email=citizen_email,
            citizen_phone=payload.citizen_phone.strip() if payload.citizen_phone else None,
            title=payload.title.strip() if index == 0 else f"{payload.title.strip()[:140]} — {category}",
            description=issue["evidence"] or payload.description.strip(),
            location=payload.location.strip() if payload.location else None,
            category=category,
            department=issue["department"],
            confidence=issue["confidence"] or 0,
            priority=issue["priority"],
        )
        db.add(complaint)
        db.flush()
        if parent_reference_id is None:
            parent_reference_id = complaint.reference_id
        review_remark = (
            "Complaint received and queued for manual category review."
            if issue["manual_review_recommended"]
            else "Complaint received and routed automatically."
        )
        db.add(StatusEvent(complaint_id=complaint.id, status="Submitted", remark=review_remark))
        complaints.append(complaint)
    db.commit()
    for complaint in complaints:
        db.refresh(complaint)
    primary = complaints[0]
    return {
        "reference_id": primary.reference_id,
        "status": primary.status,
        "multiple_tickets_created": len(complaints) > 1,
        "related_complaints": [
            {"reference_id": item.reference_id, "category": item.category, "department": item.department}
            for item in complaints
        ],
        **result,
    }


@app.get("/api/complaints/{reference_id}")
def get_complaint(reference_id: str, db: Session = Depends(get_db)) -> dict:
    return serialize_complaint(complaint_or_404(reference_id, db), db)


@app.post("/api/complaints/{reference_id}/attachments", status_code=status.HTTP_201_CREATED)
def upload_attachment(reference_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict:
    complaint = complaint_or_404(reference_id, db)
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Only JPG, PNG, and PDF files are allowed.")
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".pdf"}:
        raise HTTPException(status_code=415, detail="Unsupported file extension.")
    content = file.file.read(MAX_UPLOAD_SIZE + 1)
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="Attachment must be 5 MB or smaller.")
    stored_name = f"{uuid4().hex}{suffix}"
    target = UPLOAD_DIR / stored_name
    with target.open("wb") as destination:
        destination.write(content)
    attachment = Attachment(
        complaint_id=complaint.id,
        original_name=Path(file.filename or "attachment").name,
        stored_name=stored_name,
        content_type=file.content_type,
    )
    db.add(attachment)
    db.commit()
    return {"name": attachment.original_name, "url": f"/uploads/{stored_name}"}


@app.post("/api/complaints/{reference_id}/feedback", status_code=status.HTTP_201_CREATED)
def add_feedback(reference_id: str, payload: FeedbackCreate, db: Session = Depends(get_db)) -> dict:
    complaint = complaint_or_404(reference_id, db)
    if complaint.feedback:
        raise HTTPException(status_code=409, detail="Feedback has already been submitted for this complaint.")
    feedback = Feedback(complaint_id=complaint.id, rating=payload.rating, comment=payload.comment.strip() if payload.comment else None)
    db.add(feedback)
    db.commit()
    return {"message": "Thank you for your feedback."}


@app.patch("/api/complaints/{reference_id}/status")
def update_status(
    reference_id: str,
    payload: StatusUpdate,
    db: Session = Depends(get_db),
    _officer: User = Depends(require_officer),   # Now protected — officer/admin only.
) -> dict:
    complaint = complaint_or_404(reference_id, db)
    complaint.status = payload.status
    db.add(StatusEvent(complaint_id=complaint.id, status=payload.status, remark=payload.remark.strip()))
    db.commit()
    return serialize_complaint(complaint_or_404(reference_id, db), db)


app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
