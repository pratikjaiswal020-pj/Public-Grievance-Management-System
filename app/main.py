from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session, selectinload

from .database import Base, DATA_DIR, engine, get_db
from .ml_service import predict
from .models import Attachment, Complaint, Feedback, StatusEvent


UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "application/pdf"}
MAX_UPLOAD_SIZE = 5 * 1024 * 1024

# This small project has no migrations yet, so initialise its local SQLite schema
# when the application is imported. Use Alembic migrations when evolving a deployed DB.
Base.metadata.create_all(bind=engine)


def ensure_schema_upgrades() -> None:
    """Apply the small SQLite upgrade needed for linked multi-issue tickets."""
    columns = {column["name"] for column in inspect(engine).get_columns("complaints")}
    if "parent_reference_id" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE complaints ADD COLUMN parent_reference_id VARCHAR(32)"))


ensure_schema_upgrades()

app = FastAPI(title="Public Grievance Management System", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Narrow to the deployed frontend origin in production.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/predict")
def classify_complaint(payload: PredictionRequest) -> dict:
    return predict(payload.text)


@app.post("/api/complaints", status_code=status.HTTP_201_CREATED)
def create_complaint(payload: ComplaintCreate, db: Session = Depends(get_db)) -> dict:
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
    for index, category in enumerate(selected_categories):
        issue = issues_by_category[category]
        complaint = Complaint(
            reference_id=generate_reference_id(db),
            parent_reference_id=parent_reference_id,
            citizen_name=payload.citizen_name.strip(),
            citizen_email=str(payload.citizen_email) if payload.citizen_email else None,
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
def update_status(reference_id: str, payload: StatusUpdate, db: Session = Depends(get_db)) -> dict:
    """Officer-facing endpoint. Add authentication/role checks before production use."""
    complaint = complaint_or_404(reference_id, db)
    complaint.status = payload.status
    db.add(StatusEvent(complaint_id=complaint.id, status=payload.status, remark=payload.remark.strip()))
    db.commit()
    return serialize_complaint(complaint_or_404(reference_id, db), db)


app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
