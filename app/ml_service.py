"""Small, load-once wrapper around the trained grievance classifier."""
from functools import lru_cache
from pathlib import Path
import pickle
import re

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
URGENT_KEYWORDS = {
    "fire", "blast", "emergency", "accident", "hospital", "ventilator", "blood",
    "dacoity", "danger", "electric shock", "burst pipe", "death", "aag", "goli",
    "urgent", "immediate", "hazard", "sparking", "collapse",
}

DEPARTMENT_BY_CATEGORY = {
    "Electricity": "Electricity Department",
    "Water Supply": "Water Supply Department",
    "Sanitation & Garbage": "Municipal Sanitation Department",
    "Roads & Infrastructure": "Public Works Department",
    "Healthcare & Hospitals": "Health Department",
    "Police & Law and Order": "Police Department",
    "Land Records & Revenue": "Revenue Department",
    "Education & Schools": "Education Department",
    "Pension & Provident Fund": "Social Welfare Department",
    "Ration & Public Distribution System": "Food & Civil Supplies Department",
    "Municipal Certificates": "Municipal Corporation",
    "Corruption & Bribery": "Vigilance Department",
    "Employment & Labour": "Labour Department",
    "Banking & Financial Services": "Financial Services Department",
}

# Routing policy. These values should be reviewed against a separate validation
# set whenever the classifier is retrained.
PRIMARY_CONFIDENCE_THRESHOLD = 0.75
MANUAL_REVIEW_THRESHOLD = 0.60
CLOSE_CONFIDENCE_GAP = 0.05

# These explicit service signals let a mixed complaint be separated before it
# is sent to departments. They complement (rather than replace) the trained
# single-label classifier, whose dataset contains one category per row.
CATEGORY_SIGNAL_PATTERNS = {
    "Water Supply": r"\b(?:water|pani|paani|pipeline|borewell|tanker)\b",
    "Electricity": r"\b(?:electricity|bijli|power(?:\s+cut|\s+supply)?|transformer|load\s+shedding)\b",
    "Ration & Public Distribution System": r"\b(?:ration|pds|food|grain|anaj|chawal|gehu|wheat|rice)\b",
    "Sanitation & Garbage": r"\b(?:garbage|kachra|sewage|drain|gandagi|waste)\b",
    "Roads & Infrastructure": r"\b(?:road|sadak|pothole|gaddh[ae]|footpath|bridge|culvert)\b",
    "Healthcare & Hospitals": r"\b(?:hospital|doctor|ambulance|medicine|dawai|health)\b",
    "Police & Law and Order": r"\b(?:police|fir|theft|chori|crime|dacoity)\b",
    "Land Records & Revenue": r"\b(?:land|zameen|mutation|registry|demarcation|patwari)\b",
    "Education & Schools": r"\b(?:school|teacher|scholarship|education)\b",
    "Pension & Provident Fund": r"\b(?:pension|epf|provident\s+fund|pf)\b",
    "Municipal Certificates": r"\b(?:certificate|birth\s+certificate|death\s+certificate|marriage\s+certificate)\b",
    "Corruption & Bribery": r"\b(?:bribe|bribery|corruption|rishwat|ghoos)\b",
    "Employment & Labour": r"\b(?:labour|labor|worker|wages|salary|employment|mgnrega|mnrega)\b",
    "Banking & Financial Services": r"\b(?:bank|atm|kyc|account|loan|cheque)\b",
}


@lru_cache
def _artifacts():
    candidates = [
        ("best_model_aug.pkl", "tfidf_vectorizer_aug.pkl", "label_encoder_aug.pkl"),
        ("best_model.pkl", "tfidf_vectorizer.pkl", "label_encoder.pkl"),
    ]
    for model_name, vectorizer_name, labels_name in candidates:
        paths = [ROOT / name for name in (model_name, vectorizer_name, labels_name)]
        if all(path.exists() for path in paths):
            with paths[0].open("rb") as model_file, paths[1].open("rb") as vectorizer_file, paths[2].open("rb") as labels_file:
                svm_path = ROOT / model_name.replace("best_model", "router_svm_model")
                svm = None
                if svm_path.exists():
                    with svm_path.open("rb") as svm_file:
                        svm = pickle.load(svm_file)
                return pickle.load(model_file), svm, pickle.load(vectorizer_file), pickle.load(labels_file)
    raise RuntimeError("No trained classifier artifacts were found in the project root.")


def _prediction(model, vector, encoder) -> dict:
    """Return a calibrated category probability for one already-trained model."""
    probabilities = model.predict_proba(vector)[0]
    predicted_index = int(np.argmax(probabilities))
    return {
        "category": encoder.inverse_transform([predicted_index])[0],
        "confidence": float(probabilities[predicted_index]),
    }


def _predict_single(text: str) -> dict:
    primary_model, svm_model, vectorizer, encoder = _artifacts()
    cleaned = text.lower().strip()
    vector = vectorizer.transform([cleaned])
    primary = _prediction(primary_model, vector, encoder)
    selected = primary
    selected_model = "Logistic Regression"
    routing_reason = "Primary model confidence is at least 75%."
    manual_review = False
    model_predictions = {"logistic_regression": {"category": primary["category"], "confidence": round(primary["confidence"] * 100, 1)}}

    # The SVM is intentionally evaluated only for uncertain primary predictions.
    if primary["confidence"] < PRIMARY_CONFIDENCE_THRESHOLD:
        if svm_model is None:
            manual_review = primary["confidence"] < MANUAL_REVIEW_THRESHOLD
            routing_reason = "The calibrated SVM artifact is unavailable; using the primary-model result."
        else:
            svm = _prediction(svm_model, vector, encoder)
            model_predictions["calibrated_linear_svm"] = {"category": svm["category"], "confidence": round(svm["confidence"] * 100, 1)}

            if svm["category"] == primary["category"]:
                if svm["confidence"] > primary["confidence"]:
                    selected, selected_model = svm, "Calibrated Linear SVM"
                routing_reason = "Both models agree on the category."
                if selected["confidence"] < MANUAL_REVIEW_THRESHOLD:
                    manual_review = True
                    routing_reason = "Both models agree, but their calibrated confidence is below 60%."
            elif max(primary["confidence"], svm["confidence"]) < MANUAL_REVIEW_THRESHOLD:
                manual_review = True
                routing_reason = "Both models are below 60% confidence and disagree."
            elif abs(primary["confidence"] - svm["confidence"]) <= CLOSE_CONFIDENCE_GAP:
                manual_review = True
                routing_reason = "The models disagree and their confidence scores are within 5 percentage points."
            elif svm["confidence"] > primary["confidence"]:
                selected, selected_model = svm, "Calibrated Linear SVM"
                routing_reason = "The models disagree; the SVM has materially higher calibrated confidence."
            else:
                routing_reason = "The models disagree; Logistic Regression has materially higher confidence."

    category = selected["category"]
    confidence = selected["confidence"]
    urgent = any(keyword in cleaned for keyword in URGENT_KEYWORDS)
    return {
        "category": category,
        "department": DEPARTMENT_BY_CATEGORY.get(category, "Public Grievance Cell"),
        "confidence": round(confidence * 100, 1),
        "priority": "High" if urgent else "Normal",
        "manual_review_recommended": manual_review,
        "selected_model": selected_model,
        "routing_reason": routing_reason,
        "model_predictions": model_predictions,
    }


def _issue_segments(text: str) -> list[str]:
    """Split common lists of problems while retaining the user's wording."""
    return [
        segment.strip(" ,:-")
        for segment in re.split(r"(?:[.!?;\n]+|\b(?:and|aur)\s+(?=(?:no|not|without|lack|nahi|nahin)\b))", text, flags=re.IGNORECASE)
        if segment.strip(" ,:-")
    ]


def detect_issues(text: str) -> list[dict]:
    """Find distinct service issues in a complaint and keep their evidence."""
    issues: dict[str, dict] = {}
    for segment in _issue_segments(text):
        matched_categories = [
            category for category, pattern in CATEGORY_SIGNAL_PATTERNS.items()
            if re.search(pattern, segment, flags=re.IGNORECASE)
        ]
        if not matched_categories:
            continue
        route = _predict_single(segment)
        for category in matched_categories:
            issue = issues.setdefault(category, {
                "category": category,
                "department": DEPARTMENT_BY_CATEGORY[category],
                "confidence": None,
                "detection_method": "explicit service keyword",
                "manual_review_recommended": False,
                "priority": route["priority"],
                "evidence": [],
            })
            issue["evidence"].append(segment)
            if route["category"] == category:
                issue["confidence"] = route["confidence"]
                issue["detection_method"] = "explicit service keyword and classifier"
                issue["manual_review_recommended"] = route["manual_review_recommended"]
            if route["priority"] == "High":
                issue["priority"] = "High"

    if not issues:
        route = _predict_single(text)
        issues[route["category"]] = {
            "category": route["category"],
            "department": route["department"],
            "confidence": route["confidence"],
            "detection_method": "classifier",
            "manual_review_recommended": route["manual_review_recommended"],
            "priority": route["priority"],
            "evidence": [text.strip()],
        }

    return [
        {**issue, "evidence": " ".join(dict.fromkeys(issue["evidence"]))}
        for issue in issues.values()
    ]


def predict(text: str) -> dict:
    """Route the primary issue and expose all detected service issues."""
    result = _predict_single(text)
    issues = detect_issues(text)
    result["issues"] = issues
    result["multiple_issues_detected"] = len(issues) > 1
    result["issue_confirmation_required"] = len(issues) > 1
    return result
