"""
Interactive & Batch Prediction Tester for Public Grievance Classifier
Tests trained models on custom complaints (English, Hindi, Hinglish).
"""

import os
import sys
import io
import pickle
import numpy as np

# Set console encoding to UTF-8 for Hindi/Devanagari text
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Load model artifacts
MODEL_PATH = os.path.join(BASE_DIR, "best_model_aug.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer_aug.pkl")
LABEL_PATH = os.path.join(BASE_DIR, "label_encoder_aug.pkl")

# Fallback to base model if augmented doesn't exist
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
    TFIDF_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
    LABEL_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

print(f"[INFO] Loading model from: {os.path.basename(MODEL_PATH)}...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(TFIDF_PATH, "rb") as f:
    tfidf = pickle.load(f)
with open(LABEL_PATH, "rb") as f:
    le = pickle.load(f)

# Priority / Urgency detector based on keywords
URGENT_KEYWORDS = [
    "fire", "blast", "emergency", "accident", "hospital", "ventilator",
    "blood", "dacoity", "eve teasing", "danger", "electric shock", "burst pipe",
    "death", "aag", "goli", "urgent", "immediate", "hazard"
]

def predict_grievance(text: str):
    clean_text = text.lower().strip()
    vec = tfidf.transform([clean_text])
    pred_idx = model.predict(vec)[0]
    category = le.inverse_transform([pred_idx])[0]
    
    # Calculate confidence / decision margin
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        confidence = float(np.max(probs))
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(vec)[0]
        # Softmax over decision scores for LinearSVC
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)
        confidence = float(np.max(probs))
    
    # Determine priority
    is_urgent = any(kw in clean_text for kw in URGENT_KEYWORDS)
    priority = "High / Urgent" if is_urgent else "Normal / Medium"
    
    return {
        "category": category,
        "confidence": confidence,
        "priority": priority
    }

# 2. Run Built-in Test Cases
sample_complaints = [
    "Transformer blast in Sector 5, no electricity since morning and wires are sparking on road.",
    "Hamare area mein 4 din se drinking water supply nahi aa rahi hai, pipeline toot gayi hai.",
    "थाना सिविल लाइंस में FIR दर्ज नहीं की जा रही है, चोरी की शिकायत पर पुलिस टालमटोल कर रही है।",
    "Old age pension for Sunita Sharma has not been credited to bank account for 2 months.",
    "Main road par bahut bade gaddhe hain, flyover construction ka kaam adhura pada hai.",
    "Sarkari hospital mein doctor absent hain aur emergency ward mein dawai nahi mil rahi.",
    "Clerk in tehsil office is demanding Rs 2000 bribe for issuing land mutation copy.",
    "Ration dealer is charging extra money and giving less grain under PDS scheme.",
    "Municipal corporation is not issuing birth certificate even after 3 weeks of application.",
    "ATM cash out of service and passbook printing machine is not working at the branch."
]

print("\n" + "=" * 75)
print("              PUBLIC GRIEVANCE MODEL TEST RESULTS")
print("=" * 75)

for i, text in enumerate(sample_complaints, 1):
    res = predict_grievance(text)
    print(f"\n[Test #{i}]")
    print(f"Complaint : {text}")
    print(f"-> Predicted Category : \033[92m{res['category']}\033[0m")
    print(f"-> Confidence Score    : {res['confidence'] * 100:.1f}%")
    print(f"-> Assigned Priority   : {res['priority']}")

print("\n" + "=" * 75)
print("Model test suite finished successfully!")
print("=" * 75)

# 3. Interactive Mode (Optional)
if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
    print("\n--- Live Interactive Mode (Type 'exit' to quit) ---")
    while True:
        try:
            user_input = input("\nEnter your grievance text: ")
            if user_input.strip().lower() in ["exit", "quit", "q"]:
                break
            if not user_input.strip():
                continue
            res = predict_grievance(user_input)
            print(f"-> Category  : {res['category']}")
            print(f"-> Confidence: {res['confidence']*100:.1f}%")
            print(f"-> Priority  : {res['priority']}")
        except (KeyboardInterrupt, EOFError):
            break
