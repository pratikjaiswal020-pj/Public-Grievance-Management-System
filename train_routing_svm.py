"""Build the calibrated SVM second-opinion artifact used by request routing.

This uses the existing TF-IDF and label artifacts, and the same deterministic
train/validation split as augment_and_retrain.py. It never runs during an API
request. Re-run after retraining the main classifier.
"""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "grievances_augmented.csv"
VECTORIZER_PATH = ROOT / "tfidf_vectorizer_aug.pkl"
ENCODER_PATH = ROOT / "label_encoder_aug.pkl"
OUTPUT_PATH = ROOT / "router_svm_model_aug.pkl"


def main() -> None:
    if not all(path.exists() for path in (DATA_PATH, VECTORIZER_PATH, ENCODER_PATH)):
        raise RuntimeError("Missing augmented data or primary-model artifacts. Run augment_and_retrain.py first.")

    frame = pd.read_csv(DATA_PATH)
    texts = frame["text"].astype(str).str.lower().str.strip()
    with VECTORIZER_PATH.open("rb") as vectorizer_file, ENCODER_PATH.open("rb") as encoder_file:
        vectorizer = pickle.load(vectorizer_file)
        encoder = pickle.load(encoder_file)
    labels = encoder.transform(frame["category"])
    train_texts, _, train_labels, _ = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    train_vectors = vectorizer.transform(train_texts)
    svm = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2500, random_state=42), cv=5, method="sigmoid"
    )
    svm.fit(train_vectors, train_labels)
    with OUTPUT_PATH.open("wb") as output_file:
        pickle.dump(svm, output_file)
    print(f"Saved calibrated routing model: {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
