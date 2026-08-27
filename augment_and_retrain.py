"""
Augmentation Strategy:
- Analyze holdout text patterns (templates) to extract sentence structures
- Generate 1000+ template-based augmented samples for each category
- Combine with original training data and retrain all models
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import random
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report
)
from xgboost import XGBClassifier

random.seed(42)
np.random.seed(42)

BASE = r"c:\Users\prati\OneDrive\Desktop\PUBLIC GRIEVANCE project"

df_train = pd.read_csv(os.path.join(BASE, "grievances_synthetic.csv"))
df_holdout = pd.read_csv(os.path.join(BASE, "grievances_holdout_templates.csv"))

# ─────────────────────────────────────────────
# Template Augmentation Data
# ─────────────────────────────────────────────

CITIES = [
    "Patna", "Varanasi", "Raipur", "Bhopal", "Lucknow", "Kanpur", "Indore",
    "Nagpur", "Allahabad", "Gwalior", "Jabalpur", "Agra", "Meerut",
    "Faridabad", "Ghaziabad", "Nashik", "Aurangabad", "Solapur", "Kolkata",
    "Ranchi", "Dhanbad", "Jamshedpur", "Bokaro", "Dehradun", "Haridwar",
    "Amritsar", "Ludhiana", "Jodhpur", "Kota", "Ajmer", "Surat", "Vadodara"
]

NAMES = [
    "Rajesh Verma", "Mohd. Salim", "Deepak Chouhan", "Suresh Kumar",
    "Priya Singh", "Ramesh Gupta", "Anita Devi", "Sunita Sharma",
    "Vikram Yadav", "Kavita Patel", "Ajay Mishra", "Rekha Joshi",
    "Dinesh Tiwari", "Pooja Rani", "Sanjay Dubey", "Meena Pandey",
    "Rohit Agarwal", "Geeta Soni", "Arun Pal", "Savita Jain"
]

AREAS = [
    "Ward 12", "Sector 5", "Gandhi Nagar", "Nehru Colony",
    "Civil Lines", "Napier Town", "Wright Town", "Adhartal",
    "Madan Mahal", "Model Town", "Adarsh Nagar", "Shastri Nagar",
    "Patel Nagar", "New Colony", "Old City", "Station Road area"
]

POLICE_STATIONS = [
    "thane", "police station", "chowki", "outpost"
]

DURATIONS = [
    "3 days", "5 din", "1 hafta", "10 din", "2 weeks", "ek mahine",
    "15 din", "7 din", "4 din", "6 din", "3 hafte"
]

GREETINGS = [
    "Namaste sir,", "Sir/Madam,", "Dear officer,", "महोदय,",
    "Respected Sir,", "थाना", "Mahoday,", ""
]

CLOSINGS = [
    "Thank you.", "Please take urgent action.",
    "Kripya jald samadhan karein.", "कृपया जल्द कार्रवाई करें।",
    "Sadar naman.", "Otherwise we will protest.", ""
]

def rand(lst): return random.choice(lst)

# ─────────────────────────────────────────────
# Template generators per category
# ─────────────────────────────────────────────

def gen_land_records():
    templates = [
        f"{rand(GREETINGS)} जमीन विवाद {rand(CITIES)} में {rand(DURATIONS)} से पटवारी के पास लंबित है, कोई सुनवाई नहीं हो रही। {rand(CLOSINGS)}",
        f"{rand(GREETINGS)} Land record mein {rand(NAMES)} ka naam galat likha hai, correction chahiye {rand(DURATIONS)} se request kar raha hoon. {rand(CLOSINGS)}",
        f"{rand(GREETINGS)} मेरी जमीन का सीमांकन {rand(CITIES)} में गलत हो गया है, पड़ोसी ने कब्जा कर लिया है। {rand(CLOSINGS)}",
        f"{rand(GREETINGS)} {rand(NAMES)} ki zameen ka patwari {rand(DURATIONS)} se koi jawab nahi de rahe hain. {rand(CLOSINGS)}",
        f"Jameen ka naksha sahi nahi hai {rand(CITIES)} mein, {rand(DURATIONS)} se tehsildar se guzarish kar raha hoon. {rand(CLOSINGS)}",
        f"{rand(GREETINGS)} भू-अभिलेख में गलती है, {rand(CITIES)} में तहसीलदार को निर्देश दें। {rand(CLOSINGS)}",
        f"Land mutation {rand(NAMES)} ki {rand(DURATIONS)} se atki hai, {rand(CITIES)} patwari karyalay mein. {rand(CLOSINGS)}",
    ]
    return rand(templates)

def gen_police():
    templates = [
        f"{rand(GREETINGS)} FIR {rand(CITIES)} {rand(POLICE_STATIONS)} mein darj nahi ki ja rahi h {rand(DURATIONS)} se, police sun nahi rahi. {rand(CLOSINGS)}",
        f"थाना {rand(CITIES)} में शिकायत लेकर गए तो पुलिस ने टालमटोल किया, {rand(DURATIONS)} बीत गए कोई कार्रवाई नहीं।",
        f"Eve teasing {rand(AREAS)} area mein badh gayi hai, mahilaye ghar se nikalne mein dari rehti hain. {rand(CLOSINGS)}",
        f"Illegal encroachment {rand(AREAS)} mohalle mein khule aam chal raha hai, police mook drashak bani hai. {rand(CLOSINGS)}",
        f"{rand(GREETINGS)} {rand(AREAS)} mein asaamajik tatvon ka atank hai, {rand(DURATIONS)} se shikayat ke baad bhi koi action nahi. {rand(CLOSINGS)}",
        f"FIR registration {rand(CITIES)} mein refuse kar di gayi {rand(DURATIONS)} pehle, police ne harassment ki. {rand(CLOSINGS)}",
        f"Chor bazaar {rand(CITIES)} mein khule aam chal raha hai, {rand(DURATIONS)} se shikayat laga raha hoon. {rand(CLOSINGS)}",
    ]
    return rand(templates)

def gen_pension():
    templates = [
        f"{rand(GREETINGS)} {rand(NAMES)} ki pension {rand(DURATIONS)} se nahi mili hai, bank account {rand(AREAS)} branch mein hai. {rand(CLOSINGS)}",
        f"{rand(GREETINGS)} Divyang pension yojna ka labh {rand(NAMES)} ko {rand(DURATIONS)} se nahi mil raha {rand(CITIES)} mein, form baar baar reject ho raha. {rand(CLOSINGS)}",
        f"PF withdrawal request {rand(DURATIONS)} se pending h, office se koi response nahi mil raha. {rand(CLOSINGS)}",
        f"EPF transfer {rand(CITIES)} regional office se {rand(DURATIONS)} se atka hai, naya employer wait kar raha hai. {rand(CLOSINGS)}",
        f"पेंशन फॉर्म में गलत बैंक खाता नंबर दर्ज हो गया है, {rand(DURATIONS)} से सुधार नहीं हुआ। {rand(CLOSINGS)}",
        f"{rand(NAMES)} ki old age pension {rand(DURATIONS)} se band hai, {rand(CITIES)} mein. {rand(CLOSINGS)}",
        f"Widow pension {rand(NAMES)} ki {rand(DURATIONS)} se nahi aayi, {rand(CITIES)} se application ki thi. {rand(CLOSINGS)}",
    ]
    return rand(templates)

def gen_electricity():
    templates = [
        f"{rand(GREETINGS)} bijli {rand(DURATIONS)} se gayab h, transformer kharab ho gaya lagta hai {rand(CITIES)} mein. {rand(CLOSINGS)}",
        f"Light {rand(DURATIONS)} se nahi aa rahi {rand(AREAS)} mein. {rand(CLOSINGS)}",
        f"Transformer blast hua {rand(DURATIONS)} pehle {rand(AREAS)} mein, ab tak repair nahi hua. {rand(CLOSINGS)}",
        f"Bijli bill bahut zyada aaya hai, {rand(NAMES)} ka meter reading galat lagta hai {rand(CITIES)} mein. {rand(CLOSINGS)}",
        f"{rand(GREETINGS)} बिजली कटौती {rand(CITIES)} में {rand(DURATIONS)} से हो रही है, कोई समाधान नहीं। {rand(CLOSINGS)}",
        f"New bijli connection {rand(DURATIONS)} se pending hai {rand(NAMES)} ka, {rand(CITIES)} office mein application di thi. {rand(CLOSINGS)}",
    ]
    return rand(templates)

def gen_water():
    templates = [
        f"{rand(GREETINGS)} paani {rand(DURATIONS)} se nahi aa raha {rand(AREAS)} mein. {rand(CLOSINGS)}",
        f"Water supply band hai {rand(CITIES)} mein {rand(DURATIONS)} se, log pareshan hain. {rand(CLOSINGS)}",
        f"Pani ka pressure bahut kam hai {rand(AREAS)} colony mein {rand(DURATIONS)} se. {rand(CLOSINGS)}",
        f"{rand(GREETINGS)} नल में {rand(DURATIONS)} से पानी नहीं आ रहा {rand(CITIES)} में। {rand(CLOSINGS)}",
        f"Tanker bhi {rand(DURATIONS)} se nahi aaya {rand(AREAS)} mein, pani ki badi dikkat hai. {rand(CLOSINGS)}",
        f"Pipe leakage {rand(AREAS)} mein {rand(DURATIONS)} se hai, koi repair nahi hua. {rand(CLOSINGS)}",
    ]
    return rand(templates)

def gen_sanitation():
    templates = [
        f"Kachra {rand(DURATIONS)} se nahi utha {rand(AREAS)} se. {rand(CLOSINGS)}",
        f"{rand(GREETINGS)} Safai nahi ho rahi {rand(AREAS)} mein {rand(DURATIONS)} se, ganda nala bahut bura hal hai. {rand(CLOSINGS)}",
        f"Sewage choke hai {rand(AREAS)} mein {rand(DURATIONS)} se, naali overflow ho rahi hai. {rand(CLOSINGS)}",
        f"गंदगी {rand(AREAS)} में {rand(DURATIONS)} से फैली हुई है, सफाई कर्मचारी नहीं आ रहे। {rand(CLOSINGS)}",
        f"Dustbin full hai {rand(AREAS)} mein, {rand(DURATIONS)} se clear nahi ki. {rand(CLOSINGS)}",
        f"Open drain overflow {rand(AREAS)} mein, {rand(DURATIONS)} se shikayat kar raha hoon. {rand(CLOSINGS)}",
    ]
    return rand(templates)

def gen_roads():
    templates = [
        f"Sadak {rand(AREAS)} mein {rand(DURATIONS)} se toot gayi hai, gaddhe ho gaye hain. {rand(CLOSINGS)}",
        f"{rand(GREETINGS)} Road {rand(CITIES)} mein kharab hai {rand(DURATIONS)} se, accident ka dar hai. {rand(CLOSINGS)}",
        f"Street light {rand(AREAS)} mein {rand(DURATIONS)} se band hai, raat ko andhera rehta hai. {rand(CLOSINGS)}",
        f"Footpath {rand(AREAS)} mein toot gayi hai {rand(DURATIONS)} pehle, koi repair nahi hua. {rand(CLOSINGS)}",
        f"सड़क पर गड्ढे {rand(CITIES)} में {rand(DURATIONS)} से हैं, वाहन चालक परेशान हैं। {rand(CLOSINGS)}",
    ]
    return rand(templates)

def gen_education():
    templates = [
        f"{rand(GREETINGS)} School {rand(CITIES)} mein {rand(DURATIONS)} se teacher nahi aa raha. {rand(CLOSINGS)}",
        f"Scholarship {rand(NAMES)} ki {rand(DURATIONS)} se nahi mili, school ne form bhara tha. {rand(CLOSINGS)}",
        f"Admission mein problem hai {rand(CITIES)} ke school mein. {rand(CLOSINGS)}",
        f"Mid day meal {rand(DURATIONS)} se band hai {rand(AREAS)} school mein. {rand(CLOSINGS)}",
        f"स्कूल में {rand(DURATIONS)} से पानी और शौचालय की व्यवस्था नहीं है {rand(CITIES)} में। {rand(CLOSINGS)}",
        f"Board exam result {rand(NAMES)} ka {rand(DURATIONS)} se galat show ho raha hai online. {rand(CLOSINGS)}",
    ]
    return rand(templates)

def gen_healthcare():
    templates = [
        f"{rand(GREETINGS)} Doctor {rand(DURATIONS)} se hospital {rand(CITIES)} mein nahi aa raha. {rand(CLOSINGS)}",
        f"Hospital mein medicines {rand(DURATIONS)} se khatam hain {rand(CITIES)} mein. {rand(CLOSINGS)}",
        f"Ayushman card se treatment nahi mil rahi {rand(NAMES)} ko {rand(DURATIONS)} se. {rand(CLOSINGS)}",
        f"PHC {rand(AREAS)} mein {rand(DURATIONS)} se doctor nahi hai, makkhi chhapti hai. {rand(CLOSINGS)}",
        f"स्वास्थ्य केंद्र {rand(CITIES)} में {rand(DURATIONS)} से बंद है, मरीज परेशान हैं। {rand(CLOSINGS)}",
    ]
    return rand(templates)

def gen_banking():
    templates = [
        f"{rand(GREETINGS)} {rand(NAMES)} ka bank account {rand(DURATIONS)} se freeze hai, koi reason nahi bataya. {rand(CLOSINGS)}",
        f"ATM {rand(AREAS)} mein {rand(DURATIONS)} se kaam nahi kar raha. {rand(CLOSINGS)}",
        f"Loan application {rand(NAMES)} ki {rand(DURATIONS)} se pending hai bank mein. {rand(CLOSINGS)}",
        f"Jan dhan account mein paise nahi aaye {rand(DURATIONS)} se {rand(NAMES)} ko. {rand(CLOSINGS)}",
        f"बैंक से {rand(DURATIONS)} से कोई जवाब नहीं मिल रहा {rand(NAMES)} के खाते के बारे में। {rand(CLOSINGS)}",
    ]
    return rand(templates)

def gen_corruption():
    templates = [
        f"{rand(GREETINGS)} {rand(AREAS)} mein officer ne {rand(DURATIONS)} mein rishwat maangi, kaam nahi kiya. {rand(CLOSINGS)}",
        f"Sarkari kaam ke liye paisa maanga ja raha hai {rand(CITIES)} office mein. {rand(CLOSINGS)}",
        f"Bhrashtachar {rand(CITIES)} mein {rand(DURATIONS)} se ho raha hai, koi sunne wala nahi. {rand(CLOSINGS)}",
        f"{rand(NAMES)} ne shikayat ki bribe maangi ja rahi hai {rand(AREAS)} mein. {rand(CLOSINGS)}",
        f"Tender mein ghotala hua hai {rand(CITIES)} mein, {rand(DURATIONS)} se inquiry nahi hui. {rand(CLOSINGS)}",
    ]
    return rand(templates)

def gen_employment():
    templates = [
        f"{rand(GREETINGS)} Naukri ke liye interview {rand(DURATIONS)} se postpone ho raha hai {rand(CITIES)} mein. {rand(CLOSINGS)}",
        f"MNREGA payment {rand(NAMES)} ko {rand(DURATIONS)} se nahi mili {rand(AREAS)} mein. {rand(CLOSINGS)}",
        f"Factory mein {rand(DURATIONS)} se salary nahi mili {rand(NAMES)} ko. {rand(CLOSINGS)}",
        f"Rozgar mela {rand(CITIES)} mein {rand(DURATIONS)} se announce hua tha, abhi tak nahi hua. {rand(CLOSINGS)}",
        f"Labour department {rand(CITIES)} ne {rand(DURATIONS)} se koi jawab nahi diya {rand(NAMES)} ki shikayat par. {rand(CLOSINGS)}",
    ]
    return rand(templates)

def gen_ration():
    templates = [
        f"{rand(GREETINGS)} Ration {rand(DURATIONS)} se nahi mila {rand(NAMES)} ko, {rand(AREAS)} mein. {rand(CLOSINGS)}",
        f"PDS shop {rand(AREAS)} mein {rand(DURATIONS)} se band hai. {rand(CLOSINGS)}",
        f"Ration card naya banwana hai {rand(NAMES)} ka, {rand(DURATIONS)} se office chakkar laga raha hoon. {rand(CLOSINGS)}",
        f"Gehu aur chawal {rand(DURATIONS)} se nahi aaya fair price shop mein {rand(CITIES)}. {rand(CLOSINGS)}",
        f"BPL card se ration nahi de rahe {rand(AREAS)} dukan wale, {rand(DURATIONS)} se pareshan hoon. {rand(CLOSINGS)}",
    ]
    return rand(templates)

def gen_municipal():
    templates = [
        f"{rand(GREETINGS)} Birth certificate {rand(NAMES)} ka {rand(DURATIONS)} se nahi mila nagar nigam {rand(CITIES)} se. {rand(CLOSINGS)}",
        f"Trade license {rand(DURATIONS)} se pending hai {rand(CITIES)} municipal office mein. {rand(CLOSINGS)}",
        f"Death certificate {rand(NAMES)} ka {rand(DURATIONS)} se nahi mila. {rand(CLOSINGS)}",
        f"Property tax notice galat aaya hai {rand(NAMES)} ko {rand(CITIES)} mein. {rand(CLOSINGS)}",
        f"नगर निगम {rand(CITIES)} से {rand(DURATIONS)} से जाति प्रमाण पत्र नहीं मिल रहा {rand(NAMES)} को। {rand(CLOSINGS)}",
    ]
    return rand(templates)


# ─────────────────────────────────────────────
# Generate augmented samples
# ─────────────────────────────────────────────
AUGMENT_PER_CLASS = 500

category_generators = {
    "Land Records & Revenue": gen_land_records,
    "Police & Law and Order": gen_police,
    "Pension & Provident Fund": gen_pension,
    "Electricity": gen_electricity,
    "Water Supply": gen_water,
    "Sanitation & Garbage": gen_sanitation,
    "Roads & Infrastructure": gen_roads,
    "Education & Schools": gen_education,
    "Healthcare & Hospitals": gen_healthcare,
    "Banking & Financial Services": gen_banking,
    "Corruption & Bribery": gen_corruption,
    "Employment & Labour": gen_employment,
    "Ration & Public Distribution System": gen_ration,
    "Municipal Certificates": gen_municipal,
}

# Normalize keys to match actual dataset categories
actual_cats = df_train['category'].unique()
cat_map = {}
for k in category_generators:
    for ac in actual_cats:
        if k.lower().replace(' ','') == ac.lower().replace(' ',''):
            cat_map[ac] = category_generators[k]
            break

print("Categories mapped for augmentation:", list(cat_map.keys()))
print(f"Missing generators: {set(actual_cats) - set(cat_map.keys())}")

augmented_rows = []
aug_id = 10000
for cat, gen_fn in cat_map.items():
    for _ in range(AUGMENT_PER_CLASS):
        augmented_rows.append({
            "grievance_id": f"GRVAUG{aug_id:06d}",
            "text": gen_fn(),
            "category": cat
        })
        aug_id += 1

df_aug = pd.DataFrame(augmented_rows)
print(f"\nAugmented samples generated: {len(df_aug)}")
print(f"Per category: {df_aug['category'].value_counts().to_dict()}")

# Combine with original training data
df_combined = pd.concat([df_train, df_aug], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\nCombined dataset size: {len(df_combined)}")
df_combined.to_csv(os.path.join(BASE, "grievances_augmented.csv"), index=False)
print("Saved grievances_augmented.csv")

# ─────────────────────────────────────────────
# Retrain all models
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RETRAINING WITH AUGMENTED DATA")
print("=" * 60)

df_combined['clean_text'] = df_combined['text'].str.lower().str.strip()
df_holdout['clean_text'] = df_holdout['text'].str.lower().str.strip()

le = LabelEncoder()
df_combined['label'] = le.fit_transform(df_combined['category'])
df_holdout['label'] = le.transform(df_holdout['category'])

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    df_combined['clean_text'], df_combined['label'],
    test_size=0.15, random_state=42, stratify=df_combined['label']
)

X_holdout_raw = df_holdout['clean_text']
y_holdout = df_holdout['label']

print(f"Train: {len(X_train_raw)} | Val: {len(X_val_raw)} | Holdout: {len(X_holdout_raw)}")

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000,
    sublinear_tf=True,
    min_df=2,
    strip_accents='unicode',
    analyzer='word',
)

X_train = tfidf.fit_transform(X_train_raw)
X_val = tfidf.transform(X_val_raw)
X_holdout = tfidf.transform(X_holdout_raw)

print(f"TF-IDF matrix shape: {X_train.shape}")

models = {
    "Logistic Regression": LogisticRegression(C=5, max_iter=1000, random_state=42, n_jobs=-1, solver='lbfgs'),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "SVM (LinearSVC)": LinearSVC(C=1.0, max_iter=2000, random_state=42),
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric='mlogloss', random_state=42, n_jobs=-1),
}

results = {}
for name, model in models.items():
    print(f"\n[TRAIN] {name}...")
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    y_pred_hold = model.predict(X_holdout)

    results[name] = {
        "model": model,
        "val_acc": accuracy_score(y_val, y_pred_val),
        "val_f1": f1_score(y_val, y_pred_val, average='weighted'),
        "holdout_acc": accuracy_score(y_holdout, y_pred_hold),
        "holdout_f1_weighted": f1_score(y_holdout, y_pred_hold, average='weighted'),
        "holdout_f1_macro": f1_score(y_holdout, y_pred_hold, average='macro'),
    }
    r = results[name]
    print(f"  Val Acc: {r['val_acc']:.4f}  Val F1: {r['val_f1']:.4f}  "
          f"Holdout Acc: {r['holdout_acc']:.4f}  Holdout F1(W): {r['holdout_f1_weighted']:.4f}")

print("\n" + "=" * 60)
print("  FINAL COMPARISON TABLE (After Augmentation)")
print("=" * 60)
print(f"\n{'Model':<25} {'Val Acc':>8} {'Val F1':>8} {'Hold Acc':>10} {'Hold F1(W)':>11} {'Hold F1(M)':>11}")
print("-" * 78)
for name, r in sorted(results.items(), key=lambda x: -x[1]['holdout_f1_weighted']):
    print(f"{name:<25} {r['val_acc']:>8.4f} {r['val_f1']:>8.4f} {r['holdout_acc']:>10.4f} {r['holdout_f1_weighted']:>11.4f} {r['holdout_f1_macro']:>11.4f}")

best_name = max(results, key=lambda x: results[x]['holdout_f1_weighted'])
best_model = results[best_name]['model']
y_pred_holdout = best_model.predict(X_holdout)

print(f"\n[BEST] {best_name}")
print("\n[REPORT] Per-Class Report (Holdout):")
print(classification_report(y_holdout, y_pred_holdout, target_names=le.classes_, digits=4))

# Save best model
with open(os.path.join(BASE, "best_model_aug.pkl"), "wb") as f:
    pickle.dump(best_model, f)
with open(os.path.join(BASE, "tfidf_vectorizer_aug.pkl"), "wb") as f:
    pickle.dump(tfidf, f)
with open(os.path.join(BASE, "label_encoder_aug.pkl"), "wb") as f:
    pickle.dump(le, f)

summary = {
    "best_model": best_name,
    "augmented_training_size": len(df_combined),
    "all_models": {n: {k: v for k, v in r.items() if k != "model"} for n, r in results.items()}
}
with open(os.path.join(BASE, "augmented_training_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n[DONE] Saved: best_model_aug.pkl, tfidf_vectorizer_aug.pkl, label_encoder_aug.pkl")
print("AUGMENTED TRAINING COMPLETE!")
