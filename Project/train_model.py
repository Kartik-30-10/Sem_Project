import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("SEM_PROJECT.csv")

bool_cols = [
    'prior_substance_use_disorder_flag', 'overlapping_opioids_flag',
    'doctor_shopping_flag', 'concurrent_benzodiazepine_use',
    'depression_diagnosis', 'chronic_pain_diagnosis',
    'addiction_treatment_history', 'mme_escalation_flag'
]
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

feature_cols = [
    'age', 'max_daily_mme', 'opioid_use_duration_days', 'early_refill_count',
    'overlapping_opioids_flag', 'doctor_shopping_flag', 'concurrent_benzodiazepine_use',
    'depression_diagnosis', 'chronic_pain_diagnosis', 'number_of_prescribing_doctors',
    'er_visits_last_year', 'addiction_treatment_history', 'prior_substance_use_disorder_flag',
    'mme_escalation_flag', 'medication_adherence_score'
]

def calculate_risk_score(row):
    score = 0
    if row['age'] >= 60:
        score += 1
    elif 35 <= row['age'] < 60:
        score += 2
    if row['max_daily_mme'] > 90:
        score += 3
    elif row['max_daily_mme'] >= 50:
        score += 2
    elif row['max_daily_mme'] > 0:
        score += 1
    if row['opioid_use_duration_days'] > 90:
        score += 3
    elif row['opioid_use_duration_days'] > 30:
        score += 2
    if row['early_refill_count'] > 0:
        score += 2
    if row['doctor_shopping_flag']:
        score += 3
    if row['overlapping_opioids_flag']:
        score += 2
    if row['concurrent_benzodiazepine_use']:
        score += 3
    if row['depression_diagnosis']:
        score += 2
    if row['chronic_pain_diagnosis']:
        score += 1
    if row['number_of_prescribing_doctors'] >= 3:
        score += 2
    if row['er_visits_last_year'] >= 3:
        score += 2
    elif row['er_visits_last_year'] >= 1:
        score += 1
    if row['addiction_treatment_history']:
        score += 3
    if row['prior_substance_use_disorder_flag']:
        score += 4
    if row['mme_escalation_flag']:
        score += 2
    if row['medication_adherence_score'] < 70:
        score += 2
    return score

print("Calculating risk scores...")
df['risk_score'] = df.apply(calculate_risk_score, axis=1)
max_score = 35
df['risk_probability'] = df['risk_score'] / max_score
risk_thresholds = [0.33, 0.66]  # low <33%, high 33-66%, very high >66%

def get_risk_level(prob):
    if prob < risk_thresholds[0]:
        return "Low"
    elif prob < risk_thresholds[1]:
        return "High"
    else:
        return "Very High"

df['risk_level'] = df['risk_probability'].apply(get_risk_level)
df['synthetic_disorder'] = (df['risk_score'] >= 12).astype(int)
target_column = 'synthetic_disorder'

X = df[feature_cols].copy()
y = df[target_column]

print(f"Class distribution:\n{y.value_counts()}")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=94, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gb_model = GradientBoostingClassifier(random_state=94)
gb_model.fit(X_train_scaled, y_train)

preds = gb_model.predict(X_test_scaled)
acc = accuracy_score(y_test, preds)
print(f"Model Accuracy: {acc:.4f}")
print(classification_report(y_test, preds))

joblib.dump(gb_model, "rfe_gb_opioid_model.joblib")
joblib.dump(scaler, "scaler.joblib")
print("Model and scaler saved.")
