import pandas as pd
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client['SemProject']
patients_collection = db['NewData']

df = pd.read_csv("SEM_PROJECT.csv")

bool_cols = [
    'prior_substance_use_disorder_flag', 'overlapping_opioids_flag',
    'doctor_shopping_flag', 'concurrent_benzodiazepine_use',
    'depression_diagnosis', 'chronic_pain_diagnosis',
    'addiction_treatment_history', 'mme_escalation_flag'
]

for col in bool_cols:
    if df[col].dtype not in ['int64', 'int32']:
        df[col] = df[col].astype(int)

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

print("Calculating risk scores and syncing data...")
df['risk_score'] = df.apply(calculate_risk_score, axis=1)
max_score = 35
df['risk_probability'] = df['risk_score'] / max_score
risk_threshold = 12
df['synthetic_disorder'] = (df['risk_score'] >= risk_threshold).astype(int)

for _, row in df.iterrows():
    doc = row.to_dict()
    doc['patient_id'] = str(doc.get('patient_id', _))
    patients_collection.update_one(
        {'patient_id': doc['patient_id']},
        {'$set': doc},
        upsert=True
    )
print("Sync complete.")
