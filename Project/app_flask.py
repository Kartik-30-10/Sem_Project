from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
import joblib
import pandas as pd
import uuid

app = Flask(__name__)

client = MongoClient("mongodb://localhost:27017/")
db = client['SemProject']
patients_collection = db['NewData']

gb_model = joblib.load("rfe_gb_opioid_model.joblib")
scaler = joblib.load("scaler.joblib")

feature_cols = [
    'age', 'max_daily_mme', 'opioid_use_duration_days', 'early_refill_count',
    'overlapping_opioids_flag', 'doctor_shopping_flag', 'concurrent_benzodiazepine_use',
    'depression_diagnosis', 'chronic_pain_diagnosis', 'number_of_prescribing_doctors',
    'er_visits_last_year', 'addiction_treatment_history', 'prior_substance_use_disorder_flag',
    'mme_escalation_flag', 'medication_adherence_score'
]

bool_cols = [
    'prior_substance_use_disorder_flag', 'overlapping_opioids_flag',
    'doctor_shopping_flag', 'concurrent_benzodiazepine_use',
    'depression_diagnosis', 'chronic_pain_diagnosis',
    'addiction_treatment_history', 'mme_escalation_flag'
]

max_score = 35
risk_thresholds = [0.33, 0.66]  # percentage thresholds for risk levels


def calculate_risk_score(input_dict):
    score = 0
    if input_dict['age'] >= 60:
        score += 1
    elif 35 <= input_dict['age'] < 60:
        score += 2
    if input_dict['max_daily_mme'] > 90:
        score += 3
    elif input_dict['max_daily_mme'] >= 50:
        score += 2
    elif input_dict['max_daily_mme'] > 0:
        score += 1
    if input_dict['opioid_use_duration_days'] > 90:
        score += 3
    elif input_dict['opioid_use_duration_days'] > 30:
        score += 2
    if input_dict['early_refill_count'] > 0:
        score += 2
    if input_dict['doctor_shopping_flag']:
        score += 3
    if input_dict['overlapping_opioids_flag']:
        score += 2
    if input_dict['concurrent_benzodiazepine_use']:
        score += 3
    if input_dict['depression_diagnosis']:
        score += 2
    if input_dict['chronic_pain_diagnosis']:
        score += 1
    if input_dict['number_of_prescribing_doctors'] >= 3:
        score += 2
    er_visits = input_dict['er_visits_last_year']
    if er_visits >= 3:
        score += 2
    elif er_visits >= 1:
        score += 1
    if input_dict['addiction_treatment_history']:
        score += 3
    if input_dict['prior_substance_use_disorder_flag']:
        score += 4
    if input_dict['mme_escalation_flag']:
        score += 2
    if input_dict['medication_adherence_score'] < 70:
        score += 2
    return score


def get_risk_level(prob):
    if prob < risk_thresholds[0]:
        return "Low"
    elif prob < risk_thresholds[1]:
        return "High"
    else:
        return "Very High"


def predict_opioid_risk(input_dict):
    input_df = pd.DataFrame([input_dict])[feature_cols]
    input_scaled = scaler.transform(input_df)
    prob = gb_model.predict_proba(input_scaled)[0, 1]
    risk_score = calculate_risk_score(input_dict)
    risk_percent = int(prob * 100)
    risk_level = get_risk_level(prob)
    return risk_level, risk_percent, risk_score


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        pid = request.form.get("patient_id_check")
        if not pid:
            return render_template("index.html", error="Please enter a Patient ID to check.")
        record = patients_collection.find_one({"patient_id": pid})
        if not record:
            return render_template("index.html", error=f"No patient found with ID: {pid}")
        else:
            risk_level = get_risk_level(record['risk_probability'])
            risk_percent = int(record['risk_probability'] * 100)
            return render_template(
                "index.html",
                patient=record,
                risk_level=risk_level,
                risk_percent=risk_percent,
                thresholds=[int(t * 100) for t in risk_thresholds]
            )
    return render_template("index.html", thresholds=[int(t * 100) for t in risk_thresholds])


@app.route("/add_patient", methods=["GET", "POST"])
def add_patient():
    if request.method == "POST":
        try:
            # Auto-generate patient ID here
            patient_id = str(uuid.uuid4())
            input_dict = {}
            for feature in feature_cols:
                val = request.form.get(feature)
                if val is None:
                    return render_template("add_patient.html", error=f"Missing value: {feature}")
                if feature in bool_cols:
                    input_dict[feature] = int(val)
                elif feature in ['medication_adherence_score', 'max_daily_mme', 'opioid_use_duration_days']:
                    input_dict[feature] = float(val)
                else:
                    input_dict[feature] = int(val)

            risk_level, risk_percent, risk_score = predict_opioid_risk(input_dict)

            input_dict.update({
                "patient_id": patient_id,
                "risk_score": risk_score,
                "risk_probability": risk_percent / 100,
                "risk_percent": risk_percent,
                "risk_level": risk_level,
                "synthetic_disorder": int(risk_level != "Low"),
            })

            patients_collection.update_one(
                {"patient_id": patient_id},
                {"$set": input_dict},
                upsert=True
            )
            alert_msg = (
                f"Patient risk level is '{risk_level}'. Their risk is estimated "
                f"at {risk_percent}%. Risk categories: "
                f"Low (<{risk_thresholds[0]*100}%), "
                f"High ({risk_thresholds[0]*100}-{risk_thresholds[1]*100}%), "
                f"Very High (>{risk_thresholds[1]*100}%)."
            )
            return render_template("add_patient.html", success=f"Added successfully! {alert_msg}", patient_id=patient_id,
                                   thresholds=[int(t*100) for t in risk_thresholds])
        except Exception as e:
            return render_template("add_patient.html", error=str(e))
    return render_template("add_patient.html", thresholds=[int(t * 100) for t in risk_thresholds])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not request.is_json:
        return jsonify({"error": "JSON request expected."}), 400
    input_json = request.get_json()
    missing = [f for f in feature_cols if f not in input_json]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    risk_level, risk_percent, risk_score = predict_opioid_risk(input_json)
    return jsonify({
        "risk_percent": risk_percent,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_categories": {
            "Low": f"<{risk_thresholds[0]*100}%",
            "High": f"{risk_thresholds[0]*100}-{risk_thresholds[1]*100}%",
            "Very High": f">{risk_thresholds[1]*100}%"
        },
        "alert": f"Risk is {risk_percent}% and classified as {risk_level}."
    })


if __name__ == "__main__":
    app.run(debug=True)
