import joblib
import pandas as pd
from flask import Flask, request, jsonify

# --- CONFIGURATION ---
MODEL_FILE = 'kmeans_model_v2.joblib'
SCALER_FILE = 'scaler_v2.joblib'
# You MUST update this map based on your Day 3 analysis
CLUSTER_MAP = {
    3: "In Trouble", 
    0: "At Risk", 
    2: "Needs Attention", 
    1: "On Track" 
}
# This MUST match the feature list from your training script
FINAL_FEATURE_SET_V2 = ['percentComplete', 'cost_variance_ratio', 'baseline_cpi', 'baseline_spi', 'schedule_slippage_days', 'days_since_last_update', 'task_count', 'late_task_count', 'open_task_count', 'avg_task_pct_complete', 'issue_count', 'open_issue_count', 'progress_vs_time_ratio', 'change_in_cpi', 'change_in_spi', 'sentiment_trend', 'late_task_ratio', 'issue_to_task_ratio', 'reassignment_count', 'scope_change_count', 'avg_open_issue_age', 'has_blocker', 'has_risk', 'has_timeline_concern', 'has_budget_concern', 'negative_sentiment_label_count']

# --- LOAD MODELS AT STARTUP ---
app = Flask(__name__)
try:
    health_model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("✅ Prediction model and scaler loaded successfully.")
except FileNotFoundError:
    print(f"❌ ERROR: Model files not found. Ensure '{MODEL_FILE}' and '{SCALER_FILE}' are present.")
    health_model = None
    scaler = None

# --- PREDICTION ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict_health_status():
    if not health_model or not scaler:
        return jsonify({"error": "Model or scaler not loaded on server."}), 500

    features_data = request.get_json()
    if not isinstance(features_data, list) or not features_data:
        return jsonify({"error": "Input data must be a non-empty list of project features."}), 400

    # --- THE FIX STARTS HERE ---
    
    # 1. Create a full DataFrame from the incoming data, which includes 'projectID'
    full_df = pd.DataFrame(features_data)

    # 2. Extract the project IDs and keep them in a separate list for later
    project_ids = full_df['projectID'].tolist()

    # 3. Create the numeric-only DataFrame for the model, ensuring column order
    model_df = full_df[FINAL_FEATURE_SET_V2] 

    # 4. Scale the data and predict
    scaled_features = scaler.transform(model_df)
    predictions = health_model.predict(scaled_features)
    health_labels = [CLUSTER_MAP.get(p, "Unknown") for p in predictions]
    
    # 5. Assemble the response by combining the stored project_ids with the new labels
    results = [
        {"projectID": pid, "health_status": label}
        for pid, label in zip(project_ids, health_labels)
    ]
    
    # --- END OF FIX ---
    
    return jsonify(results)

if __name__ == '__main__':
    # Run this on a different port than your processing app
    app.run(debug=True, host='0.0.0.0', port=5002)