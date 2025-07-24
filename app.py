import json
import pandas as pd
import numpy as np
import requests
import re
from flask import Flask, request, jsonify
from datetime import datetime, timezone
import dirtyjson

# This line initializes the Flask application.
app = Flask(__name__)

# --- V2 CONFIGURATION ---
OLLAMA_URL = "http://localhost:11435/api/generate"
MODEL_NAME = "mistral:instruct"
NEW_SENTIMENT_LABELS = ["MilestoneAchieved", "PositiveProgress", "RiskIdentified", "Blocker", "ResourceConstraint", "Question", "DecisionNeeded", "TimelineConcern", "BudgetConcern"]
FINAL_FEATURE_SET_V2 = ['percentComplete', 'cost_variance_ratio', 'baseline_cpi', 'baseline_spi', 'schedule_slippage_days', 'days_since_last_update', 'task_count', 'late_task_count', 'open_task_count', 'avg_task_pct_complete', 'issue_count', 'open_issue_count', 'progress_vs_time_ratio', 'change_in_cpi', 'change_in_spi', 'sentiment_trend', 'late_task_ratio', 'issue_to_task_ratio', 'reassignment_count', 'scope_change_count', 'avg_open_issue_age', 'has_blocker', 'has_risk', 'has_timeline_concern', 'has_budget_concern', 'negative_sentiment_label_count']

# --- HELPER FUNCTIONS ---

def analyze_sentiment_v2(text):
    if not text:
        print("INFO: No text provided for sentiment analysis.")
        return {"labels": [], "score": 0.0, "reasoning": None}
    prompt = f'Analyze the project updates below. Return a JSON object with "labels" (a JSON array from {json.dumps(NEW_SENTIMENT_LABELS)}), "sentiment_score" (a float from -1.0 to 1.0), and "reasoning" (a brief explanation for the assigned labels and score). Only return the JSON object. Text: """{text}"""'
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "format": "json"}
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=90)
        res.raise_for_status()
        parsed = json.loads(res.json().get("response", "{}"))
        valid_labels = [lbl for lbl in parsed.get("labels", []) if lbl in NEW_SENTIMENT_LABELS]
        # **INCORPORATED CHANGE 1**
        return {
            "labels": valid_labels, 
            "score": parsed.get("sentiment_score", 0.0), 
            "reasoning": parsed.get("reasoning")
        }
    except Exception:
        return {"labels": ["ProcessingError"], "score": 0.0, "reasoning": None}

def create_feature_row_v2(current_metrics, previous_metrics, sentiment_result):
    """
    Performs V2 feature engineering with corrected timezone handling.
    """
    record = {}
    cm, pm = current_metrics, previous_metrics or {}
    
    # --- Basic & Trend Features ---
    record['percentComplete'] = cm.get('percentComplete', 0)
    current_cpi = cm.get('cpi') if isinstance(cm.get('cpi'), (int, float)) else 1.0
    previous_cpi = pm.get('cpi') if isinstance(pm.get('cpi'), (int, float)) else 1.0
    current_spi = cm.get('spi') if isinstance(cm.get('spi'), (int, float)) else 1.0
    previous_spi = pm.get('spi') if isinstance(pm.get('spi'), (int, float)) else 1.0
    current_sentiment = sentiment_result.get('score', 0.0)
    previous_sentiment = pm.get('sentimentScore') if isinstance(pm.get('sentimentScore'), (int, float)) else 0.0
    record['baseline_cpi'] = current_cpi; record['baseline_spi'] = current_spi
    record['change_in_cpi'] = current_cpi - previous_cpi; record['change_in_spi'] = current_spi - previous_spi
    record['sentiment_trend'] = current_sentiment - previous_sentiment
    record['days_since_last_update'] = (datetime.now(timezone.utc) - pd.to_datetime(cm.get('lastUpdateDate'), utc=True)).days if cm.get('lastUpdateDate') else 365

    # --- Task & Issue Aggregates ---
    all_tasks = [task for tg in cm.get('taskInfo', []) for task in tg.get('body', {}).get('data', [])]
    all_issues = [issue for ig in cm.get('issueInfo', []) for issue in ig.get('body', {}).get('data', [])]
    record['task_count'] = len(all_tasks); record['issue_count'] = len(all_issues)
    record['open_task_count'] = sum(1 for t in all_tasks if t.get('status') != 'CPL')
    record['open_issue_count'] = sum(1 for i in all_issues if i.get('status') != 'CPL')
    record['late_task_count'] = sum(1 for t in all_tasks if t.get('progressStatus') in ['BH', 'LT'])
    record['avg_task_pct_complete'] = np.mean([t.get('percentComplete', 0) for t in all_tasks]) if all_tasks else 0

    # --- 'Hidden' Signal Mining ---
    updates = cm.get('updates', [])
    record['reassignment_count'] = sum(1 for u in updates if u.get('updateType') == 'assignmentReassign')
    record['scope_change_count'] = sum(1 for u in updates if u.get('updateType') in ['taskAdd', 'taskRemove'])
    open_issues_with_date = [i for i in all_issues if i.get('status') != 'CPL' and i.get('entryDate')]
    if open_issues_with_date:
        now_utc = datetime.now(timezone.utc)
        issue_ages = [(now_utc - pd.to_datetime(i['entryDate'], utc=True)).days for i in open_issues_with_date]
        record['avg_open_issue_age'] = np.mean(issue_ages)
    else:
        record['avg_open_issue_age'] = 0

    # --- Ratios & Calculated Features ---
    # **INCORPORATED CHANGE 2**
    record['cost_variance_ratio'] = (cm.get('actualCost') or 0) / ((cm.get('plannedCost') or 0) + 1e-6)
    record['late_task_ratio'] = record['late_task_count'] / (record['task_count'] + 1e-6)
    record['issue_to_task_ratio'] = record['issue_count'] / (record['task_count'] + 1e-6)

    planned_start_date = pd.to_datetime(cm.get('plannedStartDate'), utc=True) if cm.get('plannedStartDate') else None
    planned_completion_date = pd.to_datetime(cm.get('plannedCompletionDate'), utc=True) if cm.get('plannedCompletionDate') else None

    if planned_start_date and planned_completion_date:
        planned_duration = (planned_completion_date - planned_start_date).days
    else:
        planned_duration = 0

    if planned_start_date:
        time_elapsed = (datetime.now(timezone.utc) - planned_start_date).days
    else:
        time_elapsed = 0

    pct_time_elapsed = (time_elapsed / (planned_duration + 1e-6)) if planned_duration > 0 else 0
    record['progress_vs_time_ratio'] = (cm.get('percentComplete') or 0) / (pct_time_elapsed * 100 + 1e-6)

    projected_completion_date = pd.to_datetime(cm.get('projectedCompletionDate'), utc=True) if cm.get('projectedCompletionDate') else None
    if projected_completion_date and planned_completion_date:
        record['schedule_slippage_days'] = (projected_completion_date - planned_completion_date).days
    else:
        record['schedule_slippage_days'] = 0
    
    # --- Multi-Label Sentiment Features ---
    labels = sentiment_result.get("labels", [])
    record['has_blocker'] = 1 if 'Blocker' in labels else 0; record['has_risk'] = 1 if 'RiskIdentified' in labels else 0
    record['has_timeline_concern'] = 1 if 'TimelineConcern' in labels else 0; record['has_budget_concern'] = 1 if 'BudgetConcern' in labels else 0
    negative_labels = {'Blocker', 'RiskIdentified', 'ResourceConstraint', 'TimelineConcern', 'BudgetConcern'}
    record['negative_sentiment_label_count'] = len([lbl for lbl in labels if lbl in negative_labels])
    
    # --- Final Assembly ---
    df = pd.DataFrame([record]);
    for col in FINAL_FEATURE_SET_V2:
        if col not in df.columns: df[col] = 0
    return df[FINAL_FEATURE_SET_V2]

def parse_nested_json_string(data):
    """
    Parses the double-stringified JSON array from Fusion,
    with a specific fix for the "],[ " separator issue.
    """
    if not (isinstance(data, list) and data):
        return []
    
    nested_json_string = data[0]

    if not isinstance(nested_json_string, str):
        return json.loads(json.dumps(nested_json_string)) if isinstance(nested_json_string, list) else []

    # --- THE CRITICAL FIX IS HERE ---
    # Manually fix the separator between project objects that Fusion creates.
    # This turns "[{...}],[{...}]" into "[{...},{...}]", which is valid JSON.
    print("INFO: Attempting to fix array separator issue...")
    cleaned_string = nested_json_string.replace('],[', ',')
    # --- END OF FIX ---

    try:
        # The standard parser should now work on the cleaned string.
        return json.loads(cleaned_string)
    except json.JSONDecodeError as e:
        print(f"FATAL: Could not parse the string even after cleaning. Error: {e}")
        print(f"String started with: {cleaned_string[:200]}...")
        return []

# --- FLASK APPLICATION ENDPOINTS ---

@app.route('/', methods=['GET'])
def home():
    return "✅ Flask server is running.", 200

# In app.py

@app.route('/process_projects', methods=['POST'])
def process_projects_endpoint():
    print("Received request to /process_projects")
    payload = request.get_json(silent=True)
    if not payload or not isinstance(payload, dict):
        return jsonify({"error": "Request body is not a valid JSON dictionary."}), 400

    current_projects = parse_nested_json_string(payload.get('current_data'))
    previous_projects = parse_nested_json_string(payload.get('previous_data'))

    if not current_projects:
         return jsonify({"error": "Failed to parse 'current_data' into a list of projects."}), 400

    print(f"Successfully parsed. Processing {len(current_projects)} current and {len(previous_projects)} previous projects.")

    # --- MODIFICATION START ---
    # Step 1: Create a map from the PREVIOUS data. This will become our new history file.
    # This preserves projects from the last run that might not be in the current run.
    next_run_data_map = {proj['projectID']: proj for proj in previous_projects if isinstance(proj, dict) and 'projectID' in proj}
    
    all_features_list = []
    all_passthrough_data = []

    # Step 2: Loop through CURRENT projects
    for project_data in current_projects:
        if not isinstance(project_data, dict):
            continue
        
        project_id = project_data.get('projectID')
        if not project_id:
            continue

        print(f"Processing project {project_id}...")
        # Get the previous metrics for *this specific project* from the map.
        # This will be `None` if it's a new project, which is the desired behavior.
        previous_metrics = next_run_data_map.get(project_id)

        # --- Text analysis and feature engineering (no changes here) ---
        text_for_sentiment = ""
        if 'text_data' in project_data and isinstance(project_data.get('text_data'), list):
            for item in project_data['text_data']:
                project_notes = item.get('project-notes')
                if isinstance(project_notes, list):
                    text_for_sentiment += "\n".join(project_notes) + "\n"
        sentiment_result = analyze_sentiment_v2(text_for_sentiment.strip())
        feature_row_df = create_feature_row_v2(project_data, previous_metrics, sentiment_result)
        feature_row_df.insert(0, 'projectID', project_id)
        all_features_list.append(feature_row_df)
        
        # --- Assemble passthrough and data for next run ---
        all_passthrough_data.append({
            "projectID": project_id,
            "projectName": project_data.get('name'),
            "ownerID": project_data.get('ownerID'),
            "sentiment_score": sentiment_result.get('score'),
            "sentiment_labels": sentiment_result.get('labels'),
            "sentiment_reasoning": sentiment_result.get('reasoning'),
            "curr_score": project_data.get('parameterValues', {}).get('DE:SentimentScore')
        })

        data_to_save = project_data.copy()
        data_to_save['sentimentScore'] = sentiment_result.get('score', 0.0)
        for key in ['text_data', 'updates', 'taskInfo', 'issueInfo', 'hourInfo', 'docInfo', 'userInfo', 'riskInfo', 'baselineInfo', 'expenseInfo']:
            data_to_save.pop(key, None)
        
        # Step 3: UPDATE or ADD the project's new state to our map.
        # This is the core of the new logic.
        next_run_data_map[project_id] = data_to_save

    if not all_features_list:
        return jsonify({"error": "No projects were processed into features."}), 500

    # Step 4: Convert the final, updated map back to a list for the response.
    final_data_for_next_run = list(next_run_data_map.values())
    
    engineered_df = pd.concat(all_features_list, ignore_index=True).fillna(0)
    engineered_features_json = json.loads(engineered_df.to_json(orient='records'))

    response_data = {
        "engineered_features": engineered_features_json,
        "data_for_next_run": final_data_for_next_run, # Use the complete, updated list
        "passthrough_data": all_passthrough_data
    }
    # --- MODIFICATION END ---
    
    print("Processing complete. Sending response.")
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)