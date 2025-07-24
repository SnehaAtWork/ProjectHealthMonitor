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
# The feature set is updated with new and refined features.
FINAL_FEATURE_SET_V2 = [
    'percentComplete', 'cost_variance_ratio', 'baseline_cpi', 'baseline_spi', 
    'schedule_slippage_days', 'days_since_last_update', 'task_count', 
    'late_task_count', 'open_task_count', 'avg_task_pct_complete', 'issue_count', 
    'open_issue_count', 'progress_vs_time_ratio', 'change_in_cpi', 'change_in_spi', 
    'sentiment_trend', 'late_task_ratio', 'issue_to_task_ratio', 'reassignment_count', 
    'scope_change_count', 'avg_open_issue_age', 'has_blocker', 'has_risk', 
    'has_timeline_concern', 'has_budget_concern', 
    # NEW and REFINED features below
    'tcpi', 'task_lateness_variance', 'stagnant_task_count', 'weighted_sentiment_risk'
]


# --- HELPER FUNCTIONS ---

def analyze_sentiment_v2(text):
    """
    Analyzes sentiment with more specific error handling.
    """
    if not text:
        print("INFO: No text provided for sentiment analysis.")
        return {"labels": [], "score": 0.0, "reasoning": None}
    prompt = f'Analyze the project updates below. Return a JSON object with "labels" (a JSON array from {json.dumps(NEW_SENTIMENT_LABELS)}), "sentiment_score" (a float from -1.0 to 1.0), and "reasoning" (a brief explanation for the assigned labels and score). Only return the JSON object. Text: """{text}"""'
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "format": "json"}
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=90)
        res.raise_for_status()
        # The LLM's response is parsed from a JSON string within a JSON object
        parsed = json.loads(res.json().get("response", "{}"))
        valid_labels = [lbl for lbl in parsed.get("labels", []) if lbl in NEW_SENTIMENT_LABELS]
        return {
            "labels": valid_labels, 
            "score": parsed.get("sentiment_score", 0.0), 
            "reasoning": parsed.get("reasoning")
        }
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Network or request error calling LLM: {e}")
        return {"labels": ["NetworkError"], "score": 0.0, "reasoning": "Could not connect to the sentiment analysis model."}
    except json.JSONDecodeError:
        print(f"ERROR: Failed to parse JSON response from LLM.")
        return {"labels": ["ParsingError"], "score": 0.0, "reasoning": "Received a malformed response from the sentiment model."}
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during sentiment analysis: {e}")
        return {"labels": ["ProcessingError"], "score": 0.0, "reasoning": "An unknown error occurred."}

# --- REFACTORED FEATURE ENGINEERING HELPERS ---

def calculate_financial_metrics(record, cm, pm):
    """Calculates all financial and EVM metrics for the project."""
    record['percentComplete'] = cm.get('percentComplete', 0)
    record['cost_variance_ratio'] = (cm.get('actualCost') or 0) / ((cm.get('plannedCost') or 0) + 1e-6)
    
    current_cpi = cm.get('cpi') if isinstance(cm.get('cpi'), (int, float)) else 1.0
    previous_cpi = pm.get('cpi') if isinstance(pm.get('cpi'), (int, float)) else 1.0
    current_spi = cm.get('spi') if isinstance(cm.get('spi'), (int, float)) else 1.0
    previous_spi = pm.get('spi') if isinstance(pm.get('spi'), (int, float)) else 1.0

    record['baseline_cpi'] = current_cpi
    record['baseline_spi'] = current_spi
    record['change_in_cpi'] = current_cpi - previous_cpi
    record['change_in_spi'] = current_spi - previous_spi
    
    # NEW: Calculate To-Complete Performance Index (TCPI)
    budget_at_completion = cm.get('plannedCost', 0)
    actual_cost = cm.get('actualCost', 0)
    earned_value = (cm.get('percentComplete', 0) / 100) * budget_at_completion
    work_remaining = budget_at_completion - earned_value
    funds_remaining = budget_at_completion - actual_cost
    record['tcpi'] = work_remaining / (funds_remaining + 1e-6) if funds_remaining > 0 else 100.0

    return record

def calculate_schedule_metrics(record, cm):
    """Calculates all schedule, progress, and time-based metrics."""
    record['days_since_last_update'] = (datetime.now(timezone.utc) - pd.to_datetime(cm.get('lastUpdateDate'), utc=True)).days if cm.get('lastUpdateDate') else 365
    
    planned_start_date = pd.to_datetime(cm.get('plannedStartDate'), utc=True) if cm.get('plannedStartDate') else None
    planned_completion_date = pd.to_datetime(cm.get('plannedCompletionDate'), utc=True) if cm.get('plannedCompletionDate') else None
    
    planned_duration = (planned_completion_date - planned_start_date).days if planned_start_date and planned_completion_date else 0
    time_elapsed = (datetime.now(timezone.utc) - planned_start_date).days if planned_start_date else 0
    pct_time_elapsed = (time_elapsed / (planned_duration + 1e-6)) if planned_duration > 0 else 0
    record['progress_vs_time_ratio'] = (cm.get('percentComplete') or 0) / (pct_time_elapsed * 100 + 1e-6)

    projected_completion_date = pd.to_datetime(cm.get('projectedCompletionDate'), utc=True) if cm.get('projectedCompletionDate') else None
    record['schedule_slippage_days'] = (projected_completion_date - planned_completion_date).days if projected_completion_date and planned_completion_date else 0
    
    return record

def calculate_task_issue_metrics(record, all_tasks, all_issues):
    """Calculates all metrics derived from project tasks and issues."""
    record['task_count'] = len(all_tasks)
    record['issue_count'] = len(all_issues)
    record['open_task_count'] = sum(1 for t in all_tasks if t.get('status') != 'CPL')
    record['open_issue_count'] = sum(1 for i in all_issues if i.get('status') != 'CPL')
    record['late_task_count'] = sum(1 for t in all_tasks if t.get('progressStatus') in ['BH', 'LT'])
    record['avg_task_pct_complete'] = np.mean([t.get('percentComplete', 0) for t in all_tasks]) if all_tasks else 0
    record['late_task_ratio'] = record['late_task_count'] / (record['task_count'] + 1e-6)
    record['issue_to_task_ratio'] = record['issue_count'] / (record['task_count'] + 1e-6)

    # NEW: Calculate variance in task lateness
    late_tasks = [t for t in all_tasks if t.get('progressStatus') in ['BH', 'LT']]
    if late_tasks:
        lateness_days = []
        for task in late_tasks:
            planned_date = pd.to_datetime(task.get('plannedCompletionDate'), utc=True)
            actual_date = pd.to_datetime(task.get('actualCompletionDate'), utc=True)
            if pd.isna(actual_date): actual_date = datetime.now(timezone.utc)
            if planned_date and actual_date: lateness_days.append((actual_date - planned_date).days)
        record['task_lateness_variance'] = np.var(lateness_days) if lateness_days else 0
    else:
        record['task_lateness_variance'] = 0

    # NEW: Count stagnant open tasks
    stagnant_tasks = 0
    for task in all_tasks:
        if task.get('status') != 'CPL':
            last_update = pd.to_datetime(task.get('lastUpdateDate'), utc=True)
            if last_update and (datetime.now(timezone.utc) - last_update).days > 21:
                stagnant_tasks += 1
    record['stagnant_task_count'] = stagnant_tasks
    
    open_issues_with_date = [i for i in all_issues if i.get('status') != 'CPL' and i.get('entryDate')]
    if open_issues_with_date:
        now_utc = datetime.now(timezone.utc)
        issue_ages = [(now_utc - pd.to_datetime(i['entryDate'], utc=True)).days for i in open_issues_with_date]
        record['avg_open_issue_age'] = np.mean(issue_ages)
    else:
        record['avg_open_issue_age'] = 0
        
    return record

def calculate_activity_metrics(record, updates):
    """Calculates metrics from the project update stream (hidden signals)."""
    record['reassignment_count'] = sum(1 for u in updates if u.get('updateType') == 'assignmentReassign')
    record['scope_change_count'] = sum(1 for u in updates if u.get('updateType') in ['taskAdd', 'taskRemove'])
    return record

def calculate_sentiment_features(record, sentiment_result, pm):
    """Calculates trend and risk features from sentiment analysis results."""
    current_sentiment = sentiment_result.get('score', 0.0)
    previous_sentiment = pm.get('sentimentScore') if isinstance(pm.get('sentimentScore'), (int, float)) else 0.0
    record['sentiment_trend'] = current_sentiment - previous_sentiment
    
    labels = sentiment_result.get("labels", [])
    record['has_blocker'] = 1 if 'Blocker' in labels else 0
    record['has_risk'] = 1 if 'RiskIdentified' in labels else 0
    record['has_timeline_concern'] = 1 if 'TimelineConcern' in labels else 0
    record['has_budget_concern'] = 1 if 'BudgetConcern' in labels else 0
    
    # NEW: Weighted risk score based on sentiment label severity
    sentiment_risk_weights = {
        "Blocker": 3.0, "ResourceConstraint": 2.0, "TimelineConcern": 1.5,
        "BudgetConcern": 1.5, "RiskIdentified": 1.0, "DecisionNeeded": 0.5, "Question": 0.2
    }
    record['weighted_sentiment_risk'] = sum(sentiment_risk_weights.get(lbl, 0) for lbl in labels)
    
    return record
    
def create_feature_row_v2(current_metrics, previous_metrics, sentiment_result):
    """
    Performs V2 feature engineering using a modular, refactored approach.
    """
    record = {}
    cm, pm = current_metrics, previous_metrics or {}
    
    # Extract nested data collections
    all_tasks = [task for tg in cm.get('taskInfo', []) for task in tg.get('body', {}).get('data', [])]
    all_issues = [issue for ig in cm.get('issueInfo', []) for issue in ig.get('body', {}).get('data', [])]
    updates = cm.get('updates', [])

    # Call helper functions to build the feature record
    record = calculate_financial_metrics(record, cm, pm)
    record = calculate_schedule_metrics(record, cm)
    record = calculate_task_issue_metrics(record, all_tasks, all_issues)
    record = calculate_activity_metrics(record, updates)
    record = calculate_sentiment_features(record, sentiment_result, pm)
    
    # --- Final Assembly ---
    df = pd.DataFrame([record])
    for col in FINAL_FEATURE_SET_V2:
        if col not in df.columns: df[col] = 0
    return df[FINAL_FEATURE_SET_V2]

def parse_nested_json_string(data):
    """
    Parses the double-stringified JSON array from Fusion,
    with a specific fix for the "],[ " separator issue.
    """
    if not (isinstance(data, list) and data): return []
    nested_json_string = data[0]
    if not isinstance(nested_json_string, str):
        return json.loads(json.dumps(nested_json_string)) if isinstance(nested_json_string, list) else []

    print("INFO: Attempting to fix array separator issue...")
    cleaned_string = nested_json_string.replace('],[', ',')

    try:
        return json.loads(cleaned_string)
    except json.JSONDecodeError as e:
        print(f"FATAL: Could not parse the string even after cleaning. Error: {e}")
        print(f"String started with: {cleaned_string[:200]}...")
        return []

# --- FLASK APPLICATION ENDPOINTS ---

@app.route('/', methods=['GET'])
def home():
    return "✅ Flask server is running.", 200

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

    next_run_data_map = {proj['projectID']: proj for proj in previous_projects if isinstance(proj, dict) and 'projectID' in proj}
    all_features_list = []
    all_passthrough_data = []

    for project_data in current_projects:
        if not (isinstance(project_data, dict) and project_data.get('projectID')):
            continue
        project_id = project_data.get('projectID')
        print(f"Processing project {project_id}...")
        
        previous_metrics = next_run_data_map.get(project_id)

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
        keys_to_remove = ['text_data', 'updates', 'taskInfo', 'issueInfo', 'hourInfo', 'docInfo', 'userInfo', 'riskInfo', 'baselineInfo', 'expenseInfo']
        for key in keys_to_remove:
            data_to_save.pop(key, None)
        
        next_run_data_map[project_id] = data_to_save

    if not all_features_list:
        return jsonify({"error": "No projects were processed into features."}), 500

    final_data_for_next_run = list(next_run_data_map.values())
    engineered_df = pd.concat(all_features_list, ignore_index=True).fillna(0)
    engineered_features_json = json.loads(engineered_df.to_json(orient='records'))

    response_data = {
        "engineered_features": engineered_features_json,
        "data_for_next_run": final_data_for_next_run,
        "passthrough_data": all_passthrough_data
    }
    
    print("Processing complete. Sending response.")
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)