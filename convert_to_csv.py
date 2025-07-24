import json
import pandas as pd

# --- THIS IS THE ONLY CHANGE MADE TO THIS SCRIPT ---
FINAL_FEATURE_SET_V2 = [
    'percentComplete', 'cost_variance_ratio', 'baseline_cpi', 'baseline_spi', 
    'schedule_slippage_days', 'days_since_last_update', 'task_count', 
    'late_task_count', 'open_task_count', 'avg_task_pct_complete', 'issue_count', 
    'open_issue_count', 'progress_vs_time_ratio', 'change_in_cpi', 'change_in_spi', 
    'sentiment_trend', 'late_task_ratio', 'issue_to_task_ratio', 'reassignment_count', 
    'scope_change_count', 'avg_open_issue_age', 'has_blocker', 'has_risk', 
    'has_timeline_concern', 'has_budget_concern', 
    'tcpi', 'task_lateness_variance', 'stagnant_task_count', 'weighted_sentiment_risk'
]

def convert_training_data():
    """
    Converts the JSON output from the API into the CSV file needed for training.
    """
    try:
        df = pd.read_json('engineered_project_features_v2.json')
        
        # Keep the full version for Day 3 analysis
        df.to_csv('engineered_project_features_v2.csv', index=False)

        # Create the numeric-only version for Day 2 model training
        model_ready_df = df[FINAL_FEATURE_SET_V2]
        model_ready_df.to_csv('model_ready_project_features_v2.csv', index=False)
        
        print("✅ Successfully created 'engineered_project_features_v2.csv' and 'model_ready_project_features_v2.csv'.")
        print("You can now run the Day 2 and Day 3 training scripts.")

    except Exception as e:
        print(f"❌ Error converting file: {e}")

if __name__ == '__main__':
    convert_training_data()