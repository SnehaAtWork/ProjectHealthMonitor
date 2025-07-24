# Save this file as: analyze_clusters_v2.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- V2 Configuration ---
CLUSTERED_DATA_FILE = 'clustered_project_data_v2.csv'
ANALYSIS_PLOTS_DIR = 'cluster_analysis_plots_v2'
# --- THIS IS THE ONLY CHANGE MADE TO THIS SCRIPT ---
FINAL_FEATURE_SET_V2 = [
    'percentComplete', 'cost_variance_ratio', 'baseline_cpi', 'baseline_spi', 
    'schedule_slippage_days', 'days_since_last_update', 'task_count', 
    'late_task_count', 'open_task_count', 'avg_task_pct_complete', 'issue_count', 
    'open_issue_count', 'progress_vs_time_ratio', 'change_in_cpi', 
    'change_in_spi', 'sentiment_trend', 'late_task_ratio', 'issue_to_task_ratio',
    'reassignment_count', 'scope_change_count', 'avg_open_issue_age',
    'has_blocker', 'has_risk', 'has_timeline_concern', 'has_budget_concern',
    'tcpi', 'task_lateness_variance', 'stagnant_task_count', 'weighted_sentiment_risk'
]

def analyze_clusters_v2():
    """
    Analyzes the V2 clustered data to help interpret and label the clusters.
    Generates a summary table and visual plots for comparison.
    """
    if not os.path.exists(CLUSTERED_DATA_FILE):
        print(f"Error: File not found at '{CLUSTERED_DATA_FILE}'. Please run train_model_v2.py first.")
        return

    print(f"Loading V2 clustered data from '{CLUSTERED_DATA_FILE}'...")
    df = pd.read_csv(CLUSTERED_DATA_FILE)

    # --- Step 1: Quantitative Analysis (The Summary Table) ---
    print("\n--- V2 Cluster Profile Analysis (Mean Values) ---")
    
    # Group by the cluster label and calculate the mean for all model features
    cluster_profiles = df.groupby('health_cluster')[FINAL_FEATURE_SET_V2].mean().round(3)
    
    # Transposing the table makes it much easier to compare clusters
    print(cluster_profiles.T)
    cluster_profiles.T.to_csv('v2_cluster_analysis_summary.csv')
    print("\n✅ Full summary table saved to 'v2_cluster_analysis_summary.csv'")

    # --- Step 2: Visual Analysis (The Plots) ---
    if not os.path.exists(ANALYSIS_PLOTS_DIR):
        os.makedirs(ANALYSIS_PLOTS_DIR)
    
    print(f"\nGenerating and saving comparison plots to '{ANALYSIS_PLOTS_DIR}/'...")
    for feature in FINAL_FEATURE_SET_V2:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='health_cluster', y=feature, data=df, palette='viridis', ci=None)
        plt.title(f'V2 Comparison of "{feature}" Across Clusters')
        plt.savefig(os.path.join(ANALYSIS_PLOTS_DIR, f'plot_{feature}.png'))
        plt.close() # Close the plot to avoid displaying it

    print("✅ Analysis plots have been saved.")
    print("\n--- Day 3 Analysis Complete ---")
    print("Review the summary table and plots to assign meaningful labels to your new V2 clusters.")

if __name__ == "__main__":
    analyze_clusters_v2()