# Save this file as: train_model_v2.py

import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# --- V2 Configuration ---
INPUT_FILE = 'model_ready_project_features_v2.csv'
ENGINEERED_FILE = 'engineered_project_features_v2.csv'
CLUSTERED_OUTPUT_FILE = 'clustered_project_data_v2.csv'
MODEL_FILE = 'kmeans_model_v2.joblib'
SCALER_FILE = 'scaler_v2.joblib'
MAX_CLUSTERS_TO_TEST = 10

def train_health_model_v2():
    """
    Loads the V2 preprocessed data, scales it, determines the optimal number 
    of clusters using the Elbow Method, trains the final KMeans model, 
    and saves the model and scaler artifacts.
    """
    # --- Step 1: Load the preprocessed data ---
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at '{INPUT_FILE}'.")
        print("Please run the 'convert_to_csv.py' script first.")
        return

    print(f"Loading V2 preprocessed data from '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)
    
    # --- Step 2: Scale the Features ---
    print("Scaling V2 features using StandardScaler...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)

    # --- Step 3: Determine Optimal K using the Elbow Method ---
    print(f"Calculating model inertia for K from 1 to {MAX_CLUSTERS_TO_TEST}...")
    inertia_values = []
    k_range = range(1, MAX_CLUSTERS_TO_TEST + 1)
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans_temp.fit(scaled_features)
        inertia_values.append(kmeans_temp.inertia_)

    # Plot the Elbow Method graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia_values, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Sum of squared distances)')
    plt.title('V2 Elbow Method for Optimal K')
    plt.xticks(k_range)
    plt.grid(True)
    print("\n--- ACTION REQUIRED ---")
    print("The 'Elbow Method' plot has been generated.")
    print("Look for the 'elbow' point on the curve where the rate of decrease slows down.")
    print("Close the plot window to continue.")
    plt.show()

    # --- Step 4: Get User Input for K and Train Final Model ---
    optimal_k = 0
    while optimal_k <= 1:
        try:
            optimal_k = int(input("Enter the optimal number of clusters (K) for the V2 model: "))
            if optimal_k <= 1: print("Please enter a number greater than 1.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    print(f"\nTraining final V2 K-Means model with K={optimal_k}...")
    final_kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=42)
    final_kmeans.fit(scaled_features)
    print("V2 Model training complete.")

    # --- Step 5: Save the Model and Scaler ---
    joblib.dump(final_kmeans, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"✅ V2 Model saved to '{MODEL_FILE}'")
    print(f"✅ V2 Scaler saved to '{SCALER_FILE}'")

    # --- Step 6: Save Clustered Data for Analysis in Day 3 ---
    full_df = pd.read_csv(ENGINEERED_FILE)
    full_df['health_cluster'] = final_kmeans.labels_
    full_df.to_csv(CLUSTERED_OUTPUT_FILE, index=False)
    print(f"✅ Clustered data for analysis saved to '{CLUSTERED_OUTPUT_FILE}'")
    print("\n--- Day 2 Training Complete ---")

if __name__ == "__main__":
    train_health_model_v2()