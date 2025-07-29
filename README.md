# Project Sentinel: Setup and Running Instructions

Project Sentinel is an automated intelligence system that provides proactive health scores for projects in Adobe Workfront by leveraging sentiment analysis and machine learning.

## System Components

The system consists of two main parts that work together:
1.  **Adobe Workfront Fusion Scenario:** Acts as the "Orchestrator." It gathers all the necessary data, calls our custom AI services, and logs the final results back into Workfront.
2.  **Python Backend (The "Brain"):** This is the intelligence core of the system, made of two microservices:
    *   **Processor API (`app.py`):** Takes raw project data and transforms it into a clean, feature-rich dataset.
    *   **Predictor API (`predict_api.py`):** Takes the feature-rich data and uses the trained ML model to calculate the final health score.

---

## 1. Prerequisites

Before you begin, ensure you have the following installed on the machine that will run the Python backend:

*   **Python (version 3.10 or higher)**
*   **pip** (Python's package installer)
*   **Ollama:** The engine for our local language model. Follow the instructions at [ollama.ai](https://ollama.ai/) to install it.
    *   After installing Ollama, you must download the specific model used by this project. Open your terminal and run:
        ```bash
        ollama pull mistral:instruct
        ```
*   **ngrok:** A tool to create a secure tunnel from the internet to our local Python APIs. [Download it here](https://ngrok.com/download).

---

## 2. Setup & Model Training

This is a one-time process to train the initial machine learning model.

### Step 2.1: Prepare the Python Environment

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-folder>
    ```

2.  **Create a Virtual Environment:** (This creates a clean, isolated workspace for this project's Python packages).
    ```bash
    # Create the environment folder
    python -m venv venv

    # Activate it (on Windows)
    .\venv\Scripts\activate

    # Activate it (on macOS/Linux)
    source venv/bin/activate
    ```

3.  **Install Required Libraries:**
    ```bash
    pip install Flask pandas numpy requests dirtyjson scikit-learn joblib
    ```

### Step 2.2: The "Bootstrap" Run (Generate Your First Training Dataset)

Your Fusion scenario is smartly designed to help create your initial training data.

1.  **Start the Backend Services:**
    *   In one terminal, start the Ollama AI model server:
        ```bash
        ollama serve
        ```
    *   In a second terminal (with your `venv` activated), start the **Processor API**—this is the service that will engineer the features.
        ```bash
        python app.py
        ```
    *   In a third terminal, expose the Processor API to the internet using ngrok:
        ```bash
        ngrok http 5001
        ```
        (Copy the `https://...ngrok-free.app` URL it provides).

2.  **Run Fusion to Capture Data:**
    *   In your Fusion scenario, update the HTTP module that calls `/process_projects` to use the `ngrok` URL from the previous step.
    *   For this first run, ensure your scenario is set to pull a good variety of projects (at least 20 is recommended).
    *   Run the Fusion scenario **once**. It is configured to automatically save a file named `engineered_project_features_v2.json` to your specified cloud storage (e.g., OneDrive, Google Drive).

3.  **Prepare Training Files:**
    *   **Download** the `engineered_project_features_v2.json` file from your cloud storage and place it in this project's main folder.

### Step 2.3: Train the Machine Learning Model

Now we'll use the data you just generated to build the model's intelligence.

1.  **Convert Data to CSV:** This script prepares the JSON data for our model training tools.
    ```bash
    python convert_to_csv.py
    ```

2.  **Train the Model:** This script runs the K-Means algorithm to find natural groupings in your project data.
    ```bash
    python train_model_v2.py
    ```
    *   An "Elbow Method" plot will appear. Look for the "elbow" of the curve—the point where the line starts to flatten out. This is your optimal number of health categories.
    *   Close the plot and enter your chosen number (e.g., `4`) into the terminal.
    *   This will create your trained model files: `kmeans_model_v2.joblib` and `scaler_v2.joblib`.

3.  **Analyze & Label the Clusters:** The model has created groups, but we need to give them meaningful names.
    ```bash
    python analyze_clusters_v2.py
    ```
    *   This script will print a summary table and save plots to the `cluster_analysis_plots_v2` folder.
    *   Study this analysis to understand the "personality" of each cluster. Is Cluster 2 consistently late? Does Cluster 0 have negative sentiment?
    *   Based on your analysis, open **`predict_api.py`** and update the `CLUSTER_MAP` dictionary with your final labels (e.g., `{0: "At Risk", 1: "On Track", 2: "In Trouble"}`).

---

## 3. Running in Operational Mode

Once the model is trained, the system is ready for its day-to-day automated runs.

1.  **Start the AI Model:** In a terminal, ensure the Ollama server is running.
    ```bash
    ollama serve
    ```

2.  **Start Both Python APIs:** In two separate terminals (both with `venv` activated), start both the Processor and the Predictor services.
    ```bash
    # In Terminal 2 (Processor):
    python app.py

    # In Terminal 3 (Predictor):
    python predict_api.py
    ```

3.  **Start ngrok for Both APIs:** In a fourth terminal, use a configuration file (`ngrok.yml`) to start both tunnels with a single, convenient command.
    ```bash
    # Your ngrok.yml should contain:
    # authtoken: YOUR_TOKEN
    # tunnels:
    #   processor:
    #     proto: http
    #     addr: 5001
    #   predictor:
    #     proto: http
    #     addr: 5002

    # Run this command to start both tunnels:
    ngrok start --all
    ```
    *   This will give you **two separate public URLs**.

4.  **Activate the Fusion Scenario:**
    *   Update the two HTTP modules in your live Fusion scenario with the correct corresponding `ngrok` URLs.
    *   Turn on the scenario and set its schedule. The Project Sentinel system is now fully operational.```
