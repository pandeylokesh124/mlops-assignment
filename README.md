# Heart Disease Classification: MLOps Pipeline

This repository contains an end-to-end MLOps pipeline for classifying heart disease. It demonstrates data preprocessing, model training, experiment tracking, model registry, and containerized API deployment using FastAPI and Docker.

## Project Architecture

1. **Data Preprocessing**: Generic pipeline using `ColumnTransformer`, `StandardScaler`, and `OneHotEncoder` to prepare raw data.
2. **Model Training**: Object-oriented scripts for training Random Forest, SVC (Calibrated), and XGBoost models. Preprocessing and modeling steps are bundled into a Scikit-Learn `Pipeline` to prevent data leakage and simplify inference.
3. **Experiment Tracking & Registry**: **MLflow** is used to track cross-validation metrics, parameters, and log model artifacts (`.skops` format). Models are registered in the MLflow Model Registry.
4. **Model Serving**: A **FastAPI** application serves predictions. It dynamically downloads the registered model pipeline from the MLflow tracking server during application startup.
5. **Containerization**: The FastAPI service is containerized using **Docker**, communicating with the host-hosted MLflow server over HTTP.

## Project Structure

```
├── data/
│   └── processed/
│       └── heart_disease_clean.csv      # Cleaned dataset
├── docker/
│   └── Dockerfile                       # Docker configuration for FastAPI
├── experiments/
│   ├── mlflow.db                        # MLflow SQLite backend store
│   └── mlruns/                          # MLflow artifact storage
├── helm/
│   └── templates/
│       ├── deployment.yaml    # K8s Deployment (Replicas, Envs, Probes)
│       └── service.yaml       # K8s Service (NodePort 30080)
├── models/
│   └── classification_models.py
├── outputs/
│   └── predictions/
│           └── output.json
│   └── reports/
├── src/
│   ├── apis/
│   │   └── predict_api.py               # FastAPI application
│   ├── feature/
│   │   └── prepare_data.py              # Data preprocessing logic
│   └── models/
│       ├── random_forest.py                 # Random Forest training pipeline
│       ├── support_vector_classifier.py     # SVC training pipeline
│       └── xgboost_model.py                 # XGBoost training pipeline
├── visualizations/
│   └── visualize_data.py                 # Data visualisations
├── requirements.txt                     # Python dependencies
├── orchestrate.ps1                      # Powershell script to automate the deployment
└── README.md                            # Project documentation
```

## Setup & Installation

### Prerequisites
* Python 3.10+
* Docker Desktop (with WSL 2 backend if on Windows)
* Git
* Kubernetes
* fastapi>=0.110.0
* uvicorn>=0.28.0
* pydantic>=2.6.4
* mlflow>=2.11.0
* scikit-learn>=1.4.0
* skops>=0.10.0
* pandas>=2.2.0
* numpy>=1.26.0
* prometheus-fastapi-instrumentator
* mlflow-skinny
* pytest

### 1. Clone the repository and install dependencies
```bash
git clone <your-github-repo-url>
cd MLOpsAssignment
pip install -r requirements.txt
```

## Running MLflow Tracking Server

To allow the Docker container to fetch model artifacts over HTTP, start the MLflow server with `--serve-artifacts` and host binding configured:

**PowerShell (Windows Host):**
```powershell
$env:MLFLOW_ALLOWED_HOSTS="*"

mlflow server `
  --backend-store-uri sqlite:///$(pwd)/experiments/mlflow.db `
  --artifacts-destination file:///$(pwd)/experiments/mlruns `
  --host 0.0.0.0 `
  --port 5000 `
  --serve-artifacts `
  --artifacts-host host.docker.internal `
  --workers 1
```

## Training & Registering Models

With the MLflow server running, execute a training script. The script builds a Scikit-Learn Pipeline, evaluates it via Stratified K-Fold CV, and logs it to MLflow.

```powershell
# Set tracking URI to local server
$env:MLFLOW_TRACKING_URI="http://localhost:5000"

# Run script. This script trains the different models and logs all the params, metrics and model to the mlflow for experiment tracking
python experiments/mlflow_runner.py
```

*After training, open the MLflow UI (`http://localhost:5000`), navigate to the Model Registry, and register the new run (e.g., as "Random_Forest_model", Version 1).*

## Docker Deployment & Configuration

### Docker Image Overview
The service is packaged in a lightweight container built using Python 3.10-slim. It installs required dependencies (`fastapi`, `uvicorn`, `mlflow`, `scikit-learn`, `skops`, `xgboost`, `pandas`) and exposes port `8000`.

### Container Environment Variables
When running the Docker container, you can configure how it connects to MLflow and which model version to pull using environment variables:

| Environment Variable | Description | Default Value |
| :--- | :--- | :--- |
| `MLFLOW_TRACKING_URI` | The URI where the MLflow tracking server is hosted. Use `http://host.docker.internal:5000` for Windows/Mac hosts. | `http://host.docker.internal:5000` |
| `MODEL_NAME` | The registered model name in MLflow. | `Random_Forest_model` |
| `MODEL_STAGE` | The specific model version number (e.g., `"1"`, `"2"`) or alias (e.g., `"challenger"`) to load. | `1` |

### 1. Build the Docker Image
```powershell
docker build -f docker/Dockerfile -t heart-disease-api:v1 .
```

### 2. Run the Container
```powershell
docker run -d `
  --name heart-disease-service `
  -p 8000:8000 `
  -e MLFLOW_TRACKING_URI="http://host.docker.internal:5000" `
  -e MODEL_NAME="Random Forest model" `
  -e MODEL_STAGE="1" `
  heart-disease-api:v1
```

---

## API Endpoints Reference

The FastAPI service exposes the following endpoints for health monitoring and real-time inference:

### 1. Health Check (`GET /health`)
Verifies that the FastAPI service is running and checks whether the MLflow model pipeline has been successfully fetched and loaded into memory during startup.

* **Request:** `GET http://localhost:8000/health`
* **Response Body:**
  ```json
  {
    "status": "healthy",
    "model_loaded": true
  }
  ```

### 2. Prediction Endpoint (`POST /predict`)
Accepts raw patient clinical features (13 base attributes). The embedded Scikit-Learn `Pipeline` automatically applies the saved `ColumnTransformer` (scaling and one-hot encoding) before passing the transformed features to the classifier.

* **Request:** `POST http://localhost:8000/predict`
* **Content-Type:** `application/json`
* **Request Payload Example:**
  ```json
  {
    "age": 63.0,
    "sex": 1,
    "cp": 3,
    "trestbps": 145.0,
    "chol": 233.0,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150.0,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }
  ```
* **Response Body Example:**
  ```json
  {
    "predicted_class": 0
  }
  ```

---

## Testing the API via PowerShell

**Health Check:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health"
```

**Prediction Request:**
```powershell
$body = @{
    age = 63
    sex = 1
    cp = 3
    trestbps = 145
    chol = 233
    fbs = 1
    restecg = 0
    thalach = 150
    exang = 0
    oldpeak = 2.3
    slope = 0
    ca = 0
    thal = 1
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $body
```
