import os
import shutil
from contextlib import asynccontextmanager

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "Random Forest model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "1")

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    try:
        if MODEL_STAGE.isdigit():
            model_version_details = client.get_model_version(name=MODEL_NAME, version=MODEL_STAGE)
        else:
            model_version_details = client.get_model_version_by_alias(name=MODEL_NAME, alias=MODEL_STAGE)

        run_id = model_version_details.run_id

        local_dir = "/tmp/model_artifacts"
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)

        print(f"Downloading artifacts for run {run_id}...")
        downloaded_path = client.download_artifacts(run_id, "model", dst_path=local_dir)
        print(f"Artifacts downloaded to: {downloaded_path}")

        model = mlflow.pyfunc.load_model(downloaded_path)
        print(f"Loaded model successfully from local path.")

    except Exception as e:
        print(f"Failed to load model from MLflow: {e}")

    yield
    model = None


app = FastAPI(
    title="Heart Disease Classification API",
    description="Inference service for MLflow Registered Random Forest Model",
    version="1.0.0",
    lifespan=lifespan,
)

class PatientFeatures(BaseModel):
    age: float = Field(..., example=63.0)
    sex: int = Field(..., example=1)
    cp: int = Field(..., example=3)
    trestbps: float = Field(..., example=145.0)
    chol: float = Field(..., example=233.0)
    fbs: int = Field(..., example=1)
    restecg: int = Field(..., example=0)
    thalach: float = Field(..., example=150.0)
    exang: int = Field(..., example=0)
    oldpeak: float = Field(..., example=2.3)
    slope: int = Field(..., example=0)
    ca: int = Field(..., example=0)
    thal: int = Field(..., example=1)


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(patient: PatientFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    input_data = pd.DataFrame([patient.model_dump()])

    try:
        prediction = model.predict(input_data)
        prediction_class = int(prediction[0])

        response = {"predicted_class": prediction_class}

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0].tolist()
            response["probabilities"] = {f"class_{i}": float(p) for i, p in enumerate(probabilities)}

        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

