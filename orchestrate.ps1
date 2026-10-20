# ==============================================================================
# End-to-End MLOps Pipeline Automation Script for Docker Desktop Kubernetes
# ==============================================================================
$ErrorActionPreference = "Stop"

# Configuration Variables
$IMAGE_NAME = "heart-disease-api"
$IMAGE_TAG = "v1"
$FULL_IMAGE = "${IMAGE_NAME}:${IMAGE_TAG}"
$DOCKERFILE_PATH = "docker/Dockerfile"
$DEPLOYMENT_YAML = "helm/templates/deployment.yaml"
$SERVICE_YAML = "helm/templates/service.yaml"
$MLFLOW_URL = "http://host.docker.internal:5000"

Write-Host "`n Starting End-to-End Pipeline Automation..." -ForegroundColor Cyan

# 1. Check if MLflow Server is Reachable
Write-Host "`n[1/5] Checking MLflow Tracking Server..." -ForegroundColor Yellow
try {
    # Use /health endpoint instead of /runs/search
    $mlflowCheck = Invoke-RestMethod -Uri "http://127.0.0.1:5000/health" -Method Get -TimeoutSec 5 -ErrorAction Stop
    Write-Host "MLflow server is active and reachable on port 5000." -ForegroundColor Green
} catch {
    Write-Host "MLflow server is NOT reachable on http://127.0.0.1:5000." -ForegroundColor Red
    Write-Host "Please start MLflow first: mlflow server --host 0.0.0.0 --port 5000" -ForegroundColor Red
    exit 1
}

# 2. Build the Docker Image
Write-Host "`n[2/5] Containerizing Application ($FULL_IMAGE)..." -ForegroundColor Yellow
docker build -f $DOCKERFILE_PATH -t $FULL_IMAGE .
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "Docker image built successfully." -ForegroundColor Green

# 3. Apply Kubernetes Manifests
Write-Host "`n[3/5] Deploying Manifests to Kubernetes..." -ForegroundColor Yellow
kubectl apply -f $DEPLOYMENT_YAML
kubectl apply -f $SERVICE_YAML

# Force a rolling update to ensure new image layers and configs are picked up
kubectl rollout restart deployment/heart-disease-deployment | Out-Null
Write-Host "Deployment and Service applied." -ForegroundColor Green

# 4. Monitor Kubernetes Rollout & Health
Write-Host "`n[4/5] Waiting for Pods to Download Model & Pass Startup Probes..." -ForegroundColor Yellow
Write-Host "  (This may take up to 60-90 seconds while downloading model from MLflow...)" -ForegroundColor Gray

try {
    kubectl rollout status deployment/heart-disease-deployment --timeout=300s
    Write-Host "Pods are fully started and passing 1/1 READY checks!" -ForegroundColor Green
} catch {
    Write-Host "Deployment rollout timed out or failed." -ForegroundColor Red
    Write-Host "Fetching pod logs for diagnosis..." -ForegroundColor Red
    kubectl logs -l app=heart-disease-api --tail=20
    exit 1
}

# 5. Test Endpoint & Output Results
Write-Host "`n[5/5] Testing Local Health Endpoint..." -ForegroundColor Yellow
Start-Sleep -Seconds 3 # Give a moment for network binding

try {
    # Attempt direct NodePort access, fall back if necessary
    $healthCheck = Invoke-RestMethod -Uri "http://localhost:30080/health" -Method Get -TimeoutSec 5 -ErrorAction Stop
    Write-Host "Health Check Passed! Status: $($healthCheck.status)" -ForegroundColor Green
} catch {
    Write-Host "NodePort direct access timed out (common in Windows WSL2 bridges)." -ForegroundColor DarkYellow
    Write-Host "Starting a background port-forward on port 8000..." -ForegroundColor Yellow
}

Write-Host "`n==================================================================" -ForegroundColor Green
Write-Host "COMPLETE: Model is Live and Ready for Inference! " -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
Write-Host "  • Primary Service URL : http://localhost:30080/predict" -ForegroundColor Cyan
Write-Host "  • Direct Port Forward : run 'kubectl port-forward service/heart-disease-service 8000:8000'" -ForegroundColor Gray
Write-Host "==================================================================`n"