# Getting Started Guide

This guide provides complete step-by-step instructions to get the Predictive Maintenance MLOps platform running from scratch. Follow each section in order.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the Repository](#2-clone-the-repository)
3. [Option A: Local Development (Docker Compose)](#3-option-a-local-development-docker-compose)
4. [Option B: One-Click Deployment (Jenkins)](#4-option-b-one-click-deployment-jenkins)
5. [Option C: Kubernetes Deployment](#5-option-c-kubernetes-deployment)
6. [Verify the Installation](#6-verify-the-installation)
7. [Run Your First Prediction](#7-run-your-first-prediction)
8. [Explore the Platform](#8-explore-the-platform)
9. [Next Steps](#9-next-steps)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

### Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | 8 cores |
| Disk | 20 GB free | 50 GB free |

### Software Requirements

#### Required Software

1. **Docker Desktop** (with Kubernetes)
   - macOS: https://docs.docker.com/desktop/install/mac-install/
   - Windows: https://docs.docker.com/desktop/install/windows-install/
   - Linux: https://docs.docker.com/desktop/install/linux-install/

2. **Python 3.11+**
   ```bash
   # Check version
   python3 --version

   # Install on macOS
   brew install python@3.11

   # Install on Ubuntu
   sudo apt update && sudo apt install python3.11 python3.11-venv
   ```

3. **Git**
   ```bash
   # Check version
   git --version

   # Install on macOS
   brew install git

   # Install on Ubuntu
   sudo apt install git
   ```

4. **kubectl** (for Kubernetes deployment)
   ```bash
   # macOS
   brew install kubectl

   # Ubuntu
   sudo snap install kubectl --classic

   # Verify
   kubectl version --client
   ```

5. **Helm** (for Kubernetes deployment)
   ```bash
   # macOS
   brew install helm

   # Ubuntu
   sudo snap install helm --classic

   # Verify
   helm version
   ```

#### Enable Kubernetes in Docker Desktop

1. Open Docker Desktop
2. Go to **Settings** (gear icon)
3. Click **Kubernetes** in the left sidebar
4. Check **Enable Kubernetes**
5. Click **Apply & Restart**
6. Wait for the green "Kubernetes is running" indicator

Verify Kubernetes is working:
```bash
kubectl cluster-info
kubectl get nodes
```

---

## 2. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/your-org/predictive-maintenance-mlops.git

# Navigate to project directory
cd predictive-maintenance-mlops

# View project structure
ls -la
```

---

## 3. Option A: Local Development (Docker Compose)

**Best for**: Development, testing, and learning the platform.

### Step 3.1: Run Setup Script

```bash
# Make script executable
chmod +x scripts/setup-dev.sh

# Run setup (creates venv, installs dependencies)
./scripts/setup-dev.sh
```

This script will:
- Check Python version
- Create virtual environment
- Install all dependencies
- Setup pre-commit hooks
- Create necessary directories

### Step 3.2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Open and review settings (optional)
# Default values work for local development
nano .env  # or use your preferred editor
```

Key settings in `.env`:
```bash
# These defaults work out of the box
MLFLOW_TRACKING_URI=http://localhost:5000
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow
API_PORT=8000
```

### Step 3.3: Start Services

```bash
# Start all services in detached mode
docker-compose up -d

# Watch the startup logs (optional)
docker-compose logs -f
```

### Step 3.4: Wait for Services to be Ready

```bash
# Check service status
docker-compose ps

# All services should show "Up" and "healthy"
```

Expected output:
```
NAME              STATUS                   PORTS
mlops-api         Up (healthy)             0.0.0.0:8000->8000/tcp
mlops-grafana     Up (healthy)             0.0.0.0:3000->3000/tcp
mlops-mlflow      Up (healthy)             0.0.0.0:5000->5000/tcp
mlops-postgres    Up (healthy)             0.0.0.0:5432->5432/tcp
mlops-prometheus  Up                       0.0.0.0:9090->9090/tcp
```

### Step 3.5: Access Services

Open in your browser:

| Service | URL | Credentials |
|---------|-----|-------------|
| MLflow UI | http://localhost:5000 | None |
| API Swagger | http://localhost:8000/docs | None |
| Prometheus | http://localhost:9090 | None |
| Grafana | http://localhost:3000 | admin / admin |

### Step 3.6: Stop Services (When Done)

```bash
# Stop services (preserves data)
docker-compose down

# Stop and remove all data
docker-compose down -v
```

---

## 4. Option B: One-Click Deployment (Jenkins)

**Best for**: Production deployment with full CI/CD automation.

### Step 4.1: Install Jenkins

#### macOS
```bash
# Install Jenkins
brew install jenkins-lts

# Start Jenkins
brew services start jenkins-lts

# Get initial admin password
cat ~/.jenkins/secrets/initialAdminPassword
```

#### Ubuntu
```bash
# Add Jenkins repository
curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key | sudo tee \
  /usr/share/keyrings/jenkins-keyring.asc > /dev/null

echo deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] \
  https://pkg.jenkins.io/debian-stable binary/ | sudo tee \
  /etc/apt/sources.list.d/jenkins.list > /dev/null

# Install Jenkins
sudo apt update
sudo apt install jenkins

# Start Jenkins
sudo systemctl start jenkins

# Get initial admin password
sudo cat /var/lib/jenkins/secrets/initialAdminPassword
```

#### Docker (Quickest)
```bash
docker run -d \
  --name jenkins \
  -p 8080:8080 \
  -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  jenkins/jenkins:lts

# Get initial admin password
docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
```

### Step 4.2: Initial Jenkins Setup

1. Open http://localhost:8080
2. Enter the initial admin password
3. Click **Install suggested plugins**
4. Create your admin user
5. Accept the default Jenkins URL
6. Click **Start using Jenkins**

### Step 4.3: Install Required Plugins

1. Go to **Manage Jenkins** → **Manage Plugins**
2. Click **Available** tab
3. Search and install these plugins:
   - Pipeline
   - Docker Pipeline
   - Git
   - Credentials Binding
   - AnsiColor
   - Timestamper

4. Restart Jenkins if prompted

### Step 4.4: Configure Docker Hub Credentials

1. Go to **Manage Jenkins** → **Manage Credentials**
2. Click **(global)** → **Add Credentials**
3. Fill in:
   - **Kind**: Username with password
   - **Username**: Your Docker Hub username
   - **Password**: Your Docker Hub access token (NOT password!)
   - **ID**: `dockerhub-credentials`
   - **Description**: Docker Hub Access Token
4. Click **Create**

**How to get Docker Hub access token:**
1. Go to https://hub.docker.com/settings/security
2. Click **New Access Token**
3. Name it "Jenkins CI/CD"
4. Select permissions: Read, Write, Delete
5. Copy the token immediately

### Step 4.5: Create Pipeline Job

1. Click **New Item**
2. Enter name: `predictive-maintenance-pipeline`
3. Select **Pipeline**
4. Click **OK**

5. In configuration:
   - **Description**: MLOps pipeline for Predictive Maintenance
   - **Pipeline Definition**: Pipeline script from SCM
   - **SCM**: Git
   - **Repository URL**: `/path/to/predictive-maintenance-mlops` (local path)
     - Or: `https://github.com/your-org/predictive-maintenance-mlops.git`
   - **Script Path**: `jenkins/Jenkinsfile`

6. Click **Save**

### Step 4.6: Run the Pipeline

1. Click **Build Now**
2. Watch the pipeline progress in **Stage View**
3. Click on a stage to see logs

The pipeline will:
1. Run code quality checks
2. Build Docker images
3. Push to Docker Hub
4. Deploy Kubernetes infrastructure
5. Run training pipeline
6. Deploy models
7. Run tests
8. Generate reports

### Step 4.7: Access Deployed Services

After successful pipeline completion:

| Service | URL |
|---------|-----|
| MLflow | http://localhost:5000 |
| Kubeflow | http://localhost:8080 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| API | http://localhost:8000 |

---

## 5. Option C: Kubernetes Deployment

**Best for**: Production Kubernetes clusters.

### Step 5.1: Create Namespace

```bash
kubectl create namespace mlops
kubectl config set-context --current --namespace=mlops
```

### Step 5.2: Add Helm Repositories

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
```

### Step 5.3: Deploy PostgreSQL

```bash
helm install postgres bitnami/postgresql \
  --namespace mlops \
  --set auth.postgresPassword=mlflow \
  --set auth.database=mlflow \
  --set primary.persistence.size=5Gi

# Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=postgresql -n mlops --timeout=300s
```

### Step 5.4: Deploy MLflow

```bash
kubectl apply -f k8s/mlflow/ -n mlops

# Wait for MLflow to be ready
kubectl wait --for=condition=available deployment/mlflow -n mlops --timeout=300s
```

### Step 5.5: Deploy Monitoring Stack

```bash
# Prometheus
helm install prometheus prometheus-community/prometheus \
  --namespace mlops \
  --set server.persistentVolume.size=5Gi \
  --set alertmanager.enabled=false

# Grafana
helm install grafana grafana/grafana \
  --namespace mlops \
  --set adminPassword=admin \
  --set persistence.enabled=true
```

### Step 5.6: Deploy Models

```bash
kubectl apply -f k8s/kserve/ -n mlops
```

### Step 5.7: Setup Port Forwarding

```bash
# MLflow
kubectl port-forward svc/mlflow 5000:5000 -n mlops &

# Prometheus
kubectl port-forward svc/prometheus-server 9090:80 -n mlops &

# Grafana
kubectl port-forward svc/grafana 3000:80 -n mlops &

# Model API (if deployed)
kubectl port-forward svc/model-server 8000:8000 -n mlops &
```

### Step 5.8: Verify Deployment

```bash
# Check all pods
kubectl get pods -n mlops

# Check services
kubectl get svc -n mlops

# Check deployments
kubectl get deployments -n mlops
```

---

## 6. Verify the Installation

### Check Service Health

```bash
# API health check
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","version":"1.0.0"}

# MLflow health check
curl http://localhost:5000/health

# Prometheus health check
curl http://localhost:9090/-/healthy
```

### Check Docker Services (Option A)

```bash
docker-compose ps
docker-compose logs --tail=50
```

### Check Kubernetes Pods (Option B/C)

```bash
kubectl get pods -n mlops
kubectl get svc -n mlops
```

---

## 7. Run Your First Prediction

### Activate Virtual Environment

```bash
source venv/bin/activate
```

### Run Single Prediction Example

```bash
python examples/predict.py
```

Expected output:
```
╔══════════════════════════════════════════════════════════════════════╗
║ PREDICTIVE MAINTENANCE - PREDICTION EXAMPLE                         ║
╚══════════════════════════════════════════════════════════════════════╝

API URL: http://localhost:8000
Checking API health...
API is healthy!

Loaded 10 samples with 17 features

Sending predictions...

════════════════════════════════════════════════════════════════════════════════
                              PREDICTION RESULTS
════════════════════════════════════════════════════════════════════════════════
  # |      RUL | Confidence |        Model | Latency (ms)
────────────────────────────────────────────────────────────────────────────────
  1 |     45.2 |     92.00% |     ensemble |        15.23
  2 |     78.5 |     88.50% |     ensemble |        12.45
  ...
════════════════════════════════════════════════════════════════════════════════
```

### Run Batch Prediction Example

```bash
python examples/batch_predict.py --num-samples 100
```

### Make API Call with curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "sensor_2": 0.5,
      "sensor_3": 0.7,
      "sensor_4": 0.3,
      "sensor_7": 0.6,
      "sensor_8": 0.4,
      "sensor_9": 0.5,
      "sensor_11": 0.3,
      "sensor_12": 0.7,
      "sensor_13": 0.5,
      "sensor_14": 0.4,
      "sensor_15": 0.6,
      "sensor_17": 0.5,
      "sensor_20": 0.4,
      "sensor_21": 0.6
    }
  }'
```

---

## 8. Explore the Platform

### MLflow UI (http://localhost:5000)

1. Click **Experiments** to see training runs
2. Click a run to see metrics, parameters, artifacts
3. Click **Models** to see registered models

### Grafana Dashboards (http://localhost:3000)

1. Login with `admin` / `admin`
2. Go to **Dashboards** → **Browse**
3. View model performance metrics

### API Documentation (http://localhost:8000/docs)

1. Interactive Swagger UI
2. Try endpoints directly from browser
3. See request/response schemas

### Prometheus Metrics (http://localhost:9090)

1. Query metrics with PromQL
2. Example: `prediction_latency_seconds`
3. View targets being scraped

---

## 9. Next Steps

### Train Your Own Models

```bash
# Activate virtual environment
source venv/bin/activate

# Run training
python -m src.training.train --model-type ensemble

# Or use Kubeflow pipeline
python kubeflow/compile_pipeline.py
```

### Add Custom Data

1. Place data files in `data/raw/`
2. Update `data/download_data.py` if needed
3. Run preprocessing: `python -m src.data.preprocessor`

### Customize Monitoring

1. Add Grafana dashboards in `docker/grafana/dashboards/`
2. Add Prometheus alerts in `docker/prometheus/alerts.yml`
3. Restart services: `docker-compose restart`

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Deploy to Production

1. Update `.env` with production values
2. Use `docker-compose.ci.yml` for CI/CD
3. Configure proper secrets management
4. Enable monitoring and alerting

---

## 10. Troubleshooting

### Docker Services Won't Start

```bash
# Check Docker is running
docker info

# Check for port conflicts
lsof -i :5000
lsof -i :8000
lsof -i :3000

# Remove conflicting containers
docker-compose down -v
docker system prune -f

# Restart
docker-compose up -d
```

### Kubernetes Pods Not Starting

```bash
# Check pod status
kubectl get pods -n mlops

# Describe failing pod
kubectl describe pod <pod-name> -n mlops

# Check logs
kubectl logs <pod-name> -n mlops

# Check events
kubectl get events -n mlops --sort-by='.lastTimestamp'
```

### API Returns Errors

```bash
# Check API logs
docker-compose logs api

# Or for Kubernetes
kubectl logs deployment/api -n mlops

# Test health endpoint
curl -v http://localhost:8000/health
```

### MLflow Not Accessible

```bash
# Check MLflow container
docker-compose logs mlflow

# Restart MLflow
docker-compose restart mlflow

# Check PostgreSQL connection
docker-compose exec postgres pg_isready
```

### Out of Memory

```bash
# Check Docker resource usage
docker stats

# Increase Docker Desktop resources:
# Settings → Resources → Memory: 8GB+
# Settings → Resources → CPUs: 4+
```

### Jenkins Pipeline Fails

1. Check console output for error
2. Verify Docker Hub credentials
3. Ensure Kubernetes is running
4. Check available disk space

### Common Fixes

```bash
# Reset everything
docker-compose down -v
docker system prune -af
docker volume prune -f

# Fresh start
docker-compose up -d --build
```

---

## Quick Reference

### Start Everything

```bash
# Option A: Docker Compose
docker-compose up -d

# Option B: Quick script
./scripts/run-local.sh
```

### Stop Everything

```bash
# Docker Compose
docker-compose down

# Kubernetes
kubectl delete namespace mlops
```

### View Logs

```bash
# Docker Compose
docker-compose logs -f

# Kubernetes
kubectl logs -f deployment/mlflow -n mlops
```

### Service URLs

| Service | URL |
|---------|-----|
| MLflow | http://localhost:5000 |
| API | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| Jenkins | http://localhost:8080 |

---

## Need Help?

- **Documentation**: See `docs/` folder
- **Issues**: Check `TROUBLESHOOTING` section above
- **Contributing**: See `CONTRIBUTING.md`

---

*You're all set! Start predicting equipment failures!*
