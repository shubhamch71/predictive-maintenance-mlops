# MLOps Platform Scripts

Deployment and management scripts for the Predictive Maintenance MLOps platform on Docker Desktop Kubernetes.

## Prerequisites

### System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 12 GB |
| CPU | 4 cores | 6 cores |
| Disk | 30 GB | 50 GB |

### Required Software

1. **Docker Desktop** with Kubernetes enabled
   - Download: https://www.docker.com/products/docker-desktop
   - Enable Kubernetes: Settings → Kubernetes → Enable Kubernetes

2. **kubectl** (included with Docker Desktop)
   - Verify: `kubectl version`

3. **Python 3.9+** (for training and drift simulation)
   - Download: https://www.python.org/downloads/

4. **Helm** (optional, for Prometheus stack)
   - Install: https://helm.sh/docs/intro/install/

## Quick Start

```bash
# 1. Make scripts executable
chmod +x scripts/*.sh

# 2. Set up Kubernetes environment
./scripts/setup_local_k8s.sh

# 3. Deploy MLflow
./scripts/deploy_mlflow.sh

# 4. Run training pipeline
./scripts/run_training_pipeline.sh

# 5. Deploy models
./scripts/deploy_models.sh

# 6. Access all services
./scripts/access_all_services.sh
```

## Scripts Overview

### 1. setup_local_k8s.sh

Sets up the complete Kubernetes environment on Docker Desktop.

**What it does:**
- Verifies Docker Desktop and Kubernetes are running
- Creates namespaces (mlops, monitoring, argo)
- Installs Kubeflow Pipelines
- Installs KServe for model serving
- Installs Argo Workflows for automation
- Installs Prometheus and Grafana for monitoring

**Usage:**
```bash
./scripts/setup_local_k8s.sh
```

**Duration:** 5-10 minutes (depending on image downloads)

---

### 2. deploy_mlflow.sh

Deploys MLflow tracking server with PostgreSQL backend.

**What it does:**
- Deploys PostgreSQL database
- Creates persistent volumes for artifacts
- Deploys MLflow server
- Sets up port-forwarding to localhost:5000
- Configures MLFLOW_TRACKING_URI environment variable

**Usage:**
```bash
./scripts/deploy_mlflow.sh
```

**Access:** http://localhost:5000

---

### 3. run_training_pipeline.sh

Compiles and runs the training pipeline on Kubeflow.

**What it does:**
- Compiles the Kubeflow pipeline YAML
- Submits pipeline run to Kubeflow
- Monitors pipeline execution
- Logs results to MLflow

**Usage:**
```bash
# Run on Kubeflow
./scripts/run_training_pipeline.sh

# Run locally (without Kubeflow)
./scripts/run_training_pipeline.sh --local
```

**Access:** http://localhost:8080 (Kubeflow UI)

---

### 4. deploy_models.sh

Builds and deploys model serving infrastructure.

**What it does:**
- Builds Docker image for model serving
- Pushes to local registry (or DockerHub)
- Deploys KServe InferenceServices
- Sets up port-forwarding for model API
- Runs smoke tests

**Usage:**
```bash
# Use local registry
./scripts/deploy_models.sh

# Use DockerHub
./scripts/deploy_models.sh --dockerhub

# Use custom registry
./scripts/deploy_models.sh --registry my-registry.com
```

**Access:** http://localhost:8000

---

### 5. simulate_drift.py

Simulates data drift to test the monitoring and retraining pipeline.

**What it does:**
- Generates synthetic drifted sensor data
- Sends predictions to the model API
- Calculates drift metrics (PSI, KS statistic)
- Triggers retraining workflow via webhook
- Generates drift report

**Usage:**
```bash
# Run drift simulation
python scripts/simulate_drift.py

# With custom parameters
python scripts/simulate_drift.py --num-requests 500 --drift-magnitude 0.2

# Cleanup
python scripts/simulate_drift.py --cleanup
```

**Output:** `drift_report.json`

---

### 6. access_all_services.sh

Starts port-forwards for all services and creates a dashboard.

**What it does:**
- Sets up port-forwarding for all services
- Creates an HTML dashboard with links
- Opens dashboard in browser

**Services:**
| Service | URL |
|---------|-----|
| MLflow | http://localhost:5000 |
| Kubeflow | http://localhost:8080 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Model API | http://localhost:8000 |
| Argo Workflows | http://localhost:2746 |

**Usage:**
```bash
./scripts/access_all_services.sh
```

---

### 7. stop_all_services.sh

Stops all port-forward processes.

**Usage:**
```bash
./scripts/stop_all_services.sh
```

---

### 8. cleanup.sh

Complete cleanup of all resources.

**What it does:**
- Stops all port-forwards
- Deletes Kubernetes namespaces and resources
- Removes Docker images (optional)
- Cleans up local files

**Usage:**
```bash
# Standard cleanup
./scripts/cleanup.sh

# Full cleanup including Docker images
./scripts/cleanup.sh --all

# Cleanup but keep raw data
./scripts/cleanup.sh --keep-data
```

## Docker Desktop Configuration

### Enable Kubernetes

1. Open Docker Desktop
2. Go to **Settings** (gear icon)
3. Click **Kubernetes** in the left sidebar
4. Check **Enable Kubernetes**
5. Click **Apply & Restart**
6. Wait for Kubernetes to start (green indicator)

### Allocate Resources

1. Go to **Settings** → **Resources**
2. Configure:
   - **CPUs:** 4-6
   - **Memory:** 8-12 GB
   - **Swap:** 2 GB
   - **Disk image size:** 50 GB
3. Click **Apply & Restart**

### Windows-Specific (WSL2)

1. Enable **WSL 2 based engine** in Settings → General
2. In WSL2 settings, ensure adequate memory allocation:
   ```
   # Create/edit ~/.wslconfig
   [wsl2]
   memory=12GB
   processors=6
   ```
3. Restart WSL: `wsl --shutdown`

## Troubleshooting

### Docker Desktop Not Starting

**Symptoms:** Docker icon spinning, Kubernetes not available

**Solutions:**
1. Restart Docker Desktop
2. Reset Kubernetes cluster: Settings → Kubernetes → Reset Kubernetes Cluster
3. Check system resources (RAM, disk space)
4. On Windows: Restart WSL2 (`wsl --shutdown`)

### Kubernetes Context Issues

**Symptoms:** `kubectl` commands fail or connect to wrong cluster

**Solution:**
```bash
# List contexts
kubectl config get-contexts

# Switch to docker-desktop
kubectl config use-context docker-desktop

# Verify
kubectl cluster-info
```

### Insufficient Resources

**Symptoms:** Pods stuck in `Pending` or `OOMKilled`

**Solutions:**
1. Increase Docker Desktop memory allocation
2. Close other applications
3. Scale down replicas:
   ```bash
   kubectl scale deployment <name> --replicas=1 -n mlops
   ```

### Port Conflicts

**Symptoms:** "Port already in use" errors

**Solutions:**
```bash
# Find process using port
lsof -i :5000  # macOS/Linux
netstat -ano | findstr :5000  # Windows

# Kill process
kill <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows

# Or use different ports
export MLFLOW_PORT=5001
```

### Pods Not Starting

**Symptoms:** Pods in `ImagePullBackOff` or `CrashLoopBackOff`

**Solutions:**
```bash
# Check pod details
kubectl describe pod <pod-name> -n <namespace>

# Check logs
kubectl logs <pod-name> -n <namespace>

# For ImagePullBackOff: Check image name and registry access
# For CrashLoopBackOff: Check application logs and configuration
```

### Storage Issues

**Symptoms:** PVC stuck in `Pending`

**Solutions:**
```bash
# Check storage class
kubectl get storageclass

# Docker Desktop uses 'hostpath' provisioner
# Ensure PVC requests use the correct storage class or leave it default
```

### KServe Not Working

**Symptoms:** InferenceService not getting Ready

**Solutions:**
1. Check KServe controller:
   ```bash
   kubectl get pods -n kserve
   kubectl logs -n kserve deploy/kserve-controller-manager
   ```

2. Check Istio/networking (KServe may need Istio):
   ```bash
   # For local development, use RawDeployment mode
   kubectl patch configmap/inferenceservice-config -n kserve \
     --type=merge -p '{"data":{"deploy":"{\"defaultDeploymentMode\": \"RawDeployment\"}"}}'
   ```

## Performance Tips

### Faster Image Pulls

Use a local registry to avoid repeated pulls:
```bash
# Start local registry
docker run -d -p 5000:5000 --name registry registry:2

# Tag and push images locally
docker tag myimage:latest localhost:5000/myimage:latest
docker push localhost:5000/myimage:latest
```

### Reduce Resource Usage

1. **Scale down unused components:**
   ```bash
   kubectl scale deployment prometheus-grafana --replicas=0 -n monitoring
   ```

2. **Use resource limits:**
   All manifests include resource requests/limits to prevent runaway usage.

3. **Clean up completed pods:**
   ```bash
   kubectl delete pods --field-selector=status.phase=Succeeded -A
   kubectl delete pods --field-selector=status.phase=Failed -A
   ```

### Monitor Resource Usage

```bash
# Kubernetes dashboard (if installed)
kubectl proxy
# Open: http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/

# Node resource usage
kubectl top nodes

# Pod resource usage
kubectl top pods -A
```

## File Structure

```
scripts/
├── README.md                 # This file
├── setup_local_k8s.sh       # Kubernetes setup
├── deploy_mlflow.sh         # MLflow deployment
├── run_training_pipeline.sh # Pipeline execution
├── deploy_models.sh         # Model deployment
├── simulate_drift.py        # Drift simulation
├── access_all_services.sh   # Port-forward setup
├── stop_all_services.sh     # Stop port-forwards
└── cleanup.sh               # Complete cleanup
```

## Support

- **Documentation:** See project README.md
- **Issues:** Check troubleshooting section above
- **Logs:** Check `*.log` files in project root after running scripts
