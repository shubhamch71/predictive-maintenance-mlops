# Verification Checklist

Use this checklist to verify that the Predictive Maintenance MLOps platform is fully functional.

## Pre-Deployment Checks

### Environment Setup

- [ ] Python 3.11+ installed
- [ ] Docker Desktop installed and running
- [ ] Kubernetes enabled in Docker Desktop
- [ ] kubectl configured and working (`kubectl cluster-info`)
- [ ] Helm installed (`helm version`)
- [ ] At least 8GB RAM allocated to Docker
- [ ] At least 4 CPU cores allocated to Docker

### Credentials Configuration

- [ ] Docker Hub account created
- [ ] Docker Hub access token generated (not password!)
- [ ] Jenkins credentials configured (`dockerhub-credentials`)
- [ ] `.env` file created from `.env.example`
- [ ] Environment variables properly set

## File Verification

### Jenkins CI/CD Files

- [ ] `jenkins/Jenkinsfile` - Main pipeline definition
- [ ] `jenkins/deploy-all.sh` - Master deployment script
- [ ] `jenkins/docker-credentials.sh` - Docker auth helper
- [ ] `jenkins/cleanup.sh` - Cleanup script

### Docker Files

- [ ] `docker-compose.yml` - Local development setup
- [ ] `docker-compose.ci.yml` - CI/CD overrides
- [ ] `docker/mlflow/Dockerfile` - MLflow image
- [ ] `docker/training/Dockerfile` - Training image
- [ ] `docker/serving/Dockerfile` - Serving image
- [ ] `docker/prometheus/prometheus.yml` - Prometheus config
- [ ] `docker/grafana/provisioning/datasources/datasources.yml` - Grafana datasources

### Requirements Files

- [ ] `requirements.txt` - Production dependencies
- [ ] `requirements-dev.txt` - Development dependencies
- [ ] `requirements-k8s.txt` - Kubernetes dependencies

### Configuration Files

- [ ] `.env.example` - Environment template
- [ ] `src/__version__.py` - Version information

### Documentation

- [ ] `README.md` - Complete with all sections
- [ ] `docs/JENKINS_SETUP.md` - Jenkins setup guide
- [ ] `docs/DOCKER_HUB_SETUP.md` - Docker Hub auth guide
- [ ] `CONTRIBUTING.md` - Contribution guidelines
- [ ] `LICENSE` - MIT License

### Example Scripts

- [ ] `examples/predict.py` - Single prediction example
- [ ] `examples/batch_predict.py` - Batch prediction example

### Utility Scripts

- [ ] `scripts/run-local.sh` - Quick start script
- [ ] `scripts/setup-dev.sh` - Development setup

### GitHub Actions

- [ ] `.github/workflows/ci.yml` - CI/CD workflow

## Deployment Verification

### Jenkins Pipeline Stages

- [ ] Stage 1: Code Quality & Testing passes
- [ ] Stage 2: Docker Images build successfully
- [ ] Stage 3: Images push to Docker Hub
- [ ] Stage 4: Kubernetes infrastructure deploys
- [ ] Stage 5: Kubeflow Pipelines install
- [ ] Stage 6: Training pipeline executes
- [ ] Stage 7: Models deploy with KServe
- [ ] Stage 8: End-to-end tests pass
- [ ] Stage 9: Reports generate
- [ ] Stage 10: Notifications send

### Docker Compose (Local)

- [ ] `docker-compose up -d` completes without errors
- [ ] PostgreSQL container healthy
- [ ] MLflow container healthy
- [ ] API container healthy
- [ ] Prometheus container running
- [ ] Grafana container running

### Service Accessibility

- [ ] MLflow UI: http://localhost:5000
- [ ] API Swagger: http://localhost:8000/docs
- [ ] API Health: http://localhost:8000/health
- [ ] Prometheus: http://localhost:9090
- [ ] Grafana: http://localhost:3000

### Kubernetes Deployment

- [ ] Namespace `mlops` created
- [ ] PostgreSQL pod running
- [ ] MLflow pod running
- [ ] Model serving pods running
- [ ] Prometheus pod running
- [ ] Grafana pod running

## Functional Verification

### MLflow

- [ ] MLflow UI loads
- [ ] Can create experiments
- [ ] Can log runs
- [ ] Model registry accessible

### API

- [ ] Health endpoint returns 200
- [ ] Predict endpoint accepts requests
- [ ] Metrics endpoint returns Prometheus format
- [ ] Swagger UI renders correctly

### Predictions

- [ ] `python examples/predict.py` runs successfully
- [ ] Predictions return valid RUL values
- [ ] Confidence scores are between 0 and 1
- [ ] Response times are < 100ms

### Monitoring

- [ ] Prometheus scraping targets
- [ ] Grafana datasource connected
- [ ] Dashboards load correctly
- [ ] Metrics visible in Prometheus

### Tests

- [ ] `pytest tests/ -v` passes
- [ ] Coverage >= 80%
- [ ] No security issues from bandit
- [ ] Code formatting passes (black)

## Production Readiness

### Security

- [ ] No hardcoded credentials in code
- [ ] Docker images use non-root users
- [ ] Secrets managed via environment variables
- [ ] Access tokens used instead of passwords

### Performance

- [ ] P95 latency < 100ms
- [ ] Throughput > 100 req/sec
- [ ] Memory usage stable
- [ ] No memory leaks detected

### Reliability

- [ ] Health checks configured
- [ ] Restart policies set
- [ ] Resource limits defined
- [ ] Logging configured

### Documentation

- [ ] README covers all use cases
- [ ] API documented with examples
- [ ] Troubleshooting guide complete
- [ ] Contributing guidelines clear

## Final Sign-Off

| Verification | Verified By | Date |
|--------------|-------------|------|
| All files created | | |
| Jenkins pipeline works | | |
| Docker images build | | |
| Images push to registry | | |
| Kubernetes deployment works | | |
| MLflow tracks experiments | | |
| Models serve predictions | | |
| Monitoring functional | | |
| Documentation complete | | |
| Ready for production | | |

---

## Quick Verification Commands

```bash
# Check all services
docker-compose ps

# Test API
curl http://localhost:8000/health

# Run prediction example
python examples/predict.py

# Run tests
pytest tests/ -v --cov=src

# Check Kubernetes
kubectl get pods -n mlops

# View logs
docker-compose logs -f
```

## Common Issues

### If services don't start:
```bash
# Check Docker resources
docker stats

# Check logs
docker-compose logs <service-name>

# Restart services
docker-compose restart
```

### If Kubernetes fails:
```bash
# Check cluster status
kubectl cluster-info

# Check pod status
kubectl describe pod <pod-name> -n mlops

# Check events
kubectl get events -n mlops
```

### If tests fail:
```bash
# Run specific test
pytest tests/test_api.py -v

# Check coverage
pytest tests/ --cov=src --cov-report=html
```
