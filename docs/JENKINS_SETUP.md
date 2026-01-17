# Jenkins Setup Guide

This guide walks you through setting up Jenkins for the Predictive Maintenance MLOps platform's one-click deployment pipeline.

## Table of Contents

1. [Jenkins Installation](#1-jenkins-installation)
2. [Plugin Configuration](#2-plugin-configuration)
3. [Credentials Setup](#3-credentials-setup)
4. [Creating the Pipeline Job](#4-creating-the-pipeline-job)
5. [Docker Hub Token Setup](#5-docker-hub-token-setup)
6. [First Pipeline Run](#6-first-pipeline-run)
7. [Automated Triggers](#7-automated-triggers)
8. [Notifications Setup](#8-notifications-setup)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Jenkins Installation

### Ubuntu/Debian

```bash
# Add Jenkins repository key
curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key | sudo tee \
  /usr/share/keyrings/jenkins-keyring.asc > /dev/null

# Add Jenkins repository
echo deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] \
  https://pkg.jenkins.io/debian-stable binary/ | sudo tee \
  /etc/apt/sources.list.d/jenkins.list > /dev/null

# Install Jenkins
sudo apt update
sudo apt install jenkins

# Start Jenkins
sudo systemctl start jenkins
sudo systemctl enable jenkins

# Get initial admin password
sudo cat /var/lib/jenkins/secrets/initialAdminPassword
```

### macOS (Homebrew)

```bash
# Install Jenkins LTS
brew install jenkins-lts

# Start Jenkins
brew services start jenkins-lts

# Get initial admin password
cat ~/.jenkins/secrets/initialAdminPassword
```

### Windows

1. Download the Windows installer from [jenkins.io/download](https://www.jenkins.io/download/)
2. Run the installer and follow the wizard
3. Jenkins will start automatically as a Windows service
4. Access Jenkins at `http://localhost:8080`
5. Find the initial admin password at:
   ```
   C:\ProgramData\Jenkins\.jenkins\secrets\initialAdminPassword
   ```

### Docker (Recommended for Testing)

```bash
# Run Jenkins in Docker
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

### Initial Setup Wizard

1. Open browser to `http://localhost:8080`
2. Enter the initial admin password
3. Click "Install suggested plugins" (or select custom)
4. Create your admin user
5. Configure Jenkins URL (use default for local)
6. Click "Start using Jenkins"

[Screenshot: Jenkins initial setup wizard]

---

## 2. Plugin Configuration

Install the following plugins for full pipeline functionality:

### Required Plugins

Navigate to: **Manage Jenkins** > **Manage Plugins** > **Available**

| Plugin | Purpose |
|--------|---------|
| **Pipeline** | Core pipeline functionality |
| **Pipeline: Stage View** | Visual pipeline stages |
| **Docker Pipeline** | Docker build/push support |
| **Git** | Git SCM integration |
| **GitHub** | GitHub webhook support |
| **Credentials Binding** | Secure credential handling |
| **SSH Credentials** | SSH key management |

### Recommended Plugins

| Plugin | Purpose |
|--------|---------|
| **Blue Ocean** | Modern UI for pipelines |
| **Kubernetes** | Deploy to Kubernetes clusters |
| **HTML Publisher** | Publish HTML reports |
| **JUnit** | Test result publishing |
| **Cobertura** | Code coverage reports |
| **Slack Notification** | Slack alerts |
| **Email Extension** | Email notifications |
| **AnsiColor** | Colored console output |
| **Timestamper** | Add timestamps to logs |
| **Build Timeout** | Prevent hanging builds |
| **Workspace Cleanup** | Clean workspaces |

### Installation Steps

1. Go to **Manage Jenkins** > **Manage Plugins**
2. Click the **Available** tab
3. Search for each plugin
4. Check the checkbox next to the plugin name
5. Click **Install without restart** or **Download now and install after restart**
6. Restart Jenkins if required

[Screenshot: Plugin installation page]

---

## 3. Credentials Setup

### Adding Docker Hub Credentials

1. Navigate to: **Manage Jenkins** > **Manage Credentials**
2. Click on **(global)** under **Stores scoped to Jenkins**
3. Click **Add Credentials**

4. Fill in the form:
   - **Kind**: Username with password
   - **Scope**: Global
   - **Username**: Your Docker Hub username
   - **Password**: Your Docker Hub access token (NOT your password!)
   - **ID**: `dockerhub-credentials` (must match Jenkinsfile)
   - **Description**: Docker Hub credentials

5. Click **Create**

[Screenshot: Adding Docker Hub credentials]

### Adding Slack Webhook (Optional)

1. In Slack, create an Incoming Webhook:
   - Go to `your-workspace.slack.com/apps`
   - Search for "Incoming WebHooks"
   - Add to channel and copy webhook URL

2. In Jenkins:
   - **Kind**: Secret text
   - **Scope**: Global
   - **Secret**: Paste the webhook URL
   - **ID**: `slack-webhook`
   - **Description**: Slack webhook for notifications

### Adding SSH Key (For Private Git Repos)

1. Generate SSH key (if needed):
   ```bash
   ssh-keygen -t ed25519 -C "jenkins@your-org.com"
   ```

2. In Jenkins:
   - **Kind**: SSH Username with private key
   - **Scope**: Global
   - **Username**: git
   - **Private Key**: Enter directly (paste key)
   - **ID**: `git-ssh-key`

---

## 4. Creating the Pipeline Job

### Step 1: Create New Item

1. From Jenkins dashboard, click **New Item**
2. Enter name: `predictive-maintenance-pipeline`
3. Select **Pipeline**
4. Click **OK**

[Screenshot: Creating new pipeline item]

### Step 2: Configure General Settings

1. **Description**: MLOps pipeline for Predictive Maintenance platform

2. **Discard old builds**:
   - Check this option
   - Max # of builds to keep: 10

3. **Do not allow concurrent builds**: Check this

4. **GitHub project** (if using GitHub):
   - Project url: `https://github.com/your-org/predictive-maintenance-mlops/`

### Step 3: Configure Build Triggers

Choose one or more:

- **Poll SCM**: Check repository for changes
  - Schedule: `H/5 * * * *` (every 5 minutes)

- **GitHub hook trigger for GITScm polling**: For webhook-based triggers

- **Build periodically**: For nightly builds
  - Schedule: `H 2 * * *` (2 AM daily)

### Step 4: Configure Pipeline

1. **Definition**: Pipeline script from SCM

2. **SCM**: Git

3. **Repository URL**:
   - For local: `file:///path/to/predictive-maintenance-mlops`
   - For GitHub: `https://github.com/your-org/predictive-maintenance-mlops.git`
   - For SSH: `git@github.com:your-org/predictive-maintenance-mlops.git`

4. **Credentials**: Select if needed (for private repos)

5. **Branches to build**: `*/main`

6. **Script Path**: `jenkins/Jenkinsfile`

7. Click **Save**

[Screenshot: Pipeline configuration]

---

## 5. Docker Hub Token Setup

### Why Use Access Tokens?

| Passwords | Access Tokens |
|-----------|---------------|
| Can't be revoked individually | Can be revoked anytime |
| Full account access | Configurable permissions |
| No audit trail | Detailed audit logs |
| Shared across all uses | Unique per application |
| Compromises whole account | Limited blast radius |

### Creating an Access Token

1. Log in to [hub.docker.com](https://hub.docker.com)
2. Click your profile icon > **Account Settings**
3. Click **Security** in the left sidebar
4. Click **New Access Token**

[Screenshot: Docker Hub security settings]

5. Configure the token:
   - **Token description**: `Jenkins CI/CD`
   - **Access permissions**:
     - Read: Required for pulling images
     - Write: Required for pushing images
     - Delete: Optional (for cleanup)

6. Click **Generate**
7. **IMPORTANT**: Copy the token immediately - it won't be shown again!

[Screenshot: Token generation dialog]

### Storing Token in Jenkins

1. Go to **Manage Jenkins** > **Manage Credentials**
2. Add credential as described in Section 3
3. Use the access token as the "Password" field

### Token Security Best Practices

1. **Never commit tokens to Git**
   ```bash
   # Add to .gitignore
   .env
   *.credentials
   ```

2. **Use separate tokens for each environment**
   - `jenkins-prod-token`
   - `jenkins-staging-token`
   - `developer-local-token`

3. **Rotate tokens every 90 days**
   - Set calendar reminders
   - Document token purposes

4. **Enable 2FA on Docker Hub**
   - Account Settings > Security > Two-Factor Authentication

5. **Monitor token usage**
   - Review Docker Hub audit logs monthly
   - Revoke unused tokens

---

## 6. First Pipeline Run

### Prerequisites Checklist

Before running the pipeline, ensure:

- [ ] Docker Desktop is installed and running
- [ ] Kubernetes is enabled in Docker Desktop
- [ ] At least 8GB RAM allocated to Docker
- [ ] At least 4 CPU cores allocated
- [ ] kubectl is installed and configured
- [ ] Helm is installed
- [ ] Docker Hub credentials are configured

### Running the Pipeline

1. Navigate to your pipeline job
2. Click **Build Now** in the left sidebar
3. Watch the build progress in **Stage View**

[Screenshot: Pipeline stage view]

### Understanding Console Output

Click on a build number, then **Console Output** to see:

```
[Pipeline] Start of Pipeline
[Pipeline] node

╔══════════════════════════════════════════════════════════════════════╗
║ CODE QUALITY & TESTING                                               ║
╚══════════════════════════════════════════════════════════════════════╝

[INFO] Setting up Python virtual environment...
[INFO] Running flake8 linting...
[INFO] Running pytest with coverage...
...
```

### Successful Deployment

After successful completion:

| Service | URL | Credentials |
|---------|-----|-------------|
| MLflow | http://localhost:5000 | None |
| Prometheus | http://localhost:9090 | None |
| Grafana | http://localhost:3000 | admin / admin |
| API Swagger | http://localhost:8000/docs | None |
| Kubeflow | http://localhost:8080 | None |

---

## 7. Automated Triggers

### Git Webhook (GitHub/GitLab/Bitbucket)

#### GitHub Setup

1. Go to repository **Settings** > **Webhooks**
2. Click **Add webhook**
3. Configure:
   - **Payload URL**: `http://jenkins-url:8080/github-webhook/`
   - **Content type**: application/json
   - **Secret**: (optional, for security)
   - **Events**: Just the push event

4. In Jenkins job, enable **GitHub hook trigger for GITScm polling**

#### For Local/Gitea

1. Configure webhook in Gitea:
   - URL: `http://jenkins:8080/gitea-webhook/post`
   - HTTP Method: POST

### Poll SCM

For environments without webhooks:

```
# Poll every 5 minutes
H/5 * * * *

# Poll every hour
H * * * *

# Poll twice daily
H 8,20 * * *
```

### Scheduled Builds

```
# Nightly at 2 AM
H 2 * * *

# Weekly on Sunday at midnight
H 0 * * 0

# First of every month
H 0 1 * *
```

---

## 8. Notifications Setup

### Slack Integration

1. Install **Slack Notification Plugin**

2. Configure in **Manage Jenkins** > **Configure System**:
   - Find "Slack" section
   - **Workspace**: Your Slack workspace
   - **Credential**: Add secret text with webhook URL
   - **Default channel**: #deployments

3. Test connection with **Test Connection** button

### Email Notifications

1. Install **Email Extension Plugin**

2. Configure in **Manage Jenkins** > **Configure System**:
   - Find "Extended E-mail Notification"
   - **SMTP server**: smtp.gmail.com
   - **Default user e-mail suffix**: @your-domain.com
   - **Use SSL**: checked
   - **SMTP Port**: 465
   - **Credentials**: Add username/password

3. For Gmail, use App Password:
   - Enable 2FA on Google account
   - Generate App Password at myaccount.google.com

---

## 9. Troubleshooting

### Common Issues

#### Docker Daemon Not Running

```
Error: Cannot connect to the Docker daemon
```

**Solution**:
```bash
# macOS/Windows: Start Docker Desktop

# Linux:
sudo systemctl start docker
sudo usermod -aG docker jenkins
sudo systemctl restart jenkins
```

#### Kubernetes Not Accessible

```
Error: Unable to connect to the server
```

**Solution**:
1. Open Docker Desktop Settings
2. Go to Kubernetes tab
3. Enable Kubernetes
4. Wait for green indicator
5. Verify: `kubectl cluster-info`

#### Insufficient Resources

```
Error: 0/1 nodes are available: insufficient memory
```

**Solution**:
- Docker Desktop Settings > Resources
- Set Memory: 8 GB minimum
- Set CPUs: 4 minimum
- Click Apply & Restart

#### Port Conflicts

```
Error: bind: address already in use
```

**Solution**:
```bash
# Find process using port
lsof -i :5000

# Kill the process
kill -9 <PID>

# Or use different ports in Jenkinsfile
```

#### Permission Denied

```
Error: permission denied while trying to connect to Docker
```

**Solution**:
```bash
# Add Jenkins user to docker group
sudo usermod -aG docker jenkins
sudo systemctl restart jenkins
```

### Viewing Build Logs

1. Click on build number
2. Click **Console Output**
3. Use **Pipeline Steps** for step-by-step view
4. Use **Blue Ocean** for visual debugging

### Restarting Failed Stages

With **Restart from Stage** feature:
1. Open failed build
2. Click **Restart from Stage**
3. Select the stage to restart from

### Cleaning Up

If you need to start fresh:

```bash
# Run cleanup script
./jenkins/cleanup.sh --all --force

# Or manually:
kubectl delete namespace mlops
docker-compose down -v
docker system prune -f
```

---

## Quick Reference

### Important URLs

| Resource | URL |
|----------|-----|
| Jenkins | http://localhost:8080 |
| Docker Hub | https://hub.docker.com |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| API Docs | http://localhost:8000/docs |

### Key Files

| File | Purpose |
|------|---------|
| `jenkins/Jenkinsfile` | Main pipeline definition |
| `jenkins/deploy-all.sh` | Deployment script |
| `jenkins/cleanup.sh` | Cleanup script |
| `jenkins/docker-credentials.sh` | Docker auth helper |

### Support

- Pipeline issues: Check Jenkins console output
- Kubernetes issues: `kubectl describe pod <pod-name> -n mlops`
- Docker issues: `docker logs <container-name>`
- Report bugs: [GitHub Issues](https://github.com/your-org/predictive-maintenance-mlops/issues)
