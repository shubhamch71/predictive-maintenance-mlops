# Docker Hub Authentication Setup Guide

This guide provides comprehensive instructions for setting up Docker Hub authentication for the Predictive Maintenance MLOps platform, with emphasis on security best practices.

## Table of Contents

1. [Why Use Access Tokens](#1-why-use-access-tokens)
2. [Creating an Access Token](#2-creating-an-access-token)
3. [Using Token in Jenkins](#3-using-token-in-jenkins)
4. [Using Token in Terminal](#4-using-token-in-terminal)
5. [Security Best Practices](#5-security-best-practices)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Why Use Access Tokens?

### Security Benefits

| Feature | Password | Access Token |
|---------|----------|--------------|
| Revocation | Revokes all access | Individual token only |
| Permissions | Full account | Configurable (Read/Write/Delete) |
| Audit Trail | Limited | Detailed logs per token |
| Rotation | Disruptive | Non-disruptive |
| Sharing Risk | High (full access) | Low (scoped access) |
| 2FA Compatible | No | Yes |

### Key Advantages

1. **Fine-grained Access Control**
   - Create tokens with only necessary permissions
   - Read-only tokens for pulling images
   - Write tokens for CI/CD systems

2. **Easy Revocation**
   - Revoke compromised tokens without affecting other integrations
   - No password change required

3. **Audit Capabilities**
   - Track which token performed which action
   - Identify unauthorized access quickly

4. **Compliance**
   - Meet security requirements for SOC2, ISO27001
   - Demonstrate least-privilege access

---

## 2. Creating an Access Token

### Step-by-Step Instructions

#### Step 1: Log in to Docker Hub

1. Navigate to [hub.docker.com](https://hub.docker.com)
2. Log in with your Docker Hub account
3. If 2FA is enabled, complete verification

[Screenshot placeholder: Docker Hub login page]

#### Step 2: Navigate to Security Settings

1. Click your profile avatar in the top right
2. Select **Account Settings**
3. Click **Security** in the left sidebar

[Screenshot placeholder: Docker Hub account settings]

#### Step 3: Generate New Token

1. In the Security section, find **Access Tokens**
2. Click **New Access Token** button

[Screenshot placeholder: Access tokens section]

#### Step 4: Configure Token

Fill in the token details:

| Field | Value | Description |
|-------|-------|-------------|
| **Token description** | `Jenkins CI/CD Production` | Identifies this token's purpose |
| **Access permissions** | Read, Write, Delete | Choose based on needs |

**Permission Levels:**

- **Read**: Pull images only (for deployments)
- **Write**: Push images (for CI/CD builds)
- **Delete**: Remove images (for cleanup automation)

[Screenshot placeholder: Token configuration dialog]

#### Step 5: Generate and Copy

1. Click **Generate**
2. **IMMEDIATELY copy the token** - it will only be shown once!
3. Store it securely (password manager, secret vault)
4. Click **Copy and Close**

```
Token format example: dckr_pat_xxxxxxxxxxxxxxxxxxxxxxxxx
```

[Screenshot placeholder: Token generated dialog with copy button]

#### Step 6: Verify Token in List

After creation, verify your token appears in the list:

| Description | UUID | Created | Last Used | Permissions |
|-------------|------|---------|-----------|-------------|
| Jenkins CI/CD Production | abc123... | Jan 15, 2024 | Never | Read, Write, Delete |

---

## 3. Using Token in Jenkins

### Method 1: Username with Password Credential

1. Navigate to **Manage Jenkins** > **Manage Credentials**
2. Click **(global)** under the appropriate domain
3. Click **Add Credentials**

4. Configure:
   ```
   Kind: Username with password
   Scope: Global
   Username: your-docker-hub-username
   Password: dckr_pat_xxxxxxxxxxxxxxxxxxxxxxxxx  (paste token here)
   ID: dockerhub-credentials
   Description: Docker Hub Access Token for CI/CD
   ```

5. Click **Create**

[Screenshot placeholder: Jenkins credential configuration]

### Method 2: Using in Jenkinsfile

The Jenkinsfile uses the credential like this:

```groovy
stage('Push to Docker Registry') {
    steps {
        withCredentials([usernamePassword(
            credentialsId: 'dockerhub-credentials',
            usernameVariable: 'DOCKER_USERNAME',
            passwordVariable: 'DOCKER_PASSWORD'
        )]) {
            sh '''
                set +x  # Disable command echoing for security
                echo "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin
                set -x

                docker push ${IMAGE_NAME}:${TAG}

                docker logout
            '''
        }
    }
}
```

### Method 3: Using the Helper Script

```bash
# Set credentials in environment (done by Jenkins)
export DOCKER_USERNAME="your-username"
export DOCKER_PASSWORD="dckr_pat_xxxxxxxxxxxxxxxxxxxxxxxxx"

# Use the helper script
./jenkins/docker-credentials.sh login
```

---

## 4. Using Token in Terminal

### Basic Login

```bash
# Login with token (secure method)
echo "dckr_pat_xxxxxxxxxxxxxxxxxxxxxxxxx" | docker login -u your-username --password-stdin

# Verify login
docker info | grep Username

# Test by pulling an image
docker pull hello-world

# Logout when done
docker logout
```

### Using Environment Variables

```bash
# Set environment variables
export DOCKER_USERNAME="your-username"
export DOCKER_TOKEN="dckr_pat_xxxxxxxxxxxxxxxxxxxxxxxxx"

# Login
echo "$DOCKER_TOKEN" | docker login -u "$DOCKER_USERNAME" --password-stdin
```

### Using .env File (Local Development)

Create a `.env` file (add to .gitignore!):

```bash
# .env file - DO NOT COMMIT TO GIT
DOCKER_USERNAME=your-username
DOCKER_PASSWORD=dckr_pat_xxxxxxxxxxxxxxxxxxxxxxxxx
```

Load and use:

```bash
# Load environment
source .env

# Login
./jenkins/docker-credentials.sh login
```

### Docker Credential Store (Recommended for Local)

For secure local storage, use Docker's credential helpers:

```bash
# macOS (uses Keychain)
brew install docker-credential-helper

# Configure Docker to use it
cat > ~/.docker/config.json << 'EOF'
{
    "credsStore": "osxkeychain"
}
EOF

# Login normally - credentials stored securely
docker login -u your-username
# Enter token when prompted
```

---

## 5. Security Best Practices

### Token Management

#### 1. Never Commit Credentials to Git

```bash
# Add to .gitignore
echo ".env" >> .gitignore
echo "*.credentials" >> .gitignore
echo ".docker/config.json" >> .gitignore

# Check for accidentally committed secrets
git log --all --full-history -- "*.env"
```

#### 2. Use Separate Tokens for Each Environment

| Environment | Token Description | Permissions |
|-------------|-------------------|-------------|
| Production | `prod-jenkins-ci` | Read, Write |
| Staging | `staging-jenkins-ci` | Read, Write |
| Development | `dev-local` | Read only |
| Personal | `local-dev-john` | Read, Write |

#### 3. Rotate Tokens Regularly

**Recommended Schedule:**
- Every 90 days for CI/CD tokens
- Every 180 days for development tokens
- Immediately if any exposure suspected

**Rotation Process:**
1. Generate new token with same permissions
2. Update Jenkins/CI credentials
3. Test that new token works
4. Revoke old token

#### 4. Monitor Token Usage

Check Docker Hub audit logs regularly:

1. Go to Docker Hub > Account Settings > Security
2. Review "Access Token Activity" section
3. Look for:
   - Unexpected usage times
   - Unknown IP addresses
   - Unusual operations

#### 5. Enable Two-Factor Authentication

1. Go to Account Settings > Security
2. Enable Two-Factor Authentication
3. Use authenticator app (not SMS)
4. Save backup codes securely

### Infrastructure Security

#### Use Secret Management

Instead of environment variables, consider:

**HashiCorp Vault:**
```bash
# Store secret
vault kv put secret/docker-hub token="dckr_pat_xxx"

# Retrieve in pipeline
export DOCKER_PASSWORD=$(vault kv get -field=token secret/docker-hub)
```

**AWS Secrets Manager:**
```bash
# Store secret
aws secretsmanager create-secret \
    --name docker-hub-token \
    --secret-string "dckr_pat_xxx"

# Retrieve
export DOCKER_PASSWORD=$(aws secretsmanager get-secret-value \
    --secret-id docker-hub-token \
    --query SecretString --output text)
```

**Kubernetes Secrets:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: docker-hub-credentials
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: <base64-encoded-docker-config>
```

### Network Security

1. **Limit IP Access** (Docker Hub Teams/Business)
   - Whitelist CI/CD server IPs
   - Block access from unknown locations

2. **Use Private Registry** for sensitive images
   - AWS ECR
   - Google Container Registry
   - Self-hosted Harbor

---

## 6. Troubleshooting

### Common Errors

#### Error: "unauthorized: incorrect username or password"

**Possible Causes:**
1. Token expired or revoked
2. Token copied incorrectly (extra spaces)
3. Username doesn't match token owner
4. 2FA issues

**Solutions:**
```bash
# Verify username
echo $DOCKER_USERNAME

# Check token format (should start with dckr_pat_)
echo $DOCKER_PASSWORD | head -c 10

# Try logging in manually
docker login -u your-username
# Paste token when prompted
```

#### Error: "denied: requested access to the resource is denied"

**Possible Causes:**
1. Token doesn't have Write permission
2. Repository doesn't exist
3. Private repository access issue

**Solutions:**
1. Check token permissions in Docker Hub
2. Create repository first: `docker push` won't create it
3. Verify repository ownership

#### Error: "Error response from daemon: Get https://registry-1.docker.io/v2/: net/http: request canceled"

**Possible Causes:**
1. Network connectivity issues
2. Proxy configuration problems
3. DNS issues

**Solutions:**
```bash
# Test connectivity
curl -v https://registry-1.docker.io/v2/

# Check Docker proxy settings
docker info | grep -i proxy

# Configure proxy if needed
cat > ~/.docker/config.json << 'EOF'
{
    "proxies": {
        "default": {
            "httpProxy": "http://proxy:8080",
            "httpsProxy": "http://proxy:8080",
            "noProxy": "localhost,127.0.0.1"
        }
    }
}
EOF
```

#### Error: "token has been revoked"

**Solution:**
1. Generate new token in Docker Hub
2. Update credentials in Jenkins
3. Re-run the pipeline

### Debugging Steps

1. **Verify Token Validity:**
   ```bash
   # Attempt login and check response
   echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin 2>&1
   ```

2. **Check Docker Configuration:**
   ```bash
   # View Docker config (be careful with secrets!)
   cat ~/.docker/config.json | jq 'del(.auths[].auth)'
   ```

3. **Test API Directly:**
   ```bash
   # Get token from Hub
   TOKEN=$(curl -s -H "Content-Type: application/json" \
     -X POST -d '{"username": "'$DOCKER_USERNAME'", "password": "'$DOCKER_PASSWORD'"}' \
     https://hub.docker.com/v2/users/login/ | jq -r .token)

   # Test token
   curl -H "Authorization: JWT ${TOKEN}" \
     https://hub.docker.com/v2/repositories/$DOCKER_USERNAME/
   ```

4. **Verify Jenkins Credentials:**
   - Go to Jenkins > Manage Credentials
   - Click on the credential
   - Click "Update" and verify values

### Getting Help

- **Docker Documentation**: [docs.docker.com](https://docs.docker.com)
- **Docker Hub Support**: [hub.docker.com/support](https://hub.docker.com/support)
- **Project Issues**: [GitHub Issues](https://github.com/your-org/predictive-maintenance-mlops/issues)

---

## Quick Reference

### Token Commands

```bash
# Login with token
echo "$TOKEN" | docker login -u "$USER" --password-stdin

# Verify authentication
docker info | grep Username

# Logout
docker logout
```

### Jenkins Credential ID

```
dockerhub-credentials
```

### Token Format

```
dckr_pat_xxxxxxxxxxxxxxxxxxxxxxxxx
```

### Recommended Permissions

| Use Case | Read | Write | Delete |
|----------|------|-------|--------|
| CI/CD Pipeline | Yes | Yes | Optional |
| Deployment Only | Yes | No | No |
| Full Automation | Yes | Yes | Yes |
| Development | Yes | Yes | No |
