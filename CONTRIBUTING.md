# Contributing to Predictive Maintenance MLOps Platform

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Code Style Guidelines](#code-style-guidelines)
5. [Testing Requirements](#testing-requirements)
6. [Pull Request Process](#pull-request-process)
7. [Commit Message Convention](#commit-message-convention)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Our Standards

**Positive behaviors include:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what's best for the community
- Showing empathy towards others

**Unacceptable behaviors include:**
- Harassment of any kind
- Trolling, insulting, or derogatory comments
- Personal or political attacks
- Publishing others' private information without permission
- Any conduct inappropriate in a professional setting

### Enforcement

Instances of unacceptable behavior may be reported to the project maintainers. All complaints will be reviewed and investigated and will result in a response that is deemed necessary and appropriate.

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose
- Git
- kubectl (for Kubernetes deployment)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/predictive-maintenance-mlops.git
   cd predictive-maintenance-mlops
   ```

3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/predictive-maintenance-mlops.git
   ```

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Copy environment configuration
cp .env.example .env
# Edit .env with your settings

# Run tests to verify setup
pytest tests/ -v
```

Or use the setup script:

```bash
./scripts/setup-dev.sh
```

---

## Development Workflow

### 1. Create a Branch

Always create a feature branch from `main`:

```bash
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/fixes

### 2. Make Changes

- Write clean, readable code
- Follow the code style guidelines
- Add tests for new functionality
- Update documentation as needed

### 3. Run Quality Checks

Before committing, run all quality checks:

```bash
# Format code
black src/ tests/ --line-length 100

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Security scan
bandit -r src/ -ll

# Type checking (optional)
mypy src/

# Run tests
pytest tests/ --cov=src --cov-report=html
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add new prediction endpoint"
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

---

## Code Style Guidelines

### Python Style

We follow PEP 8 with these specifications:

| Aspect | Requirement |
|--------|-------------|
| Formatter | black |
| Line length | 100 characters |
| Import sorting | isort (black profile) |
| Docstrings | Google style |
| Type hints | Required for all functions |

### Example Function

```python
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def calculate_rul(
    sensor_data: pd.DataFrame,
    model_type: str = "ensemble",
    confidence_threshold: float = 0.8,
) -> Dict[str, float]:
    """
    Calculate Remaining Useful Life from sensor data.

    This function processes sensor readings and predicts the remaining
    operational cycles before maintenance is required.

    Args:
        sensor_data: DataFrame containing sensor readings with columns
            matching the expected feature schema.
        model_type: Type of model to use for prediction.
            Options: "ensemble", "xgboost", "random_forest", "lstm".
        confidence_threshold: Minimum confidence level for predictions.
            Predictions below this threshold will be flagged.

    Returns:
        Dictionary containing:
            - predicted_rul: Predicted remaining useful life in cycles
            - confidence: Model confidence score (0-1)
            - model_type: Model used for prediction

    Raises:
        ValueError: If sensor_data is empty or missing required columns.
        ModelNotFoundError: If specified model_type is not available.

    Example:
        >>> data = pd.DataFrame({"sensor_2": [0.5], "sensor_3": [0.7]})
        >>> result = calculate_rul(data, model_type="ensemble")
        >>> print(f"RUL: {result['predicted_rul']:.1f} cycles")
        RUL: 45.2 cycles
    """
    if sensor_data.empty:
        raise ValueError("sensor_data cannot be empty")

    # Implementation here...
    return {
        "predicted_rul": 45.2,
        "confidence": 0.92,
        "model_type": model_type,
    }
```

### Key Guidelines

1. **Functions**
   - Keep under 50 lines
   - Single responsibility
   - Maximum cyclomatic complexity: 10
   - Always include type hints

2. **Variables**
   - Use descriptive names
   - Avoid single-letter names (except loop counters)
   - Use snake_case for variables and functions
   - Use PascalCase for classes

3. **Imports**
   - Group: standard library, third-party, local
   - Sort alphabetically within groups
   - Avoid wildcard imports (`from x import *`)

4. **Comments**
   - Prefer self-documenting code
   - Use comments to explain "why", not "what"
   - Keep comments up-to-date

---

## Testing Requirements

### Coverage Requirements

- Minimum overall coverage: **80%**
- New code must have tests
- Critical paths (API endpoints, model inference) should have >90% coverage

### Test Types

1. **Unit Tests** - Test individual functions/classes
2. **Integration Tests** - Test component interactions
3. **API Tests** - Test HTTP endpoints
4. **Load Tests** - Test performance under load

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run tests matching pattern
pytest tests/ -k "test_predict"

# Run with parallel execution
pytest tests/ -n auto
```

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

from src.models.ensemble import EnsembleModel


class TestEnsembleModel:
    """Tests for the Ensemble model."""

    @pytest.fixture
    def model(self):
        """Create a model instance for testing."""
        return EnsembleModel()

    @pytest.fixture
    def sample_features(self):
        """Create sample input features."""
        return {
            "sensor_2": 0.5,
            "sensor_3": 0.7,
            "sensor_4": 0.3,
        }

    def test_predict_returns_valid_rul(self, model, sample_features):
        """Test that predict returns a valid RUL value."""
        result = model.predict(sample_features)

        assert "predicted_rul" in result
        assert 0 <= result["predicted_rul"] <= 200
        assert "confidence" in result

    def test_predict_handles_missing_features(self, model):
        """Test that predict raises error for missing features."""
        with pytest.raises(ValueError, match="Missing required features"):
            model.predict({})

    @patch("src.models.ensemble.mlflow.load_model")
    def test_loads_model_from_mlflow(self, mock_load, model):
        """Test that model is loaded from MLflow registry."""
        mock_load.return_value = Mock()

        model.load_model("production")

        mock_load.assert_called_once()
```

---

## Pull Request Process

### Before Submitting

1. [ ] All tests pass locally
2. [ ] Code follows style guidelines
3. [ ] New code has tests (coverage >= 80%)
4. [ ] Documentation updated if needed
5. [ ] Changelog updated if applicable
6. [ ] Commits follow convention
7. [ ] Branch is up-to-date with main

### PR Template

When creating a PR, include:

```markdown
## Summary
Brief description of changes (2-3 sentences)

## Changes Made
- Change 1
- Change 2
- Change 3

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

### Review Process

1. Automated checks must pass (CI/CD)
2. At least 2 approving reviews required
3. Address all review feedback
4. Squash commits before merge (if many small commits)

---

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/).

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, missing semicolons |
| `refactor` | Code change that neither fixes nor adds |
| `perf` | Performance improvement |
| `test` | Adding or fixing tests |
| `chore` | Maintenance tasks |
| `ci` | CI/CD changes |
| `build` | Build system changes |

### Examples

```bash
# Feature
git commit -m "feat(api): add batch prediction endpoint"

# Bug fix
git commit -m "fix(model): correct RMSE calculation in evaluation"

# Documentation
git commit -m "docs: update API usage examples"

# Refactor
git commit -m "refactor(training): simplify feature engineering pipeline"

# Breaking change
git commit -m "feat(api)!: change prediction response format

BREAKING CHANGE: The prediction response now includes
model_version field and removes legacy_score field."
```

---

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue with reproduction steps
- **Security**: Email security concerns privately

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
