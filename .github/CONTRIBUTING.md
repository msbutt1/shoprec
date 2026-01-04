# Contributing to ShopRec

Thanks for your interest in contributing to ShopRec! This document explains how to set up your development environment and contribute to the project.

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Git
- (Optional) Docker for testing containerization

### Fork and Clone
1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/YOUR_USERNAME/shoprec.git
cd shoprec
```

3. Add the upstream repository:
```bash
git remote add upstream https://github.com/ORIGINAL_OWNER/shoprec.git
```

### Set Up Development Environment

1. **Create a virtual environment**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Generate test data**
```bash
python scripts/generate_fake_data.py
```

4. **Train a model**
```bash
python scripts/train_model.py data/fake_purchases.csv
```

5. **Verify installation**
```bash
# Run tests
pytest tests/ -v

# Start the API server
python -m uvicorn src.api.main:app --reload
```

## Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Use prefixes:
- `feature/` for new features
- `fix/` for bug fixes
- `docs/` for documentation
- `test/` for test additions
- `refactor/` for code improvements

### 2. Make Changes
- Write clean, readable code
- Follow existing code style
- Add comments for complex logic
- Keep functions focused and small

### 3. Test Your Changes
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### 4. Commit Your Changes
Write clear commit messages:
```bash
git add .
git commit -m "feat: add new recommendation filter"
```

Commit message format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Code Style Guidelines

### Python Style
- Follow PEP 8
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use docstrings for functions and classes

Example:
```python
def recommend_products(
    user_id: int,
    top_n: int = 10,
) -> List[int]:
    """Get product recommendations for a user.
    
    Args:
        user_id: User ID to get recommendations for
        top_n: Number of recommendations to return
        
    Returns:
        List of recommended product IDs
    """
    # Implementation
    pass
```

### Imports
Group imports in this order:
1. Standard library
2. Third-party packages
3. Local modules

```python
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.recommender.utils import load_model
```

### Testing
- Write tests for new features
- Aim for good test coverage
- Use descriptive test names
- Test edge cases

Example:
```python
def test_recommend_products_returns_correct_count():
    """Test that recommend_products returns the requested number of items."""
    recommendations = recommend_products(user_id=1, top_n=5)
    assert len(recommendations) == 5
```

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Tests
```bash
# Run a specific test file
pytest tests/test_api.py -v

# Run a specific test function
pytest tests/test_api.py::test_ping_endpoint -v
```

### Run Tests with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html to view coverage report
```

## Testing the API

### Start the Development Server
```bash
python -m uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`

### Manual Testing
```bash
# Health check
curl http://localhost:8000/ping

# Get recommendations
curl http://localhost:8000/recommend/1?top_n=5

# Check model status
curl http://localhost:8000/status
```

### Interactive API Documentation
Visit `http://localhost:8000/docs` for Swagger UI

## Project Structure

```
shoprec/
├── src/
│   ├── api/              # FastAPI application
│   │   ├── main.py       # App entry point
│   │   ├── logging_config.py
│   │   └── routes/       # API endpoints
│   └── recommender/      # ML core
│       ├── train.py      # Model training
│       ├── infer.py      # Inference
│       ├── hybrid.py     # Hybrid recommendations
│       ├── embed.py      # Product embeddings
│       └── utils.py      # Utilities
├── scripts/              # CLI tools
├── tests/                # Unit tests
├── data/                 # Data files
└── models/               # Trained models
```

## Adding New Features

### Adding a New API Endpoint

1. Create a new route in `src/api/routes/`
2. Add the route to the router
3. Include the router in `src/api/main.py`
4. Write tests in `tests/test_api.py`
5. Update API documentation in README

### Adding a New ML Feature

1. Implement the feature in appropriate module (`src/recommender/`)
2. Add unit tests
3. Update training script if needed
4. Document the feature

## Common Issues

### Import Errors
Make sure you're in the project root and the virtual environment is activated:
```bash
cd shoprec
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### Model Not Found
Train a model before running the API:
```bash
python scripts/train_model.py data/fake_purchases.csv
```

### Tests Failing
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Documentation

Update documentation when:
- Adding new features
- Changing API endpoints
- Modifying configuration options
- Fixing bugs that affect usage

Update:
- `README.md` for user-facing changes
- Docstrings for code changes
- This file for development process changes

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues and pull requests first
- Be respectful and constructive

## Code Review Process

Pull requests are reviewed for:
- Code quality and style
- Test coverage
- Documentation
- Performance impact
- Breaking changes

Reviews typically take 1-3 days. Be patient and responsive to feedback.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to ShopRec!

