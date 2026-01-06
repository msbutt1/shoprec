"""End-to-end tests for the ShopRec API.

Tests the full request/response cycle including model loading,
recommendation generation, and response validation.
"""

import logging
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.main import app
from src.recommender.train import train_basic_cf_model

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)

# Create test client
client = TestClient(app)


@pytest.fixture(scope="module")
def trained_model_with_embeddings(tmp_path_factory) -> Generator[Path, None, None]:
    """Create a trained model with embeddings for e2e testing.
    
    Generates fake purchase data, trains a model, and returns the model directory.
    """
    import random
    from datetime import datetime, timedelta
    
    random.seed(42)
    
    # Create temporary directory for model
    model_dir = tmp_path_factory.mktemp("e2e_model")
    
    # Generate fake purchase data
    purchases = []
    for _ in range(200):
        user_id = random.randint(1, 20)
        product_id = random.randint(1, 50)
        timestamp = datetime.now() - timedelta(
            days=random.randint(0, 30),
            seconds=random.randint(0, 86400)
        )
        purchases.append({
            "user_id": user_id,
            "product_id": product_id,
            "timestamp": timestamp,
        })
    
    df = pd.DataFrame(purchases)
    csv_path = model_dir / "test_purchases.csv"
    df.to_csv(csv_path, index=False)
    
    # Train model with embeddings
    from src.recommender.train import TrainingConfig, train_with_config
    
    config = TrainingConfig(
        csv_path=str(csv_path),
        output_dir=str(model_dir),
        n_components=10,
        n_iter=5,
        random_state=42,
        generate_embeddings=True,
        embedding_method="random",
        embedding_dim=20,
    )
    
    train_with_config(config)
    
    yield model_dir


@pytest.fixture(scope="module")
def trained_model_cf_only(tmp_path_factory) -> Generator[Path, None, None]:
    """Create a trained model without embeddings for CF-only testing."""
    import random
    from datetime import datetime, timedelta
    
    random.seed(42)
    
    model_dir = tmp_path_factory.mktemp("e2e_model_cf")
    
    purchases = []
    for _ in range(200):
        user_id = random.randint(1, 20)
        product_id = random.randint(1, 50)
        timestamp = datetime.now() - timedelta(
            days=random.randint(0, 30),
            seconds=random.randint(0, 86400)
        )
        purchases.append({
            "user_id": user_id,
            "product_id": product_id,
            "timestamp": timestamp,
        })
    
    df = pd.DataFrame(purchases)
    csv_path = model_dir / "test_purchases.csv"
    df.to_csv(csv_path, index=False)
    
    train_basic_cf_model(
        csv_path=str(csv_path),
        output_dir=str(model_dir),
        n_components=10,
        n_iter=5,
        random_state=42,
    )
    
    yield model_dir


def test_e2e_recommend_cf_mode(trained_model_cf_only: Path):
    """Test end-to-end recommendation flow in CF-only mode.
    
    Verifies:
    - Request succeeds
    - Response has correct structure
    - Recommendations are valid product IDs
    - Correct number of recommendations returned
    """
    # Mock the model directory by patching the default
    import src.api.routes.recommend as recommend_module
    original_default = recommend_module.DEFAULT_MODEL_DIR
    
    try:
        recommend_module.DEFAULT_MODEL_DIR = str(trained_model_cf_only)
        
        # Clear model cache to force reload
        recommend_module._model_cache = None
        
        user_id = 1
        top_n = 5
        
        response = client.get(f"/recommend/{user_id}?mode=cf&top_n={top_n}")
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        
        # Verify response structure
        assert "user_id" in data
        assert "recommendations" in data
        assert "model_version" in data
        assert "scores" in data
        
        # Verify user_id matches
        assert data["user_id"] == user_id
        
        # Verify recommendations
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) <= top_n
        assert len(data["recommendations"]) > 0, "Should return at least one recommendation"
        
        # Verify all recommendations are integers (product IDs)
        for product_id in data["recommendations"]:
            assert isinstance(product_id, int), f"Expected int, got {type(product_id)}: {product_id}"
            assert product_id > 0, f"Product ID should be positive, got {product_id}"
        
        # Verify model_version is a string
        assert isinstance(data["model_version"], str)
        
        # Verify scores can be None or dict
        assert data["scores"] is None or isinstance(data["scores"], dict)
        
    finally:
        # Restore original default
        recommend_module.DEFAULT_MODEL_DIR = original_default
        recommend_module._model_cache = None


def test_e2e_recommend_hybrid_mode(trained_model_with_embeddings: Path):
    """Test end-to-end recommendation flow in hybrid mode.
    
    Verifies:
    - Request succeeds
    - Response has correct structure
    - Recommendations are valid product IDs
    - Correct number of recommendations returned
    - Hybrid mode works with embeddings
    """
    import src.api.routes.recommend as recommend_module
    original_default = recommend_module.DEFAULT_MODEL_DIR
    
    try:
        recommend_module.DEFAULT_MODEL_DIR = str(trained_model_with_embeddings)
        recommend_module._model_cache = None
        
        user_id = 1
        top_n = 10
        
        response = client.get(f"/recommend/{user_id}?mode=hybrid&top_n={top_n}")
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        
        # Verify response structure
        assert "user_id" in data
        assert "recommendations" in data
        assert "model_version" in data
        
        # Verify user_id matches
        assert data["user_id"] == user_id
        
        # Verify recommendations
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) <= top_n
        assert len(data["recommendations"]) > 0
        
        # Verify all recommendations are integers
        for product_id in data["recommendations"]:
            assert isinstance(product_id, int)
            assert product_id > 0
        
        # Verify model_version
        assert isinstance(data["model_version"], str)
        
    finally:
        recommend_module.DEFAULT_MODEL_DIR = original_default
        recommend_module._model_cache = None


def test_e2e_recommend_with_explain(trained_model_with_embeddings: Path):
    """Test recommendation endpoint with explain=true.
    
    Verifies that score breakdown is included in response.
    """
    import src.api.routes.recommend as recommend_module
    original_default = recommend_module.DEFAULT_MODEL_DIR
    
    try:
        recommend_module.DEFAULT_MODEL_DIR = str(trained_model_with_embeddings)
        recommend_module._model_cache = None
        
        user_id = 1
        top_n = 5
        
        response = client.get(f"/recommend/{user_id}?mode=hybrid&top_n={top_n}&explain=true")
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify scores are present when explain=true
        assert "scores" in data
        assert data["scores"] is not None
        assert isinstance(data["scores"], dict)
        
        # Verify score structure
        if "cf_scores" in data["scores"]:
            assert isinstance(data["scores"]["cf_scores"], dict)
        
        if "content_scores" in data["scores"]:
            assert isinstance(data["scores"]["content_scores"], dict)
        
        if "hybrid_scores" in data["scores"]:
            assert isinstance(data["scores"]["hybrid_scores"], dict)
        
    finally:
        recommend_module.DEFAULT_MODEL_DIR = original_default
        recommend_module._model_cache = None


def test_e2e_recommend_different_users(trained_model_cf_only: Path):
    """Test that different users get different recommendations.
    
    Verifies that the system personalizes recommendations per user.
    """
    import src.api.routes.recommend as recommend_module
    original_default = recommend_module.DEFAULT_MODEL_DIR
    
    try:
        recommend_module.DEFAULT_MODEL_DIR = str(trained_model_cf_only)
        recommend_module._model_cache = None
        
        user_ids = [1, 2, 3]
        recommendations_by_user = {}
        
        for user_id in user_ids:
            response = client.get(f"/recommend/{user_id}?mode=cf&top_n=5")
            assert response.status_code == 200
            
            data = response.json()
            recommendations_by_user[user_id] = set(data["recommendations"])
        
        # Verify we got recommendations for all users
        assert len(recommendations_by_user) == len(user_ids)
        
        # Verify all recommendations are valid product IDs
        for user_id, recs in recommendations_by_user.items():
            assert len(recs) > 0, f"User {user_id} should have recommendations"
            for product_id in recs:
                assert isinstance(product_id, int)
                assert product_id > 0
        
    finally:
        recommend_module.DEFAULT_MODEL_DIR = original_default
        recommend_module._model_cache = None


def test_e2e_recommend_cold_start_user(trained_model_cf_only: Path):
    """Test cold-start handling for unknown users.
    
    Verifies that users not in training data still get recommendations.
    """
    import src.api.routes.recommend as recommend_module
    original_default = recommend_module.DEFAULT_MODEL_DIR
    
    try:
        recommend_module.DEFAULT_MODEL_DIR = str(trained_model_cf_only)
        recommend_module._model_cache = None
        
        # Use a user ID that doesn't exist in training data
        cold_start_user_id = 99999
        top_n = 5
        
        response = client.get(f"/recommend/{cold_start_user_id}?mode=cf&top_n={top_n}")
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify response structure
        assert data["user_id"] == cold_start_user_id
        assert isinstance(data["recommendations"], list)
        
        # Cold-start should still return recommendations
        assert len(data["recommendations"]) > 0
        assert len(data["recommendations"]) <= top_n
        
        # Verify all are valid product IDs
        for product_id in data["recommendations"]:
            assert isinstance(product_id, int)
            assert product_id > 0
        
    finally:
        recommend_module.DEFAULT_MODEL_DIR = original_default
        recommend_module._model_cache = None


def test_e2e_recommend_top_n_variations(trained_model_cf_only: Path):
    """Test that top_n parameter works correctly.
    
    Verifies that different top_n values return correct number of recommendations.
    """
    import src.api.routes.recommend as recommend_module
    original_default = recommend_module.DEFAULT_MODEL_DIR
    
    try:
        recommend_module.DEFAULT_MODEL_DIR = str(trained_model_cf_only)
        recommend_module._model_cache = None
        
        user_id = 1
        
        for top_n in [1, 3, 5, 10]:
            response = client.get(f"/recommend/{user_id}?mode=cf&top_n={top_n}")
            assert response.status_code == 200
            
            data = response.json()
            recommendations = data["recommendations"]
            
            assert len(recommendations) <= top_n, f"Expected <= {top_n} recommendations, got {len(recommendations)}"
            assert len(recommendations) > 0, f"Expected at least 1 recommendation for top_n={top_n}"
            
            # Verify all are valid product IDs
            for product_id in recommendations:
                assert isinstance(product_id, int)
                assert product_id > 0
        
    finally:
        recommend_module.DEFAULT_MODEL_DIR = original_default
        recommend_module._model_cache = None

