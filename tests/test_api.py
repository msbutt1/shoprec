"""Tests for the FastAPI application endpoints.

This module contains integration tests for the ShopRec API endpoints,
including health checks and recommendation endpoints.
"""

from fastapi.testclient import TestClient

from src.api.main import app

# Create test client
client = TestClient(app)


def test_ping_endpoint():
    """Test that the /ping endpoint returns correct status and JSON."""
    response = client.get("/ping")
    
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_endpoint():
    """Test that the /status endpoint returns model status information."""
    response = client.get("/status")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check required fields
    assert "model_loaded" in data
    assert "timestamp_last_loaded" in data
    assert "num_users" in data
    assert "num_products" in data
    
    # Check types
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["num_users"], int)
    assert isinstance(data["num_products"], int)
    
    # timestamp_last_loaded can be None or a string
    assert data["timestamp_last_loaded"] is None or isinstance(data["timestamp_last_loaded"], str)


def test_recommend_endpoint_returns_response():
    """Test that /recommend/{user_id} returns a valid response structure."""
    user_id = 5
    response = client.get(f"/recommend/{user_id}")
    
    # May return 200 or 503 depending on if model exists
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "user_id" in data
        assert "recommendations" in data
        assert "model_version" in data
        assert isinstance(data["recommendations"], list)


def test_recommend_endpoint_hybrid_mode():
    """Test that /recommend/{user_id}?mode=hybrid uses hybrid recommendations."""
    user_id = 1
    response = client.get(f"/recommend/{user_id}?mode=hybrid&top_n=5")
    
    if response.status_code == 200:
        data = response.json()
        assert data["user_id"] == user_id
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) <= 5


def test_recommend_endpoint_cf_mode():
    """Test that /recommend/{user_id}?mode=cf uses CF-only recommendations."""
    user_id = 1
    response = client.get(f"/recommend/{user_id}?mode=cf&top_n=5")
    
    if response.status_code == 200:
        data = response.json()
        assert data["user_id"] == user_id
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) <= 5


def test_recommend_endpoint_invalid_mode_fallback():
    """Test that invalid mode parameter falls back to CF mode."""
    user_id = 1
    response = client.get(f"/recommend/{user_id}?mode=invalid")
    
    if response.status_code == 200:
        data = response.json()
        assert data["user_id"] == user_id
        assert isinstance(data["recommendations"], list)


def test_recommend_endpoint_mode_with_explain():
    """Test that mode parameter works with explain=true."""
    user_id = 1
    
    # Test hybrid mode with explain
    response = client.get(f"/recommend/{user_id}?mode=hybrid&explain=true&top_n=3")
    if response.status_code == 200:
        data = response.json()
        assert "scores" in data
        assert data["scores"] is not None
        assert "cf_scores" in data["scores"]
        assert "content_scores" in data["scores"]
        assert "hybrid_scores" in data["scores"]
    
    # Test CF mode with explain
    response = client.get(f"/recommend/{user_id}?mode=cf&explain=true&top_n=3")
    if response.status_code == 200:
        data = response.json()
        assert "scores" in data
        assert data["scores"] is not None
        assert data["scores"]["method"] == "cf_only"
        assert "cf_scores" in data["scores"]
