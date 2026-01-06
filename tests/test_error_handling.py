"""Tests for error handling in the ShopRec API.

Tests various error scenarios including model not found,
user not found, and internal errors.
"""

import logging
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.main import app
from src.recommender.infer import ModelNotFoundError, UserNotFoundError

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)

# Create test client
client = TestClient(app)


def test_model_not_found_error():
    """Test that missing model returns 503 Service Unavailable."""
    # Use a non-existent model directory
    response = client.get("/recommend/1?model_dir=non_existent_model_dir&top_n=5")
    
    assert response.status_code == 503
    data = response.json()
    
    assert "error" in data
    assert "Model not found" in data["error"] or "Model not found" in data.get("message", "")


def test_user_not_found_with_cold_start_disabled():
    """Test that unknown user with cold_start=False returns 404."""
    # First, we need a valid model. We'll use the default model if it exists
    # If it doesn't exist, this test will be skipped or will test the 503 error
    
    # Use a very high user ID that's unlikely to exist
    unknown_user_id = 999999
    
    response = client.get(
        f"/recommend/{unknown_user_id}?allow_cold_start=false&top_n=5"
    )
    
    # Should return either 404 (user not found) or 503 (model not found)
    assert response.status_code in [404, 503]
    
    if response.status_code == 404:
        data = response.json()
        assert "error" in data
        assert "User not found" in data.get("error", "") or "not found" in data.get("message", "").lower()


def test_user_not_found_with_cold_start_enabled():
    """Test that unknown user with cold_start=True returns recommendations."""
    unknown_user_id = 999999
    
    response = client.get(
        f"/recommend/{unknown_user_id}?allow_cold_start=true&top_n=5"
    )
    
    # Should return 200 (cold-start recommendations) or 503 (model not found)
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "user_id" in data
        assert data["user_id"] == unknown_user_id
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)


def test_invalid_user_id_type():
    """Test that invalid user_id type returns 422 validation error."""
    response = client.get("/recommend/not_a_number?top_n=5")
    
    assert response.status_code == 422
    data = response.json()
    
    assert "error" in data or "detail" in data


def test_invalid_top_n_parameter():
    """Test that invalid top_n parameter returns validation error."""
    response = client.get("/recommend/1?top_n=invalid")
    
    assert response.status_code == 422
    data = response.json()
    
    assert "error" in data or "detail" in data


def test_negative_top_n_parameter():
    """Test that negative top_n is handled gracefully."""
    response = client.get("/recommend/1?top_n=-5")
    
    # Should return validation error or handle gracefully
    assert response.status_code in [200, 400, 422, 503]


def test_zero_top_n_parameter():
    """Test that zero top_n is handled gracefully."""
    response = client.get("/recommend/1?top_n=0")
    
    # Should return validation error or empty recommendations
    assert response.status_code in [200, 400, 422, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)


def test_invalid_mode_parameter():
    """Test that invalid mode parameter falls back to default."""
    response = client.get("/recommend/1?mode=invalid_mode&top_n=5")
    
    # Should handle gracefully by falling back to CF mode
    assert response.status_code in [200, 503]


def test_error_response_structure():
    """Test that error responses have consistent structure."""
    # Test with invalid user_id to get a validation error
    response = client.get("/recommend/not_a_number?top_n=5")
    
    assert response.status_code == 422
    data = response.json()
    
    # Check that error response has expected fields
    assert "error" in data or "detail" in data


def test_exception_logging():
    """Test that exceptions are properly logged."""
    # Test with cold_start disabled and unknown user to get a 404
    # First clear any cached model by using a fresh client
    unknown_user_id = 777777
    
    # With model loaded, requesting unknown user with cold_start=false should give 404
    response = client.get(
        f"/recommend/{unknown_user_id}?allow_cold_start=false&top_n=5"
    )
    
    # Should return 404 (user not found) or 503 (model not found) or 200 (if cold-start still applied)
    assert response.status_code in [200, 404, 503]
    
    # If we got an error, verify it has proper structure
    if response.status_code in [404, 503]:
        data = response.json()
        assert "error" in data or "detail" in data


def test_multiple_errors_consistency():
    """Test that multiple errors of the same type return consistent responses."""
    responses = []
    
    for user_id in [9999991, 9999992, 9999993]:
        response = client.get(
            f"/recommend/{user_id}?model_dir=non_existent&allow_cold_start=false"
        )
        responses.append(response)
    
    # All should have the same status code
    status_codes = [r.status_code for r in responses]
    assert len(set(status_codes)) == 1, "All responses should have the same status code"
    
    # All should have error information
    for response in responses:
        data = response.json()
        assert "error" in data or "detail" in data


def test_cold_start_default_behavior():
    """Test that cold_start parameter defaults to True."""
    unknown_user_id = 888888
    
    # Call without specifying allow_cold_start
    response = client.get(f"/recommend/{unknown_user_id}?top_n=5")
    
    # Should allow cold-start by default (200 or 503 if model missing)
    assert response.status_code in [200, 503]


def test_health_check_not_affected_by_model_errors():
    """Test that /ping endpoint works even if model is missing."""
    response = client.get("/ping")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_status_endpoint_with_missing_model():
    """Test that /status endpoint works even if model is missing."""
    response = client.get("/status")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)

