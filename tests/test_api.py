"""Tests for the FastAPI application endpoints.

This module contains integration tests for the ShopRec API endpoints,
including health checks and recommendation endpoints.
"""

from typing import Dict

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

# Create test client
client = TestClient(app)


def test_ping_endpoint() -> None:
    """Test that the /ping endpoint returns correct status and JSON.

    Verifies:
        - Status code is 200
        - Response contains {"status": "ok"}
    """
    response = client.get("/ping")

    # Assert status code
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    # Assert response JSON
    data = response.json()
    assert isinstance(data, dict), "Response should be a dictionary"
    assert "status" in data, "Response should contain 'status' key"
    assert data["status"] == "ok", f"Expected 'ok', got '{data['status']}'"


def test_ping_endpoint_response_structure() -> None:
    """Test that /ping endpoint returns the exact expected structure.

    Verifies the response matches the expected format exactly.
    """
    response = client.get("/ping")
    assert response.status_code == 200

    expected_response = {"status": "ok"}
    assert response.json() == expected_response, (
        f"Response does not match expected format. "
        f"Got: {response.json()}, Expected: {expected_response}"
    )


def test_recommend_endpoint_returns_dummy_products() -> None:
    """Test that /recommend/{user_id} returns a list of product IDs.

    Verifies:
        - Status code is 200
        - Response contains user_id and recommendations list
        - Recommendations is a list of integers
    """
    user_id = 5
    response = client.get(f"/recommend/{user_id}")

    # Assert status code
    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}. "
        f"Response: {response.text}"
    )

    # Assert response structure
    data = response.json()
    assert isinstance(data, dict), "Response should be a dictionary"

    # Assert required fields
    assert "user_id" in data, "Response should contain 'user_id'"
    assert "recommendations" in data, "Response should contain 'recommendations'"
    assert "model_version" in data, "Response should contain 'model_version'"

    # Assert user_id matches
    assert data["user_id"] == user_id, (
        f"Expected user_id {user_id}, got {data['user_id']}"
    )

    # Assert recommendations is a list
    recommendations = data["recommendations"]
    assert isinstance(recommendations, list), (
        f"Recommendations should be a list, got {type(recommendations)}"
    )

    # Assert recommendations contains integers
    assert len(recommendations) > 0, "Recommendations list should not be empty"
    assert all(
        isinstance(rec, int) for rec in recommendations
    ), "All recommendations should be integers"


def test_recommend_endpoint_with_top_n_parameter() -> None:
    """Test that /recommend/{user_id} respects the top_n query parameter.

    Verifies:
        - top_n parameter controls number of recommendations
        - Returns correct number of recommendations
    """
    user_id = 10
    top_n = 5

    response = client.get(f"/recommend/{user_id}?top_n={top_n}")

    assert response.status_code == 200
    data = response.json()

    recommendations = data["recommendations"]
    assert len(recommendations) == top_n, (
        f"Expected {top_n} recommendations, got {len(recommendations)}"
    )


def test_recommend_endpoint_different_users() -> None:
    """Test that /recommend/{user_id} works for different user IDs.

    Verifies:
        - Endpoint accepts different user IDs
        - Returns valid responses for each
    """
    user_ids = [1, 5, 10, 42, 99]

    for user_id in user_ids:
        response = client.get(f"/recommend/{user_id}")

        assert response.status_code == 200, (
            f"Failed for user_id {user_id}: {response.status_code}"
        )

        data = response.json()
        assert data["user_id"] == user_id
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) > 0


def test_recommend_endpoint_response_model() -> None:
    """Test that /recommend/{user_id} response matches Pydantic model.

    Verifies the response structure matches the RecommendationResponse model.
    """
    user_id = 7
    response = client.get(f"/recommend/{user_id}")

    assert response.status_code == 200
    data = response.json()

    # Verify all required fields and types
    assert isinstance(data["user_id"], int)
    assert isinstance(data["recommendations"], list)
    assert isinstance(data["model_version"], str)

    # Verify recommendations are integers
    assert all(isinstance(r, int) for r in data["recommendations"])


def test_recommend_endpoint_default_top_n() -> None:
    """Test that /recommend/{user_id} uses default top_n=10 when not specified.

    Verifies the default behavior when top_n is not provided.
    """
    user_id = 3
    response = client.get(f"/recommend/{user_id}")

    assert response.status_code == 200
    data = response.json()

    # Default should be 10
    assert len(data["recommendations"]) == 10, (
        f"Expected default of 10 recommendations, got {len(data['recommendations'])}"
    )


def test_recommend_endpoint_with_large_top_n() -> None:
    """Test that /recommend/{user_id} handles large top_n values.

    Verifies the endpoint handles edge cases gracefully.
    """
    user_id = 1
    top_n = 100

    response = client.get(f"/recommend/{user_id}?top_n={top_n}")

    assert response.status_code == 200
    data = response.json()

    # Should return recommendations (may be limited by available products)
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) > 0


def test_recommend_endpoint_negative_user_id() -> None:
    """Test that /recommend/{user_id} handles negative user IDs.

    Verifies error handling for invalid user IDs.
    """
    user_id = -1
    response = client.get(f"/recommend/{user_id}")

    # Should still return 200 (handled by endpoint logic)
    # or return appropriate error status
    assert response.status_code in [200, 400, 422], (
        f"Unexpected status code: {response.status_code}"
    )


def test_api_docs_endpoint() -> None:
    """Test that FastAPI auto-generated docs endpoint is accessible.

    Verifies the /docs endpoint is available.
    """
    response = client.get("/docs")

    # Should return HTML documentation page
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_openapi_schema_endpoint() -> None:
    """Test that OpenAPI schema endpoint is accessible.

    Verifies the /openapi.json endpoint returns valid JSON schema.
    """
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    # Verify it's a valid OpenAPI schema
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema
    assert "/ping" in schema["paths"]
    assert "/recommend/{user_id}" in schema["paths"]

