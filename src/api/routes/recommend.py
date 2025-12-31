"""Recommendation endpoints for the ShopRec API.

This module provides API endpoints for generating product recommendations
based on user purchase history and collaborative filtering models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.recommender.utils import check_model_exists, load_model_artifacts

# Configure module logger
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(
    prefix="/recommend",
    tags=["recommendations"],
)

# Default model directory
DEFAULT_MODEL_DIR = "models"

# Cache for loaded model artifacts
_model_cache: Optional[Dict] = None


class RecommendationResponse(BaseModel):
    """Response model for recommendation requests.

    Attributes:
        user_id: The user ID for which recommendations were generated.
        recommendations: List of recommended product IDs.
        model_version: Version or timestamp of the model used.
    """

    user_id: int = Field(..., description="User ID for recommendations")
    recommendations: List[int] = Field(
        ..., description="List of recommended product IDs"
    )
    model_version: str = Field(
        default="0.1.0", description="Model version or timestamp"
    )


def load_model_if_needed(model_dir: str = DEFAULT_MODEL_DIR) -> Dict:
    """Load model artifacts from disk if not already loaded.

    Uses a module-level cache to avoid reloading the model on every request.

    Args:
        model_dir: Directory containing model artifacts.

    Returns:
        Dictionary containing:
            - model: Trained TruncatedSVD model
            - user_map: User ID to index mapping
            - product_map: Product ID to index mapping

    Raises:
        HTTPException: If model files are not found or cannot be loaded.
    """
    global _model_cache

    # Return cached model if available
    if _model_cache is not None:
        logger.debug("Using cached model")
        return _model_cache

    # Check if model exists
    if not check_model_exists(model_dir):
        logger.error(f"Model not found in {model_dir}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not found in {model_dir}. Please train a model first.",
        )

    # Load model artifacts
    try:
        logger.info(f"Loading model from {model_dir}")
        model, user_map, product_map = load_model_artifacts(model_dir)

        # Cache the loaded model
        _model_cache = {
            "model": model,
            "user_map": user_map,
            "product_map": product_map,
        }

        logger.info("Model loaded successfully")
        return _model_cache

    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )


@router.get("/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
    user_id: int,
    top_n: int = 10,
    model_dir: str = DEFAULT_MODEL_DIR,
) -> RecommendationResponse:
    """Get product recommendations for a user.

    Generates personalized product recommendations based on the user's
    purchase history and collaborative filtering model.

    Args:
        user_id: User ID for which to generate recommendations.
        top_n: Number of recommendations to return (default: 10).
        model_dir: Directory containing model artifacts (default: "models").

    Returns:
        RecommendationResponse containing user_id and list of recommended
        product IDs.

    Raises:
        HTTPException: If model is not found or if there's an error generating
            recommendations.

    Example:
        GET /recommend/42?top_n=5
        Returns top 5 product recommendations for user 42.
    """
    logger.info(f"Generating recommendations for user {user_id}, top_n={top_n}")

    try:
        # Load model artifacts (uses cache if available)
        model_artifacts = load_model_if_needed(model_dir)
        user_map = model_artifacts["user_map"]
        product_map = model_artifacts["product_map"]

        # Check if user exists in training data
        if user_id not in user_map:
            logger.warning(f"User {user_id} not found in training data")
            # For now, return dummy recommendations for unknown users
            # TODO: Implement cold-start strategy for new users
            dummy_recommendations = list(range(1, min(top_n + 1, 101)))
            return RecommendationResponse(
                user_id=user_id,
                recommendations=dummy_recommendations,
                model_version="0.1.0",
            )

        # TODO: Implement actual recommendation logic using the SVD model
        # For now, return static dummy product IDs
        # The real implementation will:
        # 1. Get user's latent features from the model
        # 2. Compute similarity scores with all products
        # 3. Return top-N products the user hasn't purchased yet

        # Get all product IDs from the training data
        all_product_ids = list(product_map.keys())

        # Return top_n products as dummy recommendations
        dummy_recommendations = all_product_ids[:top_n]

        logger.info(
            f"Generated {len(dummy_recommendations)} recommendations for user {user_id}"
        )

        return RecommendationResponse(
            user_id=user_id,
            recommendations=dummy_recommendations,
            model_version="0.1.0",
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(
            f"Error generating recommendations for user {user_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendations: {str(e)}",
        )


@router.post("/reload-model")
def reload_model(model_dir: str = DEFAULT_MODEL_DIR) -> Dict[str, str]:
    """Reload the model from disk.

    Forces reloading of model artifacts, clearing the cache. Useful when
    a new model has been trained and needs to be loaded without restarting
    the server.

    Args:
        model_dir: Directory containing model artifacts.

    Returns:
        Dictionary with status message.

    Raises:
        HTTPException: If model cannot be reloaded.
    """
    global _model_cache

    logger.info("Reloading model...")
    _model_cache = None  # Clear cache

    try:
        load_model_if_needed(model_dir)
        return {"status": "Model reloaded successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reload model: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}",
        )

