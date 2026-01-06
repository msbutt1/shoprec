"""API endpoints for recommendations.

Handles requests to get product recommendations.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

import numpy as np

from src.api.metrics import metrics_service
from src.recommender.infer import _compute_recommendations, _handle_cold_start_user
from src.recommender.utils import check_model_exists, load_model_artifacts
from src.recommender.hybrid import create_hybrid_recommender

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

# Track when model was last loaded
_model_load_timestamp: Optional[datetime] = None


class RecommendationResponse(BaseModel):
    """Response for recommendation requests.
    """

    user_id: int = Field(..., description="User ID for recommendations")
    recommendations: List[int] = Field(
        ..., description="List of recommended product IDs"
    )
    model_version: str = Field(
        default="0.1.0", description="Model version or timestamp"
    )
    scores: Optional[Dict] = Field(
        default=None, description="Score breakdown for explainability (if explain=true)"
    )


def load_model_if_needed(model_dir: str = DEFAULT_MODEL_DIR) -> Dict:
    """Load model if not already loaded.
    
    Uses a cache so we don't reload every time.
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
        global _model_load_timestamp
        _model_cache = {
            "model": model,
            "user_map": user_map,
            "product_map": product_map,
        }
        _model_load_timestamp = datetime.utcnow()

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
    mode: str = "hybrid",
    explain: bool = False,
    cf_weight: float = 0.7,
    content_weight: float = 0.3,
) -> RecommendationResponse:
    """Get recommendations for a user.
    
    Can use hybrid mode (CF + content) or just CF mode.
    """
    # Check mode
    mode = mode.lower() if mode else "hybrid"
    if mode not in ["hybrid", "cf"]:
        logger.warning(f"Invalid mode '{mode}', falling back to 'cf'")
        mode = "cf"
    
    logger.info(f"Generating recommendations for user {user_id}, top_n={top_n}, mode={mode}")

    # Track timing for metrics
    inference_start = time.time()

    try:
        # Load the model
        model_artifacts = load_model_if_needed(model_dir)
        model = model_artifacts["model"]
        user_map = model_artifacts["user_map"]
        product_map = model_artifacts["product_map"]

        if mode == "hybrid":
            # Use hybrid mode
            hybrid_recommender = create_hybrid_recommender(
                model=model,
                user_id_to_idx=user_map,
                product_id_to_idx=product_map,
                model_dir=model_dir,
                cf_weight=cf_weight,
                content_weight=content_weight,
            )

            if explain:
                recommendations, scores = hybrid_recommender.recommend(
                    user_id=user_id,
                    top_n=top_n,
                    user_product_matrix=None,
                    return_scores=True,
                )
            else:
                recommendations = hybrid_recommender.recommend(
                    user_id=user_id,
                    top_n=top_n,
                    user_product_matrix=None,
                    return_scores=False,
                )
                scores = None
        else:
            # Use CF-only mode
            logger.debug("Using CF-only recommendation mode")
            
            if user_id not in user_map:
                logger.warning(f"User {user_id} not found in training data. Applying cold-start strategy.")
                recommendations = _handle_cold_start_user(user_id, product_map, top_n)
                scores = {"method": "cold_start", "cf_scores": {}, "content_scores": {}} if explain else None
            else:
                user_idx = user_map[user_id]
                recommendations = _compute_recommendations(
                    user_idx=user_idx,
                    model=model,
                    product_id_to_idx=product_map,
                    top_n=top_n,
                    user_product_matrix=None,
                )
                
                if explain:
                    # Get CF scores for explainability
                    n_products = len(product_map)
                    predicted_scores = np.mean(model.components_, axis=0)
                    
                    if len(predicted_scores) != n_products:
                        logger.warning(
                            f"Dimension mismatch: model has {len(predicted_scores)} products, "
                            f"but product_map has {n_products}. Truncating or padding."
                        )
                        if len(predicted_scores) > n_products:
                            predicted_scores = predicted_scores[:n_products]
                        else:
                            padding = np.zeros(n_products - len(predicted_scores))
                            predicted_scores = np.concatenate([predicted_scores, padding])
                    
                    idx_to_product_id = {idx: pid for pid, idx in product_map.items()}
                    cf_scores = {
                        int(idx_to_product_id[idx]): float(predicted_scores[idx])
                        for idx in range(n_products)
                        if idx in idx_to_product_id
                    }
                    
                    # Normalize
                    if cf_scores:
                        min_score = min(cf_scores.values())
                        max_score = max(cf_scores.values())
                        if max_score != min_score:
                            cf_scores = {
                                pid: (score - min_score) / (max_score - min_score)
                                for pid, score in cf_scores.items()
                            }
                    
                    scores = {
                        "method": "cf_only",
                        "cf_scores": {pid: cf_scores.get(pid, 0.0) for pid in recommendations},
                        "content_scores": {},
                        "hybrid_scores": {pid: cf_scores.get(pid, 0.0) for pid in recommendations},
                        "cf_weight": 1.0,
                        "content_weight": 0.0,
                    }
                else:
                    scores = None

        # Handle case where user is not found (cold start)
        if not recommendations:
            logger.warning(
                f"No recommendations generated for user {user_id}. "
                "This may indicate the user is not in training data."
            )
            # Return empty list rather than error - cold start handled by inference module
            recommendations = []

        logger.info(
            f"Generated {len(recommendations)} recommendations for user {user_id}"
        )

        # Record metrics
        inference_latency_ms = (time.time() - inference_start) * 1000
        metrics_service.record_inference(inference_latency_ms)

        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            model_version="0.1.0",
            scores=scores if explain else None,
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not found in {model_dir}. Please train a model first.",
        )
    except ValueError as e:
        logger.error(f"Invalid input or model error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}",
        )
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
    
    Clears the cache and loads fresh model files.
    """
    global _model_cache, _model_load_timestamp

    logger.info("Reloading model...")
    _model_cache = None  # Clear cache
    _model_load_timestamp = None

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


def get_model_status(model_dir: str = DEFAULT_MODEL_DIR) -> Dict:
    """Get model status info.
    
    Returns whether model is loaded, when it was loaded, and counts.
    """
    global _model_cache, _model_load_timestamp

    model_loaded = _model_cache is not None
    timestamp_last_loaded = (
        _model_load_timestamp.isoformat() + "Z" if _model_load_timestamp else None
    )

    num_users = 0
    num_products = 0

    if model_loaded and _model_cache is not None:
        num_users = len(_model_cache.get("user_map", {}))
        num_products = len(_model_cache.get("product_map", {}))

    return {
        "model_loaded": model_loaded,
        "timestamp_last_loaded": timestamp_last_loaded,
        "num_users": num_users,
        "num_products": num_products,
    }
