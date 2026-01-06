"""Module for getting recommendations.

Uses the trained model to recommend products for users.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.recommender.utils import load_model_artifacts, check_model_exists

# Configure module logger
logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_TOP_N = 5
DEFAULT_MODEL_DIR = "models"


class UserNotFoundError(Exception):
    """Raised when user is not found in training data."""
    pass


class ModelNotFoundError(Exception):
    """Raised when model files cannot be found."""
    pass


def recommend_products_for_user(
    user_id: int,
    model_path: str = DEFAULT_MODEL_DIR,
    top_n: int = DEFAULT_TOP_N,
    user_product_matrix: Optional[csr_matrix] = None,
    allow_cold_start: bool = True,
) -> List[int]:
    """Get recommendations for a user.
    
    Loads the model and returns top N product recommendations.
    
    Args:
        user_id: User ID to get recommendations for
        model_path: Path to model directory
        top_n: Number of recommendations to return
        user_product_matrix: Optional pre-loaded user-product matrix
        allow_cold_start: If True, return popular items for unknown users.
                         If False, raise UserNotFoundError for unknown users.
    
    Returns:
        List of recommended product IDs
    
    Raises:
        ModelNotFoundError: If model files are missing
        UserNotFoundError: If user not found and allow_cold_start=False
        Exception: For other unexpected errors
    """
    import time
    
    start_time = time.time()
    
    logger.info(
        "Starting recommendation generation",
        extra={
            "user_id": user_id,
            "top_n": top_n,
            "model_path": model_path,
            "allow_cold_start": allow_cold_start,
        }
    )

    try:
        # Check if model exists
        if not check_model_exists(model_path):
            logger.error(
                "Model files not found",
                extra={
                    "user_id": user_id,
                    "model_path": model_path,
                },
                exc_info=True,
            )
            raise ModelNotFoundError(
                f"Model not found at '{model_path}'. Please train a model first."
            )
        
        # Load model artifacts
        load_start = time.time()
        try:
            model, user_id_to_idx, product_id_to_idx = load_model_artifacts(model_path)
        except FileNotFoundError as e:
            logger.error(
                "Model files not found during load",
                extra={
                    "user_id": user_id,
                    "model_path": model_path,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise ModelNotFoundError(
                f"Model files missing at '{model_path}': {str(e)}"
            ) from e
        except Exception as e:
            logger.error(
                "Failed to load model artifacts",
                extra={
                    "user_id": user_id,
                    "model_path": model_path,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            raise Exception(f"Failed to load model: {str(e)}") from e
        
        load_time = time.time() - load_start
        
        logger.info(
            "Model loaded successfully",
            extra={
                "user_id": user_id,
                "load_time_ms": round(load_time * 1000, 2),
                "num_users": len(user_id_to_idx),
                "num_products": len(product_id_to_idx),
            }
        )

        # Handle unknown user
        if user_id not in user_id_to_idx:
            if not allow_cold_start:
                logger.warning(
                    "User not found in training data",
                    extra={
                        "user_id": user_id,
                        "num_known_users": len(user_id_to_idx),
                    }
                )
                raise UserNotFoundError(
                    f"User {user_id} not found in training data. "
                    f"Known users: {min(user_id_to_idx.keys())} to {max(user_id_to_idx.keys())}"
                )
            
            logger.warning(
                "User not in training data, using cold-start strategy",
                extra={
                    "user_id": user_id,
                    "strategy": "cold_start",
                }
            )
            return _handle_cold_start_user(
                user_id, product_id_to_idx, top_n
            )

        # Get user index
        user_idx = user_id_to_idx[user_id]

        # Generate recommendations
        scoring_start = time.time()
        recommendations = _compute_recommendations(
            user_idx=user_idx,
            model=model,
            product_id_to_idx=product_id_to_idx,
            top_n=top_n,
            user_product_matrix=user_product_matrix,
        )
        scoring_time = time.time() - scoring_start
        total_time = time.time() - start_time

        logger.info(
            "Recommendations generated successfully",
            extra={
                "user_id": user_id,
                "num_recommendations": len(recommendations),
                "scoring_time_ms": round(scoring_time * 1000, 2),
                "total_time_ms": round(total_time * 1000, 2),
            }
        )

        return recommendations

    except (ModelNotFoundError, UserNotFoundError):
        # Re-raise these as-is
        raise
    except FileNotFoundError as e:
        # Legacy handling for backwards compatibility
        logger.error(
            "Model files not found",
            extra={
                "user_id": user_id,
                "model_path": model_path,
                "error": str(e),
            },
            exc_info=True,
        )
        raise ModelNotFoundError(
            f"Model files not found at '{model_path}': {str(e)}"
        ) from e
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(
            "Recommendation generation failed with unexpected error",
            extra={
                "user_id": user_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "total_time_ms": round(total_time * 1000, 2),
            },
            exc_info=True,
        )
        raise Exception(f"Failed to generate recommendations: {str(e)}") from e


def _compute_recommendations(
    user_idx: int,
    model: TruncatedSVD,
    product_id_to_idx: Dict[int, int],
    top_n: int,
    user_product_matrix: Optional[csr_matrix] = None,
) -> List[int]:
    """Get recommendations for a user using the model.
    """
    import time
    
    start_time = time.time()
    n_products = len(product_id_to_idx)

    # Make a vector for this user
    user_vector = np.zeros(n_products)

    # Use actual purchase data if we have it
    if user_product_matrix is not None:
        user_interactions = user_product_matrix[user_idx].toarray().flatten()
        user_vector = user_interactions
    else:
        logger.debug("Using model without interaction matrix")

    # Get scores for all products
    if user_product_matrix is not None:
        user_latent = model.transform([user_vector])[0]
        predicted_scores = np.dot(user_latent, model.components_)
    else:
        # Just use average if no data
        predicted_scores = np.mean(model.components_, axis=0)

    # Map back to product IDs
    idx_to_product_id = {idx: pid for pid, idx in product_id_to_idx.items()}

    # Find what user already bought
    purchased_product_indices = set()
    if user_product_matrix is not None:
        purchased_product_indices = set(
            user_product_matrix[user_idx].nonzero()[1].tolist()
        )

    # Don't recommend stuff they already have
    filtered_scores = predicted_scores.copy()
    for purchased_idx in purchased_product_indices:
        filtered_scores[purchased_idx] = -np.inf

    # Get valid products
    valid_indices = np.where(filtered_scores != -np.inf)[0]
    
    if len(valid_indices) == 0:
        logger.warning("No valid products available for recommendation")
        return []

    # Get top N
    n_available = min(top_n, len(valid_indices))
    sorted_indices = np.argsort(filtered_scores[valid_indices])[::-1]
    top_product_indices = valid_indices[sorted_indices[:n_available]]

    # Convert to product IDs
    recommended_product_ids = [
        int(idx_to_product_id[int(idx)]) for idx in top_product_indices
    ]
    
    compute_time = time.time() - start_time

    logger.debug(
        "Computed recommendations",
        extra={
            "user_idx": user_idx,
            "num_recommendations": len(recommended_product_ids),
            "compute_time_ms": round(compute_time * 1000, 2),
        }
    )

    return recommended_product_ids


def _handle_cold_start_user(
    user_id: int,
    product_id_to_idx: Dict[int, int],
    top_n: int,
) -> List[int]:
    """Handle new users who aren't in the training data.
    
    Just returns the first N products as a simple fallback.
    """
    logger.info(f"Applying cold start strategy for user {user_id}")

    all_product_ids = sorted(product_id_to_idx.keys())

    # Just return first N products
    n_available = min(top_n, len(all_product_ids))
    cold_start_recommendations = [int(pid) for pid in all_product_ids[:n_available]]

    logger.debug(
        f"Cold start recommendations for user {user_id}: "
        f"{cold_start_recommendations}"
    )

    return cold_start_recommendations


def batch_recommend_for_users(
    user_ids: List[int],
    model_path: str = DEFAULT_MODEL_DIR,
    top_n: int = DEFAULT_TOP_N,
) -> Dict[int, List[int]]:
    """Generate recommendations for multiple users in batch.

    More efficient than calling recommend_products_for_user() multiple times,
    as it loads the model once and reuses it for all users.

    Args:
        user_ids: List of user IDs for which to generate recommendations.
        model_path: Path to directory containing model artifacts.
        top_n: Number of recommendations per user.

    Returns:
        Dictionary mapping user IDs to their recommended product ID lists.

    Raises:
        FileNotFoundError: If model files are not found at model_path.

    Example:
        >>> user_ids = [1, 5, 10, 42]
        >>> recommendations = batch_recommend_for_users(
        ...     user_ids=user_ids,
        ...     top_n=5
        ... )
        >>> for user_id, recs in recommendations.items():
        ...     print(f"User {user_id}: {recs}")
    """
    logger.info(
        f"Generating batch recommendations for {len(user_ids)} users, "
        f"top_n={top_n}"
    )

    results = {}

    # Load model once for efficiency
    try:
        model, user_id_to_idx, product_id_to_idx = load_model_artifacts(model_path)
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        raise

    # Generate recommendations for each user
    for user_id in user_ids:
        try:
            if user_id not in user_id_to_idx:
                logger.warning(f"User {user_id} not found in training data")
                recommendations = _handle_cold_start_user(
                    user_id, product_id_to_idx, top_n
                )
            else:
                user_idx = user_id_to_idx[user_id]
                recommendations = _compute_recommendations(
                    user_idx=user_idx,
                    model=model,
                    product_id_to_idx=product_id_to_idx,
                    top_n=top_n,
                    user_product_matrix=None,  # Can be loaded once if needed
                )

            results[user_id] = recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations for user {user_id}: {e}")
            # Continue with other users, return empty list for failed user
            results[user_id] = []

    logger.info(f"Batch recommendations completed for {len(results)} users")

    return results
