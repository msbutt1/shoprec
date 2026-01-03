"""Inference module for generating product recommendations.

This module provides functionality to generate product recommendations using
a trained collaborative filtering model. It loads model artifacts and computes
recommendations based on user purchase history and latent feature similarities.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.recommender.utils import load_model_artifacts

# Configure module logger
logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_TOP_N = 5
DEFAULT_MODEL_DIR = "models"


def recommend_products_for_user(
    user_id: int,
    model_path: str = DEFAULT_MODEL_DIR,
    top_n: int = DEFAULT_TOP_N,
    user_product_matrix: Optional[csr_matrix] = None,
) -> List[int]:
    """Generate product recommendations for a specific user.

    Uses a trained collaborative filtering model to generate personalized
    product recommendations. The function loads the model, computes similarity
    scores for all products, and returns the top-N recommendations.

    Args:
        user_id: User ID for which to generate recommendations.
        model_path: Path to directory containing model artifacts
            (default: "models").
        top_n: Number of recommendations to return (default: 5).
        user_product_matrix: Optional pre-loaded user-product interaction matrix.
            If None, recommendations are based purely on latent features.

    Returns:
        List of recommended product IDs, sorted by relevance (most relevant first).

    Raises:
        FileNotFoundError: If model files are not found at model_path.
        ValueError: If user_id is invalid or model artifacts are corrupted.

    Example:
        >>> recommendations = recommend_products_for_user(
        ...     user_id=42,
        ...     model_path="models",
        ...     top_n=10
        ... )
        >>> print(f"Top 10 products for user 42: {recommendations}")
    """
    logger.info(
        f"Generating {top_n} recommendations for user {user_id} "
        f"using model from {model_path}"
    )

    try:
        # Load model artifacts
        model, user_id_to_idx, product_id_to_idx = load_model_artifacts(model_path)

        # Handle unknown user
        if user_id not in user_id_to_idx:
            logger.warning(f"User {user_id} not found in training data")
            return _handle_cold_start_user(
                user_id, product_id_to_idx, top_n
            )

        # Get user index
        user_idx = user_id_to_idx[user_id]

        # Generate recommendations
        recommendations = _compute_recommendations(
            user_idx=user_idx,
            model=model,
            product_id_to_idx=product_id_to_idx,
            top_n=top_n,
            user_product_matrix=user_product_matrix,
        )

        logger.info(
            f"Generated {len(recommendations)} recommendations for user {user_id}"
        )

        return recommendations

    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {e}")
        raise ValueError(f"Failed to generate recommendations: {e}") from e


def _compute_recommendations(
    user_idx: int,
    model: TruncatedSVD,
    product_id_to_idx: Dict[int, int],
    top_n: int,
    user_product_matrix: Optional[csr_matrix] = None,
) -> List[int]:
    """Compute product recommendations for a user using the SVD model.

    This function reconstructs the user-product interaction matrix using the
    SVD model's latent features and returns the top-N products with highest
    predicted scores.

    Args:
        user_idx: Index of the user in the model's user dimension.
        model: Trained TruncatedSVD model.
        product_id_to_idx: Mapping from product IDs to matrix indices.
        top_n: Number of recommendations to return.
        user_product_matrix: Optional sparse matrix of user-product interactions.
            Used to filter out already-purchased products.

    Returns:
        List of recommended product IDs.
    """
    # Get the number of products
    n_products = len(product_id_to_idx)

    # Create a one-hot vector for this user (all zeros except for this user)
    user_vector = np.zeros(n_products)

    # If we have the user-product matrix, use actual interactions
    if user_product_matrix is not None:
        user_interactions = user_product_matrix[user_idx].toarray().flatten()
        user_vector = user_interactions
    else:
        # Otherwise, create a simple indicator vector
        # This is less accurate but works when the original matrix is unavailable
        logger.debug("Using cold model inference without interaction matrix")

    # Transform the user vector to latent space and reconstruct
    # This gives us predicted scores for all products
    if user_product_matrix is not None:
        # Transform user interactions to latent space
        user_latent = model.transform([user_vector])[0]

        # Reconstruct scores for all products using the latent representation
        # Score = user_latent · product_latent^T
        # Since model.components_ contains product latent features (n_components × n_products)
        # we compute: user_latent · components_
        predicted_scores = np.dot(user_latent, model.components_)
    else:
        # Without interaction matrix, use simple latent feature similarity
        # Get average latent features as a baseline
        predicted_scores = np.mean(model.components_, axis=0)

    # Create reverse mapping from index to product ID
    idx_to_product_id = {idx: pid for pid, idx in product_id_to_idx.items()}

    # Get products user has already interacted with
    purchased_product_indices = set()
    if user_product_matrix is not None:
        purchased_product_indices = set(
            user_product_matrix[user_idx].nonzero()[1].tolist()
        )

    # Filter out already-purchased products by setting their scores to -inf
    filtered_scores = predicted_scores.copy()
    for purchased_idx in purchased_product_indices:
        filtered_scores[purchased_idx] = -np.inf

    # Get top-N product indices by score
    top_product_indices = np.argsort(filtered_scores)[::-1][:top_n]

    # Convert indices back to product IDs
    recommended_product_ids = [
        idx_to_product_id[idx] for idx in top_product_indices
    ]

    logger.debug(
        f"Top {top_n} product indices: {top_product_indices.tolist()}, "
        f"Product IDs: {recommended_product_ids}"
    )

    return recommended_product_ids


def _handle_cold_start_user(
    user_id: int,
    product_id_to_idx: Dict[int, int],
    top_n: int,
) -> List[int]:
    """Handle recommendations for users not in the training data (cold start).

    For new users without purchase history, returns popular products or
    a default set of recommendations.

    Args:
        user_id: User ID for cold start recommendations.
        product_id_to_idx: Mapping from product IDs to matrix indices.
        top_n: Number of recommendations to return.

    Returns:
        List of recommended product IDs for cold start users.
    """
    logger.info(f"Applying cold start strategy for user {user_id}")

    # Get all product IDs
    all_product_ids = sorted(product_id_to_idx.keys())

    # Return top-N products by product ID (simple fallback strategy)
    # In a production system, this would return popular products
    # based on global statistics
    cold_start_recommendations = all_product_ids[:top_n]

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

