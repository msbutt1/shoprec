"""Collaborative filtering model training module.

This module provides functionality to train a basic collaborative filtering
recommendation model using matrix factorization via Truncated SVD. It processes
user-product purchase data and learns latent features for generating recommendations.
"""

import logging
from typing import Dict, Tuple

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.recommender.utils import (
    load_csv_to_matrix,
    save_model_artifacts,
)

# Configure module logger
logger = logging.getLogger(__name__)

# Model configuration constants
DEFAULT_N_COMPONENTS = 50
DEFAULT_N_ITERATIONS = 10
DEFAULT_RANDOM_STATE = 42




def train_svd_model(
    user_product_matrix: csr_matrix,
    n_components: int = DEFAULT_N_COMPONENTS,
    n_iter: int = DEFAULT_N_ITERATIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> TruncatedSVD:
    """Train a Truncated SVD model for collaborative filtering.

    Uses matrix factorization to learn latent features from user-product
    interactions. The SVD decomposes the interaction matrix into lower-dimensional
    representations that capture user preferences and product characteristics.

    Args:
        user_product_matrix: Sparse matrix of user-product interactions.
        n_components: Number of latent features to extract. Must be less than
            min(n_users, n_products).
        n_iter: Number of iterations for randomized SVD solver.
        random_state: Random seed for reproducibility.

    Returns:
        Trained TruncatedSVD model.

    Raises:
        ValueError: If n_components is invalid or matrix is empty.
    """
    n_users, n_products = user_product_matrix.shape
    
    if n_components >= min(n_users, n_products):
        raise ValueError(
            f"n_components ({n_components}) must be less than "
            f"min(n_users, n_products) = {min(n_users, n_products)}"
        )

    if user_product_matrix.nnz == 0:
        raise ValueError("Cannot train on empty interaction matrix")

    logger.info(f"Training SVD model with {n_components} components")
    logger.info(f"Random state: {random_state}, Iterations: {n_iter}")

    model = TruncatedSVD(
        n_components=n_components,
        n_iter=n_iter,
        random_state=random_state,
    )

    model.fit(user_product_matrix)

    logger.info("Model training completed")
    logger.info(f"Explained variance ratio: {model.explained_variance_ratio_.sum():.4f}")

    return model




def train_basic_cf_model(
    csv_path: str,
    output_dir: str = "models",
    n_components: int = DEFAULT_N_COMPONENTS,
    n_iter: int = DEFAULT_N_ITERATIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[TruncatedSVD, Dict[int, int], Dict[int, int]]:
    """Train a basic collaborative filtering model from purchase data.

    This is the main entry point for training. It orchestrates the complete
    training pipeline: loading data, building the interaction matrix, training
    the SVD model, and saving artifacts.

    Args:
        csv_path: Path to CSV file with columns: user_id, product_id, timestamp.
        output_dir: Directory where model artifacts will be saved.
        n_components: Number of latent features for SVD (default: 50).
            Will be automatically adjusted if larger than matrix dimensions.
        n_iter: Number of iterations for SVD solver (default: 10).
        random_state: Random seed for reproducibility (default: 42).

    Returns:
        A tuple containing:
            - Trained TruncatedSVD model
            - Dictionary mapping user IDs to matrix indices
            - Dictionary mapping product IDs to matrix indices

    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If data is invalid or training parameters are incorrect.
        OSError: If unable to save model artifacts.

    Example:
        >>> model, user_map, product_map = train_basic_cf_model(
        ...     "data/purchases.csv",
        ...     output_dir="models",
        ...     n_components=30
        ... )
        >>> print(f"Trained model with {model.n_components} components")
    """
    logger.info("=" * 60)
    logger.info("Starting collaborative filtering model training")
    logger.info("=" * 60)

    try:
        # Step 1: Load CSV and build user-product interaction matrix
        user_product_matrix, user_id_to_idx, product_id_to_idx = (
            load_csv_to_matrix(
                csv_path,
                user_col="user_id",
                item_col="product_id",
                binary=True,
            )
        )

        # Step 3: Adjust n_components if necessary
        n_users, n_products = user_product_matrix.shape
        max_components = min(n_users, n_products) - 1
        
        if n_components >= min(n_users, n_products):
            adjusted_components = max_components
            logger.warning(
                f"Requested n_components ({n_components}) is too large for "
                f"matrix size ({n_users}x{n_products}). "
                f"Adjusting to {adjusted_components}."
            )
            n_components = adjusted_components

        # Step 4: Train SVD model
        model = train_svd_model(
            user_product_matrix,
            n_components=n_components,
            n_iter=n_iter,
            random_state=random_state,
        )

        # Step 5: Save model artifacts
        save_model_artifacts(model, user_id_to_idx, product_id_to_idx, output_dir)

        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info("=" * 60)

        return model, user_id_to_idx, product_id_to_idx

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


def main() -> None:
    """Main entry point for command-line execution."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Train model with default parameters
    csv_path = "data/fake_purchases.csv"
    output_dir = "models"

    try:
        train_basic_cf_model(csv_path, output_dir)
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        exit(1)


if __name__ == "__main__":
    main()

