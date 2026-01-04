"""Training module for the recommendation model.

Trains a model using SVD on user-product purchase data.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.recommender.utils import (
    load_csv_to_matrix,
    save_model_artifacts,
)
from src.recommender.embed import (
    generate_random_embeddings,
    generate_simulated_product_metadata,
    generate_tfidf_embeddings,
    save_embeddings,
)

# Configure module logger
logger = logging.getLogger(__name__)

# Model configuration constants
DEFAULT_N_COMPONENTS = 50
DEFAULT_N_ITERATIONS = 10
DEFAULT_RANDOM_STATE = 42
DEFAULT_OUTPUT_DIR = "models"


@dataclass
class TrainingConfig:
    """Config for training the model.
    
    Holds all the settings needed to train the recommendation model.
    """
    csv_path: str
    output_dir: str = DEFAULT_OUTPUT_DIR
    n_components: int = DEFAULT_N_COMPONENTS
    n_iter: int = DEFAULT_N_ITERATIONS
    random_state: int = DEFAULT_RANDOM_STATE
    user_col: str = "user_id"
    item_col: str = "product_id"
    auto_adjust_components: bool = True
    generate_embeddings: bool = True
    embedding_method: str = "random"  # "random" or "tfidf"
    embedding_dim: int = 50
    
    def __post_init__(self):
        """Check that config values are valid."""
        if self.n_components <= 0:
            raise ValueError(f"n_components must be positive, got {self.n_components}")
        if self.n_iter <= 0:
            raise ValueError(f"n_iter must be positive, got {self.n_iter}")
        
        csv_file = Path(self.csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
    
    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            "csv_path": self.csv_path,
            "output_dir": self.output_dir,
            "n_components": self.n_components,
            "n_iter": self.n_iter,
            "random_state": self.random_state,
            "user_col": self.user_col,
            "item_col": self.item_col,
            "auto_adjust_components": self.auto_adjust_components,
            "generate_embeddings": self.generate_embeddings,
            "embedding_method": self.embedding_method,
            "embedding_dim": self.embedding_dim,
        }


def train_svd_model(
    user_product_matrix: csr_matrix,
    n_components: int = DEFAULT_N_COMPONENTS,
    n_iter: int = DEFAULT_N_ITERATIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> TruncatedSVD:
    """Train the SVD model.
    
    Takes the user-product matrix and trains a model on it.
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


def train_with_config(config: TrainingConfig) -> Tuple[TruncatedSVD, Dict[int, int], Dict[int, int]]:
    """Train the model using a config object.
    
    Main function for training. Loads data, trains model, saves everything.
    """
    logger.info("=" * 60)
    logger.info("Starting collaborative filtering model training")
    logger.info("=" * 60)
    logger.info(f"Configuration: {config.to_dict()}")

    try:
        # Load the data
        logger.info(f"Loading data from: {config.csv_path}")
        user_product_matrix, user_id_to_idx, product_id_to_idx = (
            load_csv_to_matrix(
                config.csv_path,
                user_col=config.user_col,
                item_col=config.item_col,
                binary=True,
            )
        )

        # Fix n_components if it's too big
        n_users, n_products = user_product_matrix.shape
        n_components = config.n_components
        
        if config.auto_adjust_components and n_components >= min(n_users, n_products):
            max_components = min(n_users, n_products) - 1
            logger.warning(
                f"n_components ({n_components}) is too big for "
                f"matrix size ({n_users}x{n_products}). "
                f"Changing to {max_components}."
            )
            n_components = max_components

        # Train the model
        logger.info(f"Training model with {n_components} components")
        model = train_svd_model(
            user_product_matrix,
            n_components=n_components,
            n_iter=config.n_iter,
            random_state=config.random_state,
        )

        # Make embeddings if needed
        if config.generate_embeddings:
            logger.info(f"Generating product embeddings using method: {config.embedding_method}")
            product_ids = sorted(product_id_to_idx.keys())
            
            if config.embedding_method == "random":
                embeddings = generate_random_embeddings(
                    product_ids=product_ids,
                    embedding_dim=config.embedding_dim,
                    random_seed=config.random_state,
                )
            elif config.embedding_method == "tfidf":
                logger.info("Generating simulated product metadata for TF-IDF")
                product_metadata = generate_simulated_product_metadata(
                    product_ids=product_ids,
                    random_seed=config.random_state,
                )
                embeddings = generate_tfidf_embeddings(
                    product_metadata=product_metadata,
                    max_features=config.embedding_dim,
                )
            else:
                logger.warning(f"Unknown embedding method: {config.embedding_method}. Skipping embeddings.")
                embeddings = None
            
            if embeddings is not None:
                save_embeddings(embeddings, config.output_dir)
                logger.info(f"Saved product embeddings to: {config.output_dir}")

        # Save the model
        logger.info(f"Saving model artifacts to: {config.output_dir}")
        save_model_artifacts(model, user_id_to_idx, product_id_to_idx, config.output_dir)

        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {config.output_dir}")
        logger.info(f"Final n_components: {model.n_components}")
        logger.info(f"Explained variance: {model.explained_variance_ratio_.sum():.4f}")
        if config.generate_embeddings:
            logger.info(f"Product embeddings: method={config.embedding_method}, dim={config.embedding_dim}")
        logger.info("=" * 60)

        return model, user_id_to_idx, product_id_to_idx

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


def train_basic_cf_model(
    csv_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    n_components: int = DEFAULT_N_COMPONENTS,
    n_iter: int = DEFAULT_N_ITERATIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[TruncatedSVD, Dict[int, int], Dict[int, int]]:
    """Train the model from a CSV file.
    
    Old way of training. Still works but train_with_config is better.
    """
    config = TrainingConfig(
        csv_path=csv_path,
        output_dir=output_dir,
        n_components=n_components,
        n_iter=n_iter,
        random_state=random_state,
    )
    return train_with_config(config)


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
