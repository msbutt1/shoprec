"""CLI script to train the ShopRec recommendation model.

This script provides a command-line interface to load purchase data,
train a collaborative filtering model (Truncated SVD), and save the
trained model artifacts to disk.

The script uses a configuration-based approach for easy parameter tuning
and experimentation.

Example Usage:
    # Basic usage with defaults
    python scripts/train_model.py data/fake_purchases.csv

    # Customize output directory and components
    python scripts/train_model.py data/purchases.csv --output-dir models/exp1 --n-components 30

    # Full control over training parameters
    python scripts/train_model.py data/purchases.csv \
        --output-dir models/experiment_1 \
        --n-components 100 \
        --n-iter 20 \
        --random-state 123 \
        --verbose

    # Custom column names
    python scripts/train_model.py data/custom.csv \
        --user-col customer_id \
        --item-col sku_id
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.recommender.train import (
    TrainingConfig,
    train_with_config,
    DEFAULT_N_COMPONENTS,
    DEFAULT_N_ITERATIONS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_OUTPUT_DIR,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train the ShopRec collaborative filtering model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to CSV file containing purchase data (required columns: user_id, product_id)",
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the trained model and mappings",
    )
    model_group.add_argument(
        "--n-components",
        type=int,
        default=DEFAULT_N_COMPONENTS,
        help="Number of latent features for SVD (higher = more expressive but slower)",
    )
    model_group.add_argument(
        "--n-iter",
        type=int,
        default=DEFAULT_N_ITERATIONS,
        help="Number of iterations for the randomized SVD solver",
    )
    model_group.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for reproducibility",
    )

    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--user-col",
        type=str,
        default="user_id",
        help="Name of the user ID column in the CSV",
    )
    data_group.add_argument(
        "--item-col",
        type=str,
        default="product_id",
        help="Name of the item/product ID column in the CSV",
    )

    # Embedding configuration
    embedding_group = parser.add_argument_group("Embedding Configuration")
    embedding_group.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable product embedding generation",
    )
    embedding_group.add_argument(
        "--embedding-method",
        type=str,
        default="random",
        choices=["random", "tfidf"],
        help="Method for generating product embeddings",
    )
    embedding_group.add_argument(
        "--embedding-dim",
        type=int,
        default=50,
        help="Dimensionality of product embeddings",
    )

    # Other options
    parser.add_argument(
        "--no-auto-adjust",
        action="store_true",
        help="Disable automatic adjustment of n_components (will raise error if too large)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--save-config",
        type=str,
        metavar="PATH",
        help="Save the training configuration to a JSON file for reproducibility",
    )

    return parser.parse_args()


def save_config_to_file(config: TrainingConfig, output_path: str) -> None:
    """Save training configuration to a JSON file.

    Args:
        config: Training configuration to save.
        output_path: Path to output JSON file.
    """
    try:
        config_dict = config.to_dict()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to: {output_path}")
    except Exception as e:
        logger.warning(f"Failed to save configuration: {e}")


def main() -> None:
    """Main function for the CLI script."""
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Validate CSV path
    csv_file = Path(args.csv_path)
    if not csv_file.exists():
        logger.error(f"Error: CSV file not found at '{args.csv_path}'")
        sys.exit(1)
    if not csv_file.is_file():
        logger.error(f"Error: Path '{args.csv_path}' is not a file")
        sys.exit(1)

    # Create training configuration
    try:
        config = TrainingConfig(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            n_components=args.n_components,
            n_iter=args.n_iter,
            random_state=args.random_state,
            user_col=args.user_col,
            item_col=args.item_col,
            auto_adjust_components=not args.no_auto_adjust,
            generate_embeddings=not args.no_embeddings,
            embedding_method=args.embedding_method,
            embedding_dim=args.embedding_dim,
        )
    except Exception as e:
        logger.error(f"Failed to create training configuration: {e}")
        sys.exit(1)

    # Display configuration
    logger.info("=" * 70)
    logger.info("ShopRec Model Training CLI")
    logger.info("=" * 70)
    logger.info("Configuration:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key:25s}: {value}")
    logger.info("=" * 70)

    # Save configuration if requested
    if args.save_config:
        save_config_to_file(config, args.save_config)

    # Train the model
    try:
        model, user_map, product_map = train_with_config(config)
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("Training Summary")
        logger.info("=" * 70)
        logger.info(f"Model components: {model.n_components}")
        logger.info(f"Users in training set: {len(user_map)}")
        logger.info(f"Products in training set: {len(product_map)}")
        logger.info(f"Explained variance ratio: {model.explained_variance_ratio_.sum():.4f}")
        logger.info(f"Model artifacts saved to: {config.output_dir}")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Training completed successfully!")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
