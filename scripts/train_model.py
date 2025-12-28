"""Command-line interface for training the collaborative filtering model.

This script provides a CLI for training the ShopRec recommendation model
from purchase data stored in CSV format.

Example:
    Train a model with default settings:
        $ python scripts/train_model.py data/fake_purchases.csv

    Train with custom parameters:
        $ python scripts/train_model.py data/purchases.csv \\
            --output-dir models/production \\
            --n-components 30 \\
            --n-iter 20
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.recommender.train import (
    DEFAULT_N_COMPONENTS,
    DEFAULT_N_ITERATIONS,
    DEFAULT_RANDOM_STATE,
    train_basic_cf_model,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script.

    Args:
        verbose: If True, set log level to DEBUG. Otherwise, use INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a collaborative filtering recommendation model from CSV data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python scripts/train_model.py data/purchases.csv

  # Train with custom output directory and components
  python scripts/train_model.py data/purchases.csv --output-dir models/prod --n-components 30

  # Train with verbose logging
  python scripts/train_model.py data/purchases.csv --verbose
        """,
    )

    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to CSV file containing purchase data with columns: "
        "user_id, product_id, timestamp",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help=f"Directory where model artifacts will be saved (default: models)",
    )

    parser.add_argument(
        "--n-components",
        type=int,
        default=DEFAULT_N_COMPONENTS,
        help=f"Number of latent features for SVD (default: {DEFAULT_N_COMPONENTS}). "
        "Will be automatically adjusted if too large for data dimensions.",
    )

    parser.add_argument(
        "--n-iter",
        type=int,
        default=DEFAULT_N_ITERATIONS,
        help=f"Number of iterations for SVD solver (default: {DEFAULT_N_ITERATIONS})",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_STATE})",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    return parser.parse_args()


def validate_csv_path(csv_path: str) -> None:
    """Validate that the CSV file exists and is readable.

    Args:
        csv_path: Path to CSV file.

    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If path is not a file.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {csv_path}")


def main() -> int:
    """Main entry point for the training script.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()

        # Setup logging
        setup_logging(verbose=args.verbose)
        logger = logging.getLogger(__name__)

        # Validate CSV path
        logger.info(f"Validating CSV path: {args.csv_path}")
        validate_csv_path(args.csv_path)

        # Display configuration
        logger.info("=" * 70)
        logger.info("Training Configuration")
        logger.info("=" * 70)
        logger.info(f"CSV path:       {args.csv_path}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Components:     {args.n_components}")
        logger.info(f"Iterations:      {args.n_iter}")
        logger.info(f"Random state:   {args.random_state}")
        logger.info("=" * 70)

        # Train the model
        model, user_map, product_map = train_basic_cf_model(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            n_components=args.n_components,
            n_iter=args.n_iter,
            random_state=args.random_state,
        )

        # Display results
        logger.info("=" * 70)
        logger.info("Training Summary")
        logger.info("=" * 70)
        logger.info(f"Model components: {model.n_components}")
        logger.info(f"Number of users:  {len(user_map)}")
        logger.info(f"Number of products: {len(product_map)}")
        logger.info(f"Explained variance: {model.explained_variance_ratio_.sum():.4f}")
        logger.info(f"Model saved to: {Path(args.output_dir).absolute()}")
        logger.info("=" * 70)

        logger.info("Training completed successfully!")
        return 0

    except FileNotFoundError as e:
        logging.error(f"File error: {e}")
        return 1
    except ValueError as e:
        logging.error(f"Validation error: {e}")
        return 1
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user")
        return 130
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

