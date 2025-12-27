"""Utility functions for recommendation system.

This module provides helper functions for data loading, model artifact management,
and common operations used throughout the recommendation system.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Configure module logger
logger = logging.getLogger(__name__)

# Model artifact filenames
MODEL_FILENAME = "svd_model.joblib"
USER_MAPPING_FILENAME = "user_id_mapping.joblib"
PRODUCT_MAPPING_FILENAME = "product_id_mapping.joblib"


def load_csv_to_matrix(
    csv_path: str,
    user_col: str = "user_id",
    item_col: str = "product_id",
    value_col: Optional[str] = None,
    binary: bool = True,
) -> Tuple[csr_matrix, Dict[int, int], Dict[int, int]]:
    """Load CSV data and convert to sparse user-item interaction matrix.

    Reads a CSV file containing user-item interactions and constructs a sparse
    matrix where rows represent users and columns represent items. Supports
    both binary (purchased/not purchased) and weighted interactions.

    Args:
        csv_path: Path to CSV file containing interaction data.
        user_col: Name of the column containing user identifiers.
        item_col: Name of the column containing item/product identifiers.
        value_col: Optional name of column containing interaction values.
            If None and binary=True, uses binary values (1 for interaction).
            If None and binary=False, counts interactions.
        binary: If True, creates binary matrix (1 if interaction exists).
            If False and value_col is None, counts duplicate interactions.

    Returns:
        A tuple containing:
            - Sparse CSR matrix of shape (n_users, n_items) with interactions
            - Dictionary mapping user_id to matrix row index
            - Dictionary mapping item_id to matrix column index

    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If CSV is missing required columns or is empty.

    Example:
        >>> matrix, user_map, item_map = load_csv_to_matrix(
        ...     "data/purchases.csv",
        ...     user_col="user_id",
        ...     item_col="product_id"
        ... )
        >>> print(f"Matrix shape: {matrix.shape}")
        >>> print(f"Number of users: {len(user_map)}")
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_columns = {user_col, item_col}
    if value_col is not None:
        required_columns.add(value_col)

    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"CSV missing required columns: {missing}")

    if df.empty:
        raise ValueError("Cannot create matrix from empty CSV")

    logger.info(f"Loaded {len(df)} interaction records")

    # Get unique users and items
    unique_users = sorted(df[user_col].unique())
    unique_items = sorted(df[item_col].unique())

    # Create ID to index mappings
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}

    logger.info(f"Unique users: {len(unique_users)}")
    logger.info(f"Unique items: {len(unique_items)}")

    # Map IDs to indices
    row_indices = df[user_col].map(user_id_to_idx).values
    col_indices = df[item_col].map(item_id_to_idx).values

    # Determine interaction values
    if binary:
        # Binary matrix: 1 if interaction exists
        data = np.ones(len(df), dtype=np.float32)
    elif value_col is not None:
        # Use provided values
        data = df[value_col].values.astype(np.float32)
    else:
        # Count interactions (handle duplicates by summing)
        # Create a temporary DataFrame to count interactions
        interaction_counts = (
            df.groupby([user_col, item_col]).size().reset_index(name="count")
        )
        row_indices = interaction_counts[user_col].map(user_id_to_idx).values
        col_indices = interaction_counts[item_col].map(item_id_to_idx).values
        data = interaction_counts["count"].values.astype(np.float32)

    # Create sparse matrix
    n_users = len(unique_users)
    n_items = len(unique_items)

    user_item_matrix = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )

    # Remove explicit zeros (for binary, this handles duplicates)
    user_item_matrix.eliminate_zeros()

    logger.info(f"Matrix shape: {user_item_matrix.shape}")
    logger.info(f"Matrix density: {user_item_matrix.nnz / (n_users * n_items):.4%}")
    logger.info(f"Non-zero entries: {user_item_matrix.nnz}")

    return user_item_matrix, user_id_to_idx, item_id_to_idx


def save_model_artifacts(
    model: TruncatedSVD,
    user_id_to_idx: Dict[int, int],
    item_id_to_idx: Dict[int, int],
    output_dir: str,
    model_filename: str = MODEL_FILENAME,
    user_mapping_filename: str = USER_MAPPING_FILENAME,
    item_mapping_filename: str = PRODUCT_MAPPING_FILENAME,
) -> None:
    """Save trained model and ID mappings to disk.

    Saves the SVD model and user/item ID mappings as separate joblib files
    in the specified output directory. Creates the directory if it doesn't exist.

    Args:
        model: Trained TruncatedSVD model to save.
        user_id_to_idx: Dictionary mapping user IDs to matrix row indices.
        item_id_to_idx: Dictionary mapping item IDs to matrix column indices.
        output_dir: Directory path where artifacts will be saved.
        model_filename: Filename for the model (default: "svd_model.joblib").
        user_mapping_filename: Filename for user mapping (default: "user_id_mapping.joblib").
        item_mapping_filename: Filename for item mapping (default: "product_id_mapping.joblib").

    Raises:
        OSError: If unable to create output directory or save files.

    Example:
        >>> from sklearn.decomposition import TruncatedSVD
        >>> model = TruncatedSVD(n_components=30)
        >>> model.fit(matrix)
        >>> save_model_artifacts(
        ...     model,
        ...     user_map,
        ...     item_map,
        ...     "models"
        ... )
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model artifacts to {output_dir}")

    # Save model
    model_path = output_path / model_filename
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")

    # Save user ID mapping
    user_mapping_path = output_path / user_mapping_filename
    joblib.dump(user_id_to_idx, user_mapping_path)
    logger.info(f"Saved user mapping to {user_mapping_path}")

    # Save item ID mapping
    item_mapping_path = output_path / item_mapping_filename
    joblib.dump(item_id_to_idx, item_mapping_path)
    logger.info(f"Saved item mapping to {item_mapping_path}")


def load_model_artifacts(
    model_dir: str,
    model_filename: str = MODEL_FILENAME,
    user_mapping_filename: str = USER_MAPPING_FILENAME,
    item_mapping_filename: str = PRODUCT_MAPPING_FILENAME,
) -> Tuple[TruncatedSVD, Dict[int, int], Dict[int, int]]:
    """Load trained model and ID mappings from disk.

    Loads the SVD model and user/item ID mappings from joblib files in the
    specified directory.

    Args:
        model_dir: Directory path where artifacts are stored.
        model_filename: Filename for the model (default: "svd_model.joblib").
        user_mapping_filename: Filename for user mapping (default: "user_id_mapping.joblib").
        item_mapping_filename: Filename for item mapping (default: "product_id_mapping.joblib").

    Returns:
        A tuple containing:
            - Loaded TruncatedSVD model
            - Dictionary mapping user IDs to matrix row indices
            - Dictionary mapping item IDs to matrix column indices

    Raises:
        FileNotFoundError: If any required artifact file is missing.

    Example:
        >>> model, user_map, item_map = load_model_artifacts("models")
        >>> print(f"Model has {model.n_components} components")
        >>> print(f"Number of users: {len(user_map)}")
    """
    model_path = Path(model_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    logger.info(f"Loading model artifacts from {model_dir}")

    # Load model
    model_file = model_path / model_filename
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    model = joblib.load(model_file)
    logger.info(f"Loaded model from {model_file}")
    logger.info(f"Model components: {model.n_components}")

    # Load user ID mapping
    user_mapping_file = model_path / user_mapping_filename
    if not user_mapping_file.exists():
        raise FileNotFoundError(f"User mapping file not found: {user_mapping_file}")
    user_id_to_idx = joblib.load(user_mapping_file)
    logger.info(f"Loaded user mapping from {user_mapping_file}")
    logger.info(f"Number of users: {len(user_id_to_idx)}")

    # Load item ID mapping
    item_mapping_file = model_path / item_mapping_filename
    if not item_mapping_file.exists():
        raise FileNotFoundError(f"Item mapping file not found: {item_mapping_file}")
    item_id_to_idx = joblib.load(item_mapping_file)
    logger.info(f"Loaded item mapping from {item_mapping_file}")
    logger.info(f"Number of items: {len(item_id_to_idx)}")

    return model, user_id_to_idx, item_id_to_idx


def get_model_paths(
    model_dir: str,
    model_filename: str = MODEL_FILENAME,
    user_mapping_filename: str = USER_MAPPING_FILENAME,
    item_mapping_filename: str = PRODUCT_MAPPING_FILENAME,
) -> Tuple[Path, Path, Path]:
    """Get file paths for model artifacts without loading them.

    Useful for checking if model files exist before attempting to load.

    Args:
        model_dir: Directory path where artifacts are stored.
        model_filename: Filename for the model (default: "svd_model.joblib").
        user_mapping_filename: Filename for user mapping (default: "user_id_mapping.joblib").
        item_mapping_filename: Filename for item mapping (default: "product_id_mapping.joblib").

    Returns:
        A tuple containing Path objects for:
            - Model file path
            - User mapping file path
            - Item mapping file path
    """
    model_path = Path(model_dir)
    return (
        model_path / model_filename,
        model_path / user_mapping_filename,
        model_path / item_mapping_filename,
    )


def check_model_exists(model_dir: str) -> bool:
    """Check if all required model artifacts exist.

    Args:
        model_dir: Directory path where artifacts should be stored.

    Returns:
        True if all model files exist, False otherwise.
    """
    try:
        model_path, user_path, item_path = get_model_paths(model_dir)
        return all(path.exists() for path in [model_path, user_path, item_path])
    except Exception:
        return False

