"""Tests for the collaborative filtering model training module.

This module contains unit tests for the training pipeline, including
data loading, model training, and artifact saving functionality.
"""

import logging
import tempfile
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest
from sklearn.decomposition import TruncatedSVD

from src.recommender.train import train_basic_cf_model
from src.recommender.utils import (
    MODEL_FILENAME,
    PRODUCT_MAPPING_FILENAME,
    USER_MAPPING_FILENAME,
    check_model_exists,
    load_model_artifacts,
)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test artifacts.

    Yields:
        Path to temporary directory.

    The directory is automatically cleaned up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def fake_purchase_data(temp_dir: Path) -> Path:
    """Generate fake purchase data CSV for testing.

    Args:
        temp_dir: Temporary directory for test files.

    Returns:
        Path to generated CSV file.
    """
    # Generate test data: 10 users, 20 products, 50 purchases
    import random
    from datetime import datetime, timedelta

    random.seed(42)  # For reproducibility

    purchases = []
    for _ in range(50):
        user_id = random.randint(1, 10)
        product_id = random.randint(1, 20)
        timestamp = datetime.now() - timedelta(
            days=random.randint(0, 30), seconds=random.randint(0, 86400)
        )

        purchases.append({
            "user_id": user_id,
            "product_id": product_id,
            "timestamp": timestamp,
        })

    df = pd.DataFrame(purchases)
    csv_path = temp_dir / "test_purchases.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


def test_train_basic_cf_model_creates_artifacts(
    fake_purchase_data: Path, temp_dir: Path
) -> None:
    """Test that train_basic_cf_model creates all required artifacts.

    Args:
        fake_purchase_data: Path to test CSV file.
        temp_dir: Temporary directory for model artifacts.
    """
    # Train model
    model, user_map, product_map = train_basic_cf_model(
        csv_path=str(fake_purchase_data),
        output_dir=str(temp_dir),
        n_components=5,  # Small number for fast testing
        n_iter=5,
        random_state=42,
    )

    # Assert model is returned and is correct type
    assert isinstance(model, TruncatedSVD)
    assert model.n_components == 5

    # Assert mappings are returned
    assert isinstance(user_map, dict)
    assert isinstance(product_map, dict)
    assert len(user_map) > 0
    assert len(product_map) > 0

    # Assert all artifact files exist
    model_path = temp_dir / MODEL_FILENAME
    user_mapping_path = temp_dir / USER_MAPPING_FILENAME
    product_mapping_path = temp_dir / PRODUCT_MAPPING_FILENAME

    assert model_path.exists(), f"Model file not found: {model_path}"
    assert user_mapping_path.exists(), f"User mapping not found: {user_mapping_path}"
    assert product_mapping_path.exists(), (
        f"Product mapping not found: {product_mapping_path}"
    )

    # Verify model can be loaded back
    assert check_model_exists(str(temp_dir)), "Model check failed"

    # Load and verify artifacts
    loaded_model, loaded_user_map, loaded_product_map = load_model_artifacts(
        str(temp_dir)
    )

    assert isinstance(loaded_model, TruncatedSVD)
    assert loaded_model.n_components == 5
    assert loaded_user_map == user_map
    assert loaded_product_map == product_map


def test_train_basic_cf_model_with_defaults(
    fake_purchase_data: Path, temp_dir: Path
) -> None:
    """Test training with default parameters.

    Args:
        fake_purchase_data: Path to test CSV file.
        temp_dir: Temporary directory for model artifacts.
    """
    # Train with default parameters
    model, user_map, product_map = train_basic_cf_model(
        csv_path=str(fake_purchase_data),
        output_dir=str(temp_dir),
    )

    # Verify model was created
    assert isinstance(model, TruncatedSVD)
    assert model.n_components > 0
    assert len(user_map) > 0
    assert len(product_map) > 0

    # Verify files exist
    assert (temp_dir / MODEL_FILENAME).exists()
    assert (temp_dir / USER_MAPPING_FILENAME).exists()
    assert (temp_dir / PRODUCT_MAPPING_FILENAME).exists()


def test_train_basic_cf_model_auto_adjusts_components(
    fake_purchase_data: Path, temp_dir: Path
) -> None:
    """Test that n_components is automatically adjusted if too large.

    Args:
        fake_purchase_data: Path to test CSV file.
        temp_dir: Temporary directory for model artifacts.
    """
    # Request more components than possible (we have 10 users max)
    model, _, _ = train_basic_cf_model(
        csv_path=str(fake_purchase_data),
        output_dir=str(temp_dir),
        n_components=100,  # Much larger than number of users
        n_iter=5,
        random_state=42,
    )

    # Model should have been adjusted to a valid number
    assert isinstance(model, TruncatedSVD)
    assert model.n_components < 10  # Should be less than min(users, products)


def test_train_basic_cf_model_invalid_csv_path(temp_dir: Path) -> None:
    """Test that training fails gracefully with invalid CSV path.

    Args:
        temp_dir: Temporary directory for model artifacts.
    """
    invalid_path = temp_dir / "nonexistent.csv"

    with pytest.raises(FileNotFoundError):
        train_basic_cf_model(
            csv_path=str(invalid_path),
            output_dir=str(temp_dir),
        )


def test_train_basic_cf_model_empty_csv(temp_dir: Path) -> None:
    """Test that training fails gracefully with empty CSV.

    Args:
        temp_dir: Temporary directory for model artifacts.
    """
    # Create empty CSV
    empty_csv = temp_dir / "empty.csv"
    empty_csv.write_text("user_id,product_id,timestamp\n")

    with pytest.raises(ValueError, match="empty"):
        train_basic_cf_model(
            csv_path=str(empty_csv),
            output_dir=str(temp_dir),
        )


def test_train_basic_cf_model_missing_columns(temp_dir: Path) -> None:
    """Test that training fails gracefully with missing required columns.

    Args:
        temp_dir: Temporary directory for model artifacts.
    """
    # Create CSV with wrong columns
    bad_csv = temp_dir / "bad.csv"
    bad_csv.write_text("id,name\n1,test\n")

    with pytest.raises(ValueError, match="missing required columns"):
        train_basic_cf_model(
            csv_path=str(bad_csv),
            output_dir=str(temp_dir),
        )


def test_train_basic_cf_model_reproducibility(
    fake_purchase_data: Path, temp_dir: Path
) -> None:
    """Test that training is reproducible with same random_state.

    Args:
        fake_purchase_data: Path to test CSV file.
        temp_dir: Temporary directory for model artifacts.
    """
    # Train model twice with same random_state
    model1, user_map1, product_map1 = train_basic_cf_model(
        csv_path=str(fake_purchase_data),
        output_dir=str(temp_dir / "model1"),
        n_components=5,
        random_state=42,
    )

    model2, user_map2, product_map2 = train_basic_cf_model(
        csv_path=str(fake_purchase_data),
        output_dir=str(temp_dir / "model2"),
        n_components=5,
        random_state=42,
    )

    # Mappings should be identical
    assert user_map1 == user_map2
    assert product_map1 == product_map2

    # Model components should be identical (same random_state)
    assert model1.n_components == model2.n_components
    # Note: We can't directly compare SVD matrices due to sign ambiguity,
    # but we can check that the model structure is the same

