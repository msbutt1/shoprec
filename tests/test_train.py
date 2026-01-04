"""Tests for the collaborative filtering model training module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sklearn.decomposition import TruncatedSVD

from src.recommender.train import train_basic_cf_model


@pytest.fixture
def fake_purchase_data(tmp_path):
    """Generate fake purchase data CSV for testing."""
    import random
    from datetime import datetime, timedelta

    random.seed(42)

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
    csv_path = tmp_path / "test_purchases.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_train_basic_cf_model_creates_artifacts(fake_purchase_data, tmp_path):
    """Test that train_basic_cf_model creates all required artifacts."""
    model_dir = tmp_path / "model"
    
    model, user_map, product_map = train_basic_cf_model(
        csv_path=str(fake_purchase_data),
        output_dir=str(model_dir),
        n_components=5,
        n_iter=5,
        random_state=42,
    )

    assert isinstance(model, TruncatedSVD)
    assert model.n_components == 5
    assert isinstance(user_map, dict)
    assert isinstance(product_map, dict)
    assert len(user_map) > 0
    assert len(product_map) > 0
    
    # Check files exist
    assert (model_dir / "svd_model.joblib").exists()
    assert (model_dir / "user_id_mapping.joblib").exists()
    assert (model_dir / "product_id_mapping.joblib").exists()
