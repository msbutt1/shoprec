"""Tests for the inference module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.recommender.infer import recommend_products_for_user
from src.recommender.train import train_basic_cf_model


@pytest.fixture
def trained_model(tmp_path):
    """Create a trained model for testing."""
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

    model_dir = tmp_path / "model"
    train_basic_cf_model(
        csv_path=str(csv_path),
        output_dir=str(model_dir),
        n_components=5,
        n_iter=5,
        random_state=42,
    )

    return model_dir


def test_recommend_products_for_user_known_user_returns_n_items(trained_model):
    """Test that recommend_products_for_user returns N items for a known user."""
    user_id = 5
    top_n = 5

    recommendations = recommend_products_for_user(
        user_id=user_id,
        model_path=str(trained_model),
        top_n=top_n,
    )

    assert isinstance(recommendations, list)
    assert len(recommendations) == top_n
    assert all(isinstance(rec, int) for rec in recommendations)
    assert len(recommendations) == len(set(recommendations))


def test_recommend_products_for_user_unknown_user_returns_cold_start(trained_model):
    """Test that unknown users get cold-start recommendations."""
    user_id = 999
    top_n = 5

    recommendations = recommend_products_for_user(
        user_id=user_id,
        model_path=str(trained_model),
        top_n=top_n,
    )

    assert isinstance(recommendations, list)
    assert len(recommendations) == top_n
    assert all(isinstance(rec, int) for rec in recommendations)


def test_recommend_products_for_user_missing_model_raises_filenotfounderror(tmp_path):
    """Test that missing model files raise FileNotFoundError."""
    user_id = 5
    nonexistent_model_dir = tmp_path / "nonexistent_model"

    with pytest.raises(FileNotFoundError):
        recommend_products_for_user(
            user_id=user_id,
            model_path=str(nonexistent_model_dir),
            top_n=5,
        )
