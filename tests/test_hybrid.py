"""Tests for hybrid recommendation system.

This module contains tests for product embeddings and hybrid recommendations
that combine collaborative filtering with content-based filtering.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.recommender.embed import (
    ProductEmbeddings,
    generate_random_embeddings,
    generate_simulated_product_metadata,
    generate_tfidf_embeddings,
    load_embeddings,
    save_embeddings,
)
from src.recommender.hybrid import HybridRecommender, create_hybrid_recommender
from src.recommender.train import train_with_config, TrainingConfig
# Import generate_fake_purchases directly
import random
from datetime import datetime, timedelta
import pandas as pd


def generate_fake_purchases(num_users=50, num_products=100, num_purchases=1000):
    """Generate fake purchase data for testing."""
    purchases = []
    start_date = datetime.now() - timedelta(days=90)
    end_date = datetime.now()
    time_between = end_date - start_date
    total_seconds_between = int(time_between.total_seconds())
    
    for _ in range(num_purchases):
        user_id = random.randint(1, num_users)
        product_id = random.randint(1, num_products)
        random_seconds_offset = random.randrange(total_seconds_between)
        timestamp = start_date + timedelta(seconds=random_seconds_offset)
        
        purchases.append({
            "user_id": user_id,
            "product_id": product_id,
            "timestamp": timestamp,
        })
    
    df = pd.DataFrame(purchases)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


@pytest.fixture
def sample_product_ids():
    """Fixture providing sample product IDs."""
    return [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]


@pytest.fixture
def random_embeddings(sample_product_ids):
    """Fixture providing random product embeddings."""
    return generate_random_embeddings(
        product_ids=sample_product_ids,
        embedding_dim=10,
        random_seed=42,
    )


@pytest.fixture
def tfidf_embeddings(sample_product_ids):
    """Fixture providing TF-IDF product embeddings."""
    metadata = generate_simulated_product_metadata(
        product_ids=sample_product_ids,
        random_seed=42,
    )
    return generate_tfidf_embeddings(
        product_metadata=metadata,
        max_features=20,
    )


@pytest.fixture(scope="module")
def trained_model_with_embeddings(tmp_path_factory):
    """Fixture to train a model with embeddings."""
    model_dir = tmp_path_factory.mktemp("hybrid_model")
    csv_path = tmp_path_factory.mktemp("hybrid_data") / "fake_purchases.csv"

    # Generate fake data
    df = generate_fake_purchases(num_users=20, num_products=30, num_purchases=100)
    df.to_csv(csv_path, index=False)

    # Train with embeddings
    config = TrainingConfig(
        csv_path=str(csv_path),
        output_dir=str(model_dir),
        n_components=10,
        generate_embeddings=True,
        embedding_method="random",
        embedding_dim=15,
        random_state=42,
    )
    
    train_with_config(config)
    yield model_dir


# ===== Embedding Tests =====


def test_random_embeddings_generation(sample_product_ids):
    """Test random embedding generation."""
    embeddings = generate_random_embeddings(
        product_ids=sample_product_ids,
        embedding_dim=10,
        random_seed=42,
    )
    
    assert embeddings.embeddings.shape == (len(sample_product_ids), 10)
    assert len(embeddings.product_id_to_idx) == len(sample_product_ids)
    assert embeddings.method == "random"
    
    # Check normalization
    norms = np.linalg.norm(embeddings.embeddings, axis=1)
    np.testing.assert_array_almost_equal(norms, np.ones(len(sample_product_ids)))


def test_tfidf_embeddings_generation(sample_product_ids):
    """Test TF-IDF embedding generation."""
    metadata = generate_simulated_product_metadata(
        product_ids=sample_product_ids,
        random_seed=42,
    )
    
    embeddings = generate_tfidf_embeddings(
        product_metadata=metadata,
        max_features=20,
    )
    
    assert embeddings.embeddings.shape[0] == len(sample_product_ids)
    assert embeddings.embeddings.shape[1] <= 20
    assert len(embeddings.product_id_to_idx) == len(sample_product_ids)
    assert embeddings.method == "tfidf"
    assert embeddings.vectorizer is not None


def test_get_embedding(random_embeddings):
    """Test getting embedding for a specific product."""
    product_id = 1
    embedding = random_embeddings.get_embedding(product_id)
    
    assert embedding is not None
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (10,)
    
    # Test non-existent product
    assert random_embeddings.get_embedding(999) is None


def test_compute_similarity(random_embeddings):
    """Test computing similarity between two products."""
    similarity = random_embeddings.compute_similarity(1, 2)
    
    assert similarity is not None
    assert isinstance(similarity, float)
    assert -1 <= similarity <= 1
    
    # Similarity with self should be 1.0
    self_similarity = random_embeddings.compute_similarity(1, 1)
    assert abs(self_similarity - 1.0) < 0.01
    
    # Test with non-existent product
    assert random_embeddings.compute_similarity(1, 999) is None


def test_get_similar_products(random_embeddings):
    """Test finding similar products."""
    product_id = 1
    similar_products = random_embeddings.get_similar_products(
        product_id=product_id,
        top_n=5,
    )
    
    assert isinstance(similar_products, list)
    assert len(similar_products) <= 5
    
    # Check format
    for pid, score in similar_products:
        assert isinstance(pid, int)
        assert isinstance(score, float)
        assert pid != product_id  # Should not include the product itself
        assert -1 <= score <= 1


def test_compute_user_content_scores(random_embeddings):
    """Test computing content scores based on user purchases."""
    purchased_products = [1, 2, 3]
    scores = random_embeddings.compute_user_content_scores(
        purchased_product_ids=purchased_products,
    )
    
    assert isinstance(scores, dict)
    assert len(scores) > 0
    
    # Check all product IDs have scores
    for pid in random_embeddings.product_id_to_idx.keys():
        assert pid in scores
        assert isinstance(scores[pid], float)


def test_save_and_load_embeddings(random_embeddings, tmp_path):
    """Test saving and loading embeddings."""
    output_dir = tmp_path / "embeddings"
    
    # Save
    save_embeddings(random_embeddings, str(output_dir))
    
    # Check files exist
    assert (output_dir / "product_embeddings.joblib").exists()
    assert (output_dir / "embedding_metadata.joblib").exists()
    
    # Load
    loaded_embeddings = load_embeddings(str(output_dir))
    
    assert loaded_embeddings is not None
    assert loaded_embeddings.method == random_embeddings.method
    np.testing.assert_array_almost_equal(
        loaded_embeddings.embeddings,
        random_embeddings.embeddings
    )
    assert loaded_embeddings.product_id_to_idx == random_embeddings.product_id_to_idx


# ===== Hybrid Recommender Tests =====


def test_hybrid_recommender_initialization(random_embeddings):
    """Test HybridRecommender initialization."""
    # Create dummy SVD model
    model = TruncatedSVD(n_components=5)
    user_map = {1: 0, 2: 1, 3: 2}
    product_map = {i: i for i in range(10)}
    
    recommender = HybridRecommender(
        model=model,
        user_id_to_idx=user_map,
        product_id_to_idx=product_map,
        embeddings=random_embeddings,
        cf_weight=0.7,
        content_weight=0.3,
    )
    
    assert recommender.cf_weight == 0.7
    assert recommender.content_weight == 0.3
    assert recommender.embeddings is not None


def test_hybrid_recommender_weight_normalization():
    """Test that weights are normalized correctly."""
    model = TruncatedSVD(n_components=5)
    user_map = {1: 0}
    product_map = {1: 0}
    
    # Test with unnormalized weights
    recommender = HybridRecommender(
        model=model,
        user_id_to_idx=user_map,
        product_id_to_idx=product_map,
        embeddings=None,
        cf_weight=7,
        content_weight=3,
    )
    
    # Should be normalized to 0.7 and 0.3
    assert abs(recommender.cf_weight - 0.7) < 0.01
    assert abs(recommender.content_weight - 0.3) < 0.01


def test_hybrid_recommender_without_embeddings():
    """Test HybridRecommender falls back to CF only without embeddings."""
    model = TruncatedSVD(n_components=5)
    user_map = {1: 0, 2: 1}
    product_map = {i: i for i in range(10)}
    
    recommender = HybridRecommender(
        model=model,
        user_id_to_idx=user_map,
        product_id_to_idx=product_map,
        embeddings=None,  # No embeddings
        cf_weight=0.7,
        content_weight=0.3,
    )
    
    assert recommender.embeddings is None


def test_trained_model_has_embeddings(trained_model_with_embeddings):
    """Test that training generates and saves embeddings."""
    embeddings = load_embeddings(str(trained_model_with_embeddings))
    
    assert embeddings is not None
    assert embeddings.method == "random"
    assert embeddings.embeddings.shape[1] == 15


def test_hybrid_recommendation_with_trained_model(trained_model_with_embeddings):
    """Test generating hybrid recommendations with trained model."""
    from src.recommender.utils import load_model_artifacts
    
    # Load model artifacts
    model, user_map, product_map = load_model_artifacts(str(trained_model_with_embeddings))
    
    # Create hybrid recommender
    recommender = create_hybrid_recommender(
        model=model,
        user_id_to_idx=user_map,
        product_id_to_idx=product_map,
        model_dir=str(trained_model_with_embeddings),
        cf_weight=0.7,
        content_weight=0.3,
    )
    
    # Get recommendations for a known user
    user_id = list(user_map.keys())[0]
    recommendations = recommender.recommend(
        user_id=user_id,
        top_n=5,
    )
    
    assert isinstance(recommendations, list)
    assert len(recommendations) <= 5
    assert all(isinstance(pid, int) for pid in recommendations)


def test_hybrid_recommendation_with_explainability(trained_model_with_embeddings):
    """Test hybrid recommendations with score breakdown."""
    from src.recommender.utils import load_model_artifacts
    
    model, user_map, product_map = load_model_artifacts(str(trained_model_with_embeddings))
    
    recommender = create_hybrid_recommender(
        model=model,
        user_id_to_idx=user_map,
        product_id_to_idx=product_map,
        model_dir=str(trained_model_with_embeddings),
    )
    
    user_id = list(user_map.keys())[0]
    recommendations, scores = recommender.recommend(
        user_id=user_id,
        top_n=3,
        return_scores=True,
    )
    
    assert isinstance(scores, dict)
    assert "cf_scores" in scores
    assert "content_scores" in scores
    assert "hybrid_scores" in scores
    assert "cf_weight" in scores
    assert "content_weight" in scores
    
    # Check that all recommended products have scores
    for pid in recommendations:
        assert pid in scores["hybrid_scores"]


def test_hybrid_cold_start_user(trained_model_with_embeddings):
    """Test hybrid recommendations for cold-start users."""
    from src.recommender.utils import load_model_artifacts
    
    model, user_map, product_map = load_model_artifacts(str(trained_model_with_embeddings))
    
    recommender = create_hybrid_recommender(
        model=model,
        user_id_to_idx=user_map,
        product_id_to_idx=product_map,
        model_dir=str(trained_model_with_embeddings),
    )
    
    # Unknown user
    unknown_user_id = 99999
    recommendations = recommender.recommend(
        user_id=unknown_user_id,
        top_n=5,
    )
    
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0  # Should get cold-start recommendations


def test_score_normalization():
    """Test score normalization function."""
    model = TruncatedSVD(n_components=5)
    user_map = {1: 0}
    product_map = {1: 0, 2: 1, 3: 2}
    
    recommender = HybridRecommender(
        model=model,
        user_id_to_idx=user_map,
        product_id_to_idx=product_map,
    )
    
    # Test normalization
    scores = {1: 10.0, 2: 20.0, 3: 30.0}
    normalized = recommender._normalize_scores(scores)
    
    assert normalized[1] == 0.0  # Min score
    assert normalized[3] == 1.0  # Max score
    assert 0.0 < normalized[2] < 1.0  # Middle score
    
    # Test with equal scores
    equal_scores = {1: 5.0, 2: 5.0, 3: 5.0}
    normalized_equal = recommender._normalize_scores(equal_scores)
    assert all(score == 0.5 for score in normalized_equal.values())


# ===== Integration Tests =====


def test_cli_training_with_embeddings(tmp_path):
    """Test CLI training script generates embeddings."""
    model_dir = tmp_path / "cli_model"
    csv_path = tmp_path / "fake_purchases.csv"
    
    # Generate fake data
    df = generate_fake_purchases(num_users=10, num_products=15, num_purchases=50)
    df.to_csv(csv_path, index=False)
    
    # Train with embeddings
    config = TrainingConfig(
        csv_path=str(csv_path),
        output_dir=str(model_dir),
        n_components=5,
        generate_embeddings=True,
        embedding_method="tfidf",
        embedding_dim=10,
    )
    
    train_with_config(config)
    
    # Check embeddings were created
    embeddings = load_embeddings(str(model_dir))
    assert embeddings is not None
    assert embeddings.method == "tfidf"

