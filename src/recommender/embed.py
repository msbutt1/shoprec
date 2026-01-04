"""Product embeddings for content-based recommendations.

Makes embeddings from product data using TF-IDF or random vectors.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
EMBEDDINGS_FILENAME = "product_embeddings.joblib"
EMBEDDING_METADATA_FILENAME = "embedding_metadata.joblib"
DEFAULT_EMBEDDING_DIM = 50
DEFAULT_RANDOM_SEED = 42


class ProductEmbeddings:
    """Holds product embeddings.
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        product_id_to_idx: Dict[int, int],
        method: str = "random",
        vectorizer: Optional[TfidfVectorizer] = None,
    ):
        """Initialize.
        """
        self.embeddings = embeddings
        self.product_id_to_idx = product_id_to_idx
        self.method = method
        self.vectorizer = vectorizer
        self.idx_to_product_id = {idx: pid for pid, idx in product_id_to_idx.items()}
        
        logger.info(
            f"Initialized ProductEmbeddings: {len(product_id_to_idx)} products, "
            f"embedding_dim={embeddings.shape[1]}, method={method}"
        )
    
    def get_embedding(self, product_id: int) -> Optional[np.ndarray]:
        """Get embedding for a product.
        """
        if product_id not in self.product_id_to_idx:
            return None
        idx = self.product_id_to_idx[product_id]
        return self.embeddings[idx]
    
    def compute_similarity(
        self,
        product_id1: int,
        product_id2: int
    ) -> Optional[float]:
        """Get similarity between two products.
        """
        emb1 = self.get_embedding(product_id1)
        emb2 = self.get_embedding(product_id2)
        
        if emb1 is None or emb2 is None:
            return None
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def get_similar_products(
        self,
        product_id: int,
        top_n: int = 10,
        exclude_ids: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """Find similar products.
        """
        emb = self.get_embedding(product_id)
        if emb is None:
            logger.warning(f"Product {product_id} not found in embeddings")
            return []
        
        # Compute similarities with all products
        similarities = cosine_similarity([emb], self.embeddings)[0]
        
        # Exclude specified products
        exclude_ids = exclude_ids or []
        exclude_indices = [
            self.product_id_to_idx[pid]
            for pid in exclude_ids
            if pid in self.product_id_to_idx
        ]
        
        # Also exclude the product itself
        if product_id in self.product_id_to_idx:
            exclude_indices.append(self.product_id_to_idx[product_id])
        
        # Set excluded similarities to -inf
        for idx in exclude_indices:
            similarities[idx] = -np.inf
        
        # Get top N
        top_indices = np.argsort(similarities)[::-1][:top_n]
        results = [
            (int(self.idx_to_product_id[int(idx)]), float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] != -np.inf
        ]
        
        return results
    
    def compute_user_content_scores(
        self,
        purchased_product_ids: List[int],
        all_product_ids: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """Get content scores for products based on what user bought.
        """
        if not purchased_product_ids:
            logger.warning("No purchased products provided")
            return {}
        
        # Get embeddings for stuff user bought
        purchased_embeddings = []
        for pid in purchased_product_ids:
            emb = self.get_embedding(pid)
            if emb is not None:
                purchased_embeddings.append(emb)
        
        if not purchased_embeddings:
            logger.warning("No valid embeddings found for purchased products")
            return {}
        
        # Average the embeddings to make a user profile
        user_profile = np.mean(purchased_embeddings, axis=0)
        
        # Figure out which products to score
        if all_product_ids is None:
            all_product_ids = list(self.product_id_to_idx.keys())
        
        # Calculate similarity for each product
        scores = {}
        for pid in all_product_ids:
            emb = self.get_embedding(pid)
            if emb is not None:
                similarity = np.dot(user_profile, emb) / (
                    np.linalg.norm(user_profile) * np.linalg.norm(emb)
                )
                scores[pid] = float(similarity)
        
        return scores


def generate_random_embeddings(
    product_ids: List[int],
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> ProductEmbeddings:
    """Make random embeddings for products.
    
    Good for testing when you don't have real product data.
    """
    logger.info(
        f"Generating random embeddings for {len(product_ids)} products, "
        f"dim={embedding_dim}, seed={random_seed}"
    )
    
    np.random.seed(random_seed)
    
    # Generate random vectors
    embeddings = np.random.randn(len(product_ids), embedding_dim)
    
    # Normalize to unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    # Create ID mapping
    product_id_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}
    
    return ProductEmbeddings(
        embeddings=embeddings,
        product_id_to_idx=product_id_to_idx,
        method="random",
    )


def generate_tfidf_embeddings(
    product_metadata: Dict[int, str],
    max_features: int = DEFAULT_EMBEDDING_DIM,
    min_df: int = 1,
    max_df: float = 1.0,
) -> ProductEmbeddings:
    """Make TF-IDF embeddings from product text descriptions.
    """
    logger.info(
        f"Generating TF-IDF embeddings for {len(product_metadata)} products, "
        f"max_features={max_features}"
    )
    
    # Sort by product ID for consistent ordering
    product_ids = sorted(product_metadata.keys())
    documents = [product_metadata[pid] for pid in product_ids]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        lowercase=True,
        stop_words='english',
    )
    
    # Generate embeddings
    embeddings = vectorizer.fit_transform(documents).toarray()
    
    # Create ID mapping
    product_id_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}
    
    logger.info(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return ProductEmbeddings(
        embeddings=embeddings,
        product_id_to_idx=product_id_to_idx,
        method="tfidf",
        vectorizer=vectorizer,
    )


def generate_simulated_product_metadata(
    product_ids: List[int],
    categories: Optional[List[str]] = None,
    attributes: Optional[List[str]] = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[int, str]:
    """Generate simulated product metadata for testing.
    
    Creates fake product descriptions by randomly combining categories and attributes.
    
    Args:
        product_ids: List of product IDs.
        categories: Optional list of categories to sample from.
        attributes: Optional list of attributes to sample from.
        random_seed: Random seed for reproducibility.
        
    Returns:
        Dictionary mapping product IDs to simulated descriptions.
    """
    if categories is None:
        categories = [
            "electronics", "clothing", "home", "sports", "toys",
            "books", "food", "beauty", "automotive", "garden"
        ]
    
    if attributes is None:
        attributes = [
            "premium", "budget", "eco-friendly", "durable", "portable",
            "stylish", "compact", "professional", "casual", "luxury",
            "practical", "innovative", "classic", "modern", "vintage"
        ]
    
    np.random.seed(random_seed)
    
    metadata = {}
    for pid in product_ids:
        # Sample 1-2 categories and 2-4 attributes
        n_categories = np.random.randint(1, 3)
        n_attributes = np.random.randint(2, 5)
        
        selected_categories = np.random.choice(categories, n_categories, replace=False)
        selected_attributes = np.random.choice(attributes, n_attributes, replace=False)
        
        description = " ".join(list(selected_categories) + list(selected_attributes))
        metadata[pid] = description
    
    logger.info(f"Generated simulated metadata for {len(product_ids)} products")
    return metadata


def save_embeddings(
    embeddings: ProductEmbeddings,
    output_dir: str
) -> None:
    """Save product embeddings to disk.
    
    Args:
        embeddings: ProductEmbeddings object to save.
        output_dir: Directory to save embeddings to.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    embeddings_file = output_path / EMBEDDINGS_FILENAME
    metadata_file = output_path / EMBEDDING_METADATA_FILENAME
    
    # Save embeddings matrix and ID mapping
    joblib.dump(
        {
            "embeddings": embeddings.embeddings,
            "product_id_to_idx": embeddings.product_id_to_idx,
        },
        embeddings_file
    )
    
    # Save metadata (method, vectorizer, etc.)
    metadata = {
        "method": embeddings.method,
        "embedding_dim": embeddings.embeddings.shape[1],
        "n_products": len(embeddings.product_id_to_idx),
        "vectorizer": embeddings.vectorizer if embeddings.method == "tfidf" else None,
    }
    joblib.dump(metadata, metadata_file)
    
    logger.info(f"Saved embeddings to {output_dir}")


def load_embeddings(model_dir: str) -> Optional[ProductEmbeddings]:
    """Load product embeddings from disk.
    
    Args:
        model_dir: Directory containing saved embeddings.
        
    Returns:
        ProductEmbeddings object, or None if not found.
    """
    embeddings_file = Path(model_dir) / EMBEDDINGS_FILENAME
    metadata_file = Path(model_dir) / EMBEDDING_METADATA_FILENAME
    
    if not embeddings_file.exists() or not metadata_file.exists():
        logger.warning(f"Embeddings not found in {model_dir}")
        return None
    
    # Load embeddings and ID mapping
    data = joblib.load(embeddings_file)
    embeddings_matrix = data["embeddings"]
    product_id_to_idx = data["product_id_to_idx"]
    
    # Load metadata
    metadata = joblib.load(metadata_file)
    method = metadata["method"]
    vectorizer = metadata.get("vectorizer")
    
    logger.info(
        f"Loaded embeddings from {model_dir}: "
        f"{len(product_id_to_idx)} products, "
        f"dim={embeddings_matrix.shape[1]}, method={method}"
    )
    
    return ProductEmbeddings(
        embeddings=embeddings_matrix,
        product_id_to_idx=product_id_to_idx,
        method=method,
        vectorizer=vectorizer,
    )


def check_embeddings_exist(model_dir: str) -> bool:
    """Check if embeddings exist in the specified directory.
    
    Args:
        model_dir: Directory to check for embeddings.
        
    Returns:
        True if embeddings exist, False otherwise.
    """
    embeddings_file = Path(model_dir) / EMBEDDINGS_FILENAME
    metadata_file = Path(model_dir) / EMBEDDING_METADATA_FILENAME
    return embeddings_file.exists() and metadata_file.exists()

