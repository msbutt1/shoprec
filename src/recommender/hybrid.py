"""Hybrid recommendation module.

Combines collaborative filtering and content-based filtering.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.recommender.embed import ProductEmbeddings, load_embeddings
from src.recommender.infer import _compute_recommendations, _handle_cold_start_user

# Configure module logger
logger = logging.getLogger(__name__)

# Default weights for hybrid scoring
DEFAULT_CF_WEIGHT = 0.7  # 70% collaborative filtering
DEFAULT_CONTENT_WEIGHT = 0.3  # 30% content-based
DEFAULT_TOP_N = 10


class HybridRecommender:
    """Combines CF and content-based recommendations.
    """
    
    def __init__(
        self,
        model: TruncatedSVD,
        user_id_to_idx: Dict[int, int],
        product_id_to_idx: Dict[int, int],
        embeddings: Optional[ProductEmbeddings] = None,
        cf_weight: float = DEFAULT_CF_WEIGHT,
        content_weight: float = DEFAULT_CONTENT_WEIGHT,
    ):
        """Initialize the recommender.
        """
        self.model = model
        self.user_id_to_idx = user_id_to_idx
        self.product_id_to_idx = product_id_to_idx
        self.embeddings = embeddings
        self.cf_weight = cf_weight
        self.content_weight = content_weight
        
        # Normalize weights
        total_weight = cf_weight + content_weight
        if total_weight > 0:
            self.cf_weight = cf_weight / total_weight
            self.content_weight = content_weight / total_weight
        
        logger.info(
            f"Initialized HybridRecommender: "
            f"CF weight={self.cf_weight:.2f}, "
            f"Content weight={self.content_weight:.2f}, "
            f"Embeddings={'enabled' if embeddings else 'disabled'}"
        )
    
    def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """Normalize scores to 0-1 range.
        """
        if not scores:
            return {}
        
        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)
        
        if max_score == min_score:
            # All scores are the same, return equal weights
            return {pid: 0.5 for pid in scores}
        
        return {
            pid: (score - min_score) / (max_score - min_score)
            for pid, score in scores.items()
        }
    
    def _get_cf_scores(
        self,
        user_id: int,
        user_product_matrix: Optional[csr_matrix] = None,
    ) -> Dict[int, float]:
        """Get CF scores for all products.
        """
        if user_id not in self.user_id_to_idx:
            logger.debug(f"User {user_id} not in CF model, returning empty scores")
            return {}
        
        user_idx = self.user_id_to_idx[user_id]
        n_products = len(self.product_id_to_idx)
        
        # Get user vector
        if user_product_matrix is not None:
            user_vector = user_product_matrix[user_idx].toarray().flatten()
        else:
            user_vector = np.zeros(n_products)
        
        # Transform to latent space and reconstruct scores
        if user_product_matrix is not None and np.any(user_vector):
            user_latent = self.model.transform([user_vector])[0]
            predicted_scores = np.dot(user_latent, self.model.components_)
        else:
            # No interactions, use average
            predicted_scores = np.mean(self.model.components_, axis=0)
        
        # Convert to dictionary
        idx_to_product_id = {idx: pid for pid, idx in self.product_id_to_idx.items()}
        cf_scores = {
            int(idx_to_product_id[idx]): float(predicted_scores[idx])
            for idx in range(len(predicted_scores))
        }
        
        return cf_scores
    
    def _get_content_scores(
        self,
        user_id: int,
        purchased_product_ids: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        """Get content-based scores for all products.
        """
        if self.embeddings is None:
            logger.debug("No embeddings available, returning empty content scores")
            return {}
        
        if not purchased_product_ids:
            logger.debug(f"No purchase history for user {user_id}, returning empty content scores")
            return {}
        
        # Get content scores based on user's purchase history
        content_scores = self.embeddings.compute_user_content_scores(
            purchased_product_ids=purchased_product_ids,
            all_product_ids=list(self.product_id_to_idx.keys()),
        )
        
        return content_scores
    
    def recommend(
        self,
        user_id: int,
        top_n: int = DEFAULT_TOP_N,
        user_product_matrix: Optional[csr_matrix] = None,
        return_scores: bool = False,
    ) -> List[int] | Tuple[List[int], Dict]:
        """Get recommendations for a user.
        
        Combines CF and content scores.
        """
        logger.info(f"Generating hybrid recommendations for user {user_id}, top_n={top_n}")
        
        # Check if user exists
        user_known = user_id in self.user_id_to_idx

        # Get what products the user already bought
        purchased_product_ids = []
        if user_known and user_product_matrix is not None:
            user_idx = self.user_id_to_idx[user_id]
            purchased_indices = user_product_matrix[user_idx].nonzero()[1].tolist()
            idx_to_product_id = {idx: pid for pid, idx in self.product_id_to_idx.items()}
            purchased_product_ids = [
                int(idx_to_product_id[idx]) for idx in purchased_indices
            ]

        # Handle new users
        if not user_known:
            logger.info(f"User {user_id} not found, using cold-start")
            recommendations = _handle_cold_start_user(
                user_id, self.product_id_to_idx, top_n
            )
            if return_scores:
                return recommendations, {"method": "cold_start", "cf_scores": {}, "content_scores": {}}
            return recommendations

        # Get scores from both methods
        cf_scores = self._get_cf_scores(user_id, user_product_matrix)
        content_scores = self._get_content_scores(user_id, purchased_product_ids)

        # Normalize them
        cf_scores_norm = self._normalize_scores(cf_scores)
        content_scores_norm = self._normalize_scores(content_scores)

        # Combine the scores
        all_product_ids = set(cf_scores_norm.keys()) | set(content_scores_norm.keys())
        hybrid_scores = {}

        for pid in all_product_ids:
            cf_score = cf_scores_norm.get(pid, 0.0)
            content_score = content_scores_norm.get(pid, 0.0)

            # Mix them together
            if self.embeddings is not None and content_scores:
                hybrid_score = (
                    self.cf_weight * cf_score +
                    self.content_weight * content_score
                )
            else:
                hybrid_score = cf_score

            hybrid_scores[pid] = hybrid_score

        # Don't recommend things they already bought
        for pid in purchased_product_ids:
            if pid in hybrid_scores:
                hybrid_scores[pid] = -np.inf

        # Get the top N products
        valid_scores = {
            pid: score for pid, score in hybrid_scores.items()
            if score != -np.inf
        }

        if not valid_scores:
            logger.warning(f"No valid products to recommend for user {user_id}")
            if return_scores:
                return [], {"cf_scores": cf_scores_norm, "content_scores": content_scores_norm}
            return []

        sorted_products = sorted(
            valid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        recommendations = [int(pid) for pid, _ in sorted_products[:top_n]]

        logger.info(f"Generated {len(recommendations)} hybrid recommendations for user {user_id}")

        if return_scores:
            score_breakdown = {
                "cf_scores": {pid: cf_scores_norm.get(pid, 0.0) for pid in recommendations},
                "content_scores": {pid: content_scores_norm.get(pid, 0.0) for pid in recommendations},
                "hybrid_scores": {pid: hybrid_scores.get(pid, 0.0) for pid in recommendations},
                "cf_weight": self.cf_weight,
                "content_weight": self.content_weight,
            }
            return recommendations, score_breakdown

        return recommendations


def create_hybrid_recommender(
    model: TruncatedSVD,
    user_id_to_idx: Dict[int, int],
    product_id_to_idx: Dict[int, int],
    model_dir: str,
    cf_weight: float = DEFAULT_CF_WEIGHT,
    content_weight: float = DEFAULT_CONTENT_WEIGHT,
) -> HybridRecommender:
    """Create a hybrid recommender from saved model files.
    """
    # Try to load embeddings
    embeddings = load_embeddings(model_dir)
    
    if embeddings is None:
        logger.warning(
            f"No embeddings found in {model_dir}. "
            "Hybrid recommender will fall back to CF only."
        )
    
    return HybridRecommender(
        model=model,
        user_id_to_idx=user_id_to_idx,
        product_id_to_idx=product_id_to_idx,
        embeddings=embeddings,
        cf_weight=cf_weight,
        content_weight=content_weight,
    )

