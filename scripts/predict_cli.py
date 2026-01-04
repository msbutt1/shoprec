"""CLI script for getting product recommendations.

Useful for testing and evaluation. Gets recommendations for a user
and prints them to the console.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.recommender.hybrid import create_hybrid_recommender
from src.recommender.infer import _compute_recommendations, _handle_cold_start_user
from src.recommender.utils import load_model_artifacts

# Setup logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def get_recommendations(
    user_id: int,
    model_dir: str = "models",
    top_n: int = 10,
    mode: Literal["hybrid", "cf"] = "hybrid",
    explain: bool = False,
) -> tuple[List[int], Optional[Dict]]:
    """Get recommendations for a user.
    
    Args:
        user_id: User ID to get recommendations for
        model_dir: Directory with model files
        top_n: Number of recommendations to return
        mode: "hybrid" or "cf"
        explain: If True, also return score breakdown
        
    Returns:
        Tuple of (recommendations list, optional scores dict)
    """
    try:
        # Load model
        model, user_map, product_map = load_model_artifacts(model_dir)
        
        if mode == "hybrid":
            # Try hybrid mode
            try:
                hybrid_recommender = create_hybrid_recommender(
                    model=model,
                    user_id_to_idx=user_map,
                    product_id_to_idx=product_map,
                    model_dir=model_dir,
                )
                
                if explain:
                    recommendations, scores = hybrid_recommender.recommend(
                        user_id=user_id,
                        top_n=top_n,
                        user_product_matrix=None,
                        return_scores=True,
                    )
                    return recommendations, scores
                else:
                    recommendations = hybrid_recommender.recommend(
                        user_id=user_id,
                        top_n=top_n,
                        user_product_matrix=None,
                        return_scores=False,
                    )
                    return recommendations, None
            except Exception as e:
                logger.warning(f"Hybrid mode failed: {e}, falling back to CF")
                mode = "cf"
        
        if mode == "cf":
            # CF-only mode
            if user_id not in user_map:
                logger.warning(f"User {user_id} not in training data, using cold-start")
                recommendations = _handle_cold_start_user(user_id, product_map, top_n)
                scores = {"method": "cold_start"} if explain else None
                return recommendations, scores
            else:
                user_idx = user_map[user_id]
                recommendations = _compute_recommendations(
                    user_idx=user_idx,
                    model=model,
                    product_id_to_idx=product_map,
                    top_n=top_n,
                    user_product_matrix=None,
                )
                scores = None
                if explain:
                    scores = {"method": "cf_only", "recommendations": recommendations}
                return recommendations, scores
                
    except FileNotFoundError as e:
        print(f"Error: Model not found in {model_dir}", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Get product recommendations for a user",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/predict_cli.py 42
  python scripts/predict_cli.py 42 --top-n 5
  python scripts/predict_cli.py 42 --mode cf
  python scripts/predict_cli.py 42 --mode hybrid --explain
        """
    )
    
    parser.add_argument(
        "user_id",
        type=int,
        help="User ID to get recommendations for"
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of recommendations to return (default: 10)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hybrid", "cf"],
        default="hybrid",
        help="Recommendation mode: hybrid (CF + content) or cf (CF only) (default: hybrid)"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing model files (default: models)"
    )
    
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Show score breakdown for recommendations"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    # Get recommendations
    recommendations, scores = get_recommendations(
        user_id=args.user_id,
        model_dir=args.model_dir,
        top_n=args.top_n,
        mode=args.mode,
        explain=args.explain,
    )
    
    # Print results
    print(f"\nRecommendations for user {args.user_id} (mode: {args.mode}):")
    print(f"  Top {len(recommendations)} products: {recommendations}")
    
    if args.explain and scores:
        print(f"\nScore breakdown:")
        if "method" in scores:
            print(f"  Method: {scores['method']}")
        if "cf_scores" in scores and scores["cf_scores"]:
            print(f"  CF scores: {scores['cf_scores']}")
        if "content_scores" in scores and scores["content_scores"]:
            print(f"  Content scores: {scores['content_scores']}")
        if "hybrid_scores" in scores and scores["hybrid_scores"]:
            print(f"  Hybrid scores: {scores['hybrid_scores']}")
    
    print()


if __name__ == "__main__":
    main()

