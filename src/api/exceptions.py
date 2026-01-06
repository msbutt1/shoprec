"""Custom exceptions for the ShopRec API.

Defines specific exception types for better error handling and reporting.
"""

from typing import Any, Dict, Optional


class ShopRecException(Exception):
    """Base exception for ShopRec errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize exception.
        
        Args:
            message: Human-readable error message
            status_code: HTTP status code for API responses
            details: Additional error details for debugging
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class ModelNotFoundError(ShopRecException):
    """Raised when model files cannot be found."""
    
    def __init__(self, model_path: str, details: Optional[Dict[str, Any]] = None):
        message = f"Model not found at '{model_path}'. Please train a model first."
        super().__init__(
            message=message,
            status_code=503,
            details=details or {"model_path": model_path},
        )


class UserNotFoundError(ShopRecException):
    """Raised when a user is not found in the training data."""
    
    def __init__(self, user_id: int, details: Optional[Dict[str, Any]] = None):
        message = (
            f"User {user_id} not found in training data. "
            "Cannot generate personalized recommendations."
        )
        super().__init__(
            message=message,
            status_code=404,
            details=details or {"user_id": user_id},
        )


class ModelLoadError(ShopRecException):
    """Raised when model fails to load."""
    
    def __init__(self, model_path: str, error: Exception):
        message = f"Failed to load model from '{model_path}': {str(error)}"
        super().__init__(
            message=message,
            status_code=500,
            details={
                "model_path": model_path,
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )


class RecommendationError(ShopRecException):
    """Raised when recommendation generation fails."""
    
    def __init__(self, user_id: int, error: Exception):
        message = f"Failed to generate recommendations for user {user_id}: {str(error)}"
        super().__init__(
            message=message,
            status_code=500,
            details={
                "user_id": user_id,
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

