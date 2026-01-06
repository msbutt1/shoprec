"""Main FastAPI app.

Sets up the API server and endpoints.
"""

import logging
import traceback
from typing import Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.logging_config import RequestLoggingMiddleware, setup_logging
from src.api.metrics import metrics_service
from src.api.routes import recommend
from src.api.routes.recommend import get_model_status, ModelNotFoundError, UserNotFoundError

# Setup structured logging
setup_logging(log_level="INFO")

# Get logger
logger = logging.getLogger(__name__)

# Create FastAPI application instance
app = FastAPI(
    title="ShopRec API",
    description="Commerce-scale AI product recommendation service",
    version="0.1.0",
)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Include routers
app.include_router(recommend.router)


@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request: Request, exc: ModelNotFoundError) -> JSONResponse:
    """Handle model not found errors.
    
    Returns 503 Service Unavailable when model files are missing.
    """
    logger.error(
        "Model not found",
        extra={
            "path": str(request.url.path),
            "method": request.method,
            "error": str(exc),
        },
        exc_info=True,
    )

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Model not found",
            "message": str(exc),
            "details": "Please train a model first using the training script.",
        },
    )


@app.exception_handler(UserNotFoundError)
async def user_not_found_handler(request: Request, exc: UserNotFoundError) -> JSONResponse:
    """Handle user not found errors.
    
    Returns 404 Not Found when user is not in training data.
    """
    logger.warning(
        "User not found",
        extra={
            "path": str(request.url.path),
            "method": request.method,
            "error": str(exc),
        }
    )

    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "User not found",
            "message": str(exc),
            "details": "User not found in training data. Enable cold-start mode for fallback recommendations.",
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions.
    
    Provides consistent error format for all HTTP exceptions.
    """
    log_level = logging.ERROR if exc.status_code >= 500 else logging.WARNING
    logger.log(
        log_level,
        "HTTP exception",
        extra={
            "path": str(request.url.path),
            "method": request.method,
            "status_code": exc.status_code,
            "detail": exc.detail,
        },
        exc_info=exc.status_code >= 500,
    )

    # Handle detail that's already a dict
    if isinstance(exc.detail, dict):
        content = exc.detail
    else:
        content = {
            "error": "HTTP error",
            "message": str(exc.detail),
            "status_code": exc.status_code,
        }

    return JSONResponse(
        status_code=exc.status_code,
        content=content,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all unhandled exceptions.
    
    Logs full traceback and returns 500 Internal Server Error.
    """
    # Get full traceback
    tb_str = traceback.format_exc()
    
    logger.error(
        "Unhandled exception",
        extra={
            "path": str(request.url.path),
            "method": request.method,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": tb_str,
        },
        exc_info=True,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "type": type(exc).__name__,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors.
    
    Returns 422 Unprocessable Entity with validation error details.
    """
    logger.warning(
        "Request validation error",
        extra={
            "path": str(request.url.path),
            "method": request.method,
            "errors": exc.errors(),
        },
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "message": "Request validation failed",
            "details": exc.errors(),
        },
    )


@app.on_event("startup")
async def startup_event() -> None:
    """Log application startup."""
    logger.info("ShopRec API starting up", extra={"version": "0.1.0"})


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Log application shutdown."""
    logger.info("ShopRec API shutting down")


@app.get("/ping")
def ping() -> Dict[str, str]:
    """Health check endpoint.
    """
    return {"status": "ok"}


@app.get("/status")
def get_status() -> Dict:
    """Get model status.
    
    Returns if model is loaded, when it was loaded, and counts.
    """
    return get_model_status()


@app.get("/metrics")
def get_metrics() -> Dict:
    """Get API performance metrics.
    
    Returns:
        Dictionary with metrics:
        - inference_count: Total number of inference calls served
        - average_latency_ms: Average latency in milliseconds
        - min_latency_ms: Minimum latency observed
        - max_latency_ms: Maximum latency observed
    """
    return metrics_service.get_metrics()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import uvicorn

    # Add project root to Python path for imports
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
