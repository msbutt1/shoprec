"""Main FastAPI app.

Sets up the API server and endpoints.
"""

import logging
from typing import Dict

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.logging_config import RequestLoggingMiddleware, setup_logging
from src.api.routes import recommend
from src.api.routes.recommend import get_model_status

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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle errors.
    """
    logger.error(
        "Unhandled exception",
        extra={
            "path": str(request.url.path),
            "method": request.method,
            "error": str(exc),
            "error_type": type(exc).__name__,
        },
        exc_info=True,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "type": type(exc).__name__,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle validation errors.
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
