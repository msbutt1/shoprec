"""FastAPI application main module.

This module defines the main FastAPI application instance and core API endpoints
for the ShopRec recommendation service. It provides health check endpoints and
serves as the entry point for the API server.
"""

from typing import Dict

from fastapi import FastAPI

from src.api.routes import recommend

# Create FastAPI application instance
app = FastAPI(
    title="ShopRec API",
    description="Commerce-scale AI product recommendation service",
    version="0.1.0",
)

# Include routers
app.include_router(recommend.router)


@app.get("/ping")
def ping() -> Dict[str, str]:
    """Health check endpoint.

    Returns a simple status response to verify the API is running.

    Returns:
        Dictionary with status key set to "ok".

    Example:
        >>> response = client.get("/ping")
        >>> assert response.json() == {"status": "ok"}
    """
    return {"status": "ok"}


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

