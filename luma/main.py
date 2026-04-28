"""
Luma AI System - Application Entry Point

This module initializes the FastAPI application and configures all system components.
It serves as the main entry point for the Luma personal AI system.

Run with: uvicorn luma.main:app --reload
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from luma.config import settings
from luma.database import Base, engine, get_db
from luma.api.routes import router as api_router
from luma.api.middleware.logging import LoggingMiddleware
from luma.api.middleware.error_handler import ErrorHandlerMiddleware
from luma.utils.logger import setup_logging, get_logger


# Initialize logging
setup_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    Handles:
    - Database table creation on startup
    - Resource cleanup on shutdown
    """
    # Startup
    logger.info("Starting Luma AI System...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Luma AI System...")
    engine.dispose()
    logger.info("Database connections closed")


def create_app() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title="Luma AI System",
        description="A local-first personal AI system with memory, reasoning, and agent capabilities",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Error handling and logging middleware
    # ErrorHandlerMiddleware is added last (outermost) so it wraps LoggingMiddleware
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(LoggingMiddleware)

    # Include API routes
    app.include_router(api_router, prefix=settings.api_prefix)
    
    @app.get("/")
    async def root():
        """Root endpoint to verify system is running."""
        return {"message": "Luma is alive"}
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "version": "0.1.0"}
    
    logger.info(f"FastAPI application created with prefix: {settings.api_prefix}")
    return app


# Create application instance
app = create_app()
