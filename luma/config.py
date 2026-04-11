"""
Configuration Management for Luma AI System

This module provides centralized configuration using Pydantic Settings.
Settings can be loaded from environment variables or .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    """
    Application configuration settings.
    
    All settings have sensible defaults for local development.
    Override via environment variables or .env file.
    """
    
    # Database configuration
    database_url: str = "sqlite:///./luma.db"
    
    # API configuration
    api_prefix: str = "/api/v1"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Logging configuration
    log_level: str = "INFO"
    
    # Environment
    environment: str = "development"
    
    # Ranking Engine configuration
    ranking_alpha: float = 0.5  # Similarity weight
    ranking_beta: float = 0.3   # Recency weight
    ranking_gamma: float = 0.2  # Importance weight
    ranking_decay_constant: float = 0.0001  # Time decay rate (λ)
    ranking_similarity_threshold: float = 0.3  # Minimum similarity score
    ranking_score_threshold: float = 0.2  # Minimum final score
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate that log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(
                f"log_level must be one of {valid_levels}, got '{v}'"
            )
        return v_upper
    
    @field_validator("api_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate that port is in valid range."""
        if not 1 <= v <= 65535:
            raise ValueError(f"api_port must be between 1 and 65535, got {v}")
        return v
    
    @field_validator("ranking_alpha", "ranking_beta", "ranking_gamma")
    @classmethod
    def validate_ranking_weights(cls, v: float) -> float:
        """Validate that ranking weights are non-negative."""
        if v < 0:
            raise ValueError(f"Ranking weight must be non-negative, got {v}")
        return v
    
    @field_validator("ranking_decay_constant")
    @classmethod
    def validate_decay_constant(cls, v: float) -> float:
        """Validate that decay constant is positive."""
        if v <= 0:
            raise ValueError(f"ranking_decay_constant must be positive, got {v}")
        return v
    
    @field_validator("ranking_similarity_threshold", "ranking_score_threshold")
    @classmethod
    def validate_ranking_thresholds(cls, v: float) -> float:
        """Validate that ranking thresholds are in [0, 1]."""
        if not 0 <= v <= 1:
            raise ValueError(f"Ranking threshold must be in [0, 1], got {v}")
        return v


# Global settings instance
settings = Settings()
