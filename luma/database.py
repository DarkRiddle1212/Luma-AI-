"""
Database Configuration and Session Management

This module provides database setup and session management for the Luma system.
Separated from main.py to avoid circular imports.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from typing import Generator

from luma.config import settings


# SQLAlchemy setup
Base = declarative_base()
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency for FastAPI.
    
    Provides a database session for request handling with proper lifecycle management:
    - Creates session at request start
    - Yields session for use in route handlers
    - Commits ONLY on successful completion
    - Rolls back on ANY exception
    - Always closes session in finally block
    
    Repository methods do NOT commit - this function controls all commits.
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
