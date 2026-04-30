"""
Dependency Injection Container Module

This module provides dependency wiring functions for initializing the Luma
application with all required components following clean architecture principles.

The container is responsible for:
1. Initializing storage layer (SQLiteStorage)
2. Initializing memory manager (MemoryManager)
3. Creating memory adapter (SQLiteMemoryAdapter)
4. Getting LLM implementation
5. Creating reasoning engine with dependencies
6. Verifying all dependencies are properly configured
"""

import logging
from pathlib import Path
from typing import Optional

from luma.core.reasoning import ReasoningEngine
from luma.core.llm_interface import LLMInterface, StubLLM
from luma.core.ranking_engine import RankingEngine, RankingConfig
from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.sqlite_storage import SQLiteStorage


logger = logging.getLogger(__name__)


def load_ranking_config_from_settings() -> dict:
    """
    Load ranking configuration from application settings.
    
    This function reads ranking parameters from the global settings
    instance (which can be configured via environment variables or .env file)
    and returns them as a dictionary suitable for passing to create_ranking_engine().
    
    Returns:
        dict: Ranking configuration parameters with keys:
            - alpha: Similarity weight
            - beta: Recency weight
            - gamma: Importance weight
            - decay_constant: Time decay rate
            - similarity_threshold: Minimum similarity score
            - score_threshold: Minimum final score
    
    Example:
        >>> from luma.config import settings
        >>> config = load_ranking_config_from_settings()
        >>> engine = create_ranking_engine(**config)
    """
    from luma.config import settings
    
    return {
        "alpha": settings.ranking_alpha,
        "beta": settings.ranking_beta,
        "gamma": settings.ranking_gamma,
        "decay_constant": settings.ranking_decay_constant,
        "similarity_threshold": settings.ranking_similarity_threshold,
        "score_threshold": settings.ranking_score_threshold,
    }


def create_ranking_engine(
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    decay_constant: float = 0.0001,
    similarity_threshold: float = 0.3,
    score_threshold: float = 0.2,
    namespace: Optional[str] = None
) -> RankingEngine:
    """
    Create and configure a RankingEngine instance.
    
    This function creates a RankingEngine with the specified configuration
    parameters. The default values provide a balanced configuration suitable
    for most use cases.
    
    Args:
        alpha: Similarity weight (default: 0.5). Must be non-negative.
        beta: Recency weight (default: 0.3). Must be non-negative.
        gamma: Importance weight (default: 0.2). Must be non-negative.
        decay_constant: Time decay rate λ (default: 0.0001). Must be positive.
                       Smaller values = slower decay, larger values = faster decay.
        similarity_threshold: Minimum similarity score (default: 0.3). Range [0, 1].
        score_threshold: Minimum final score (default: 0.2). Range [0, 1].
        namespace: Optional namespace for filtering memories (default: None).
    
    Returns:
        RankingEngine: Configured ranking engine instance
    
    Raises:
        ValueError: If configuration parameters are invalid
    
    Example:
        >>> # Create with default balanced configuration
        >>> engine = create_ranking_engine()
        >>> 
        >>> # Create with similarity-focused configuration
        >>> engine = create_ranking_engine(alpha=0.7, beta=0.3, gamma=0.0)
        >>> 
        >>> # Create with recency-focused configuration
        >>> engine = create_ranking_engine(alpha=0.3, beta=0.6, gamma=0.1)
    """
    logger.info("Creating RankingEngine with configuration:")
    logger.info(f"  Weights: α={alpha}, β={beta}, γ={gamma}")
    logger.info(f"  Decay constant: λ={decay_constant}")
    logger.info(f"  Thresholds: similarity={similarity_threshold}, score={score_threshold}")
    logger.info(f"  Namespace: {namespace}")
    
    config = RankingConfig(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        decay_constant=decay_constant,
        similarity_threshold=similarity_threshold,
        score_threshold=score_threshold,
        namespace=namespace
    )
    
    ranking_engine = RankingEngine(config)
    logger.info("RankingEngine created successfully")
    
    return ranking_engine


def initialize_application(
    db_path: str = "./data/luma_memory.db",
    llm: Optional[LLMInterface] = None,
    return_storage: bool = False,
    ranking_config: Optional[dict] = None
) -> ReasoningEngine:
    """
    Initialize application with dependency injection.
    
    Wires together all components following clean architecture principles.
    This function creates the complete dependency graph from storage layer
    up to the reasoning engine, ensuring all components are properly
    initialized and connected.
    
    Initialization Steps:
        1. Initialize SQLiteStorage with database path
        2. Initialize MemoryManager with storage
        3. Create SQLiteMemoryAdapter wrapping MemoryManager
        4. Get LLM implementation (use provided or default to StubLLM)
        5. Create RankingEngine with configuration
        6. Create ReasoningEngine with LLM and memory adapter
        7. Log each initialization step
    
    Args:
        db_path: Path to SQLite database file. Defaults to "./data/luma_memory.db".
                Directory will be created if it doesn't exist.
        llm: Optional LLM implementation. If None, uses StubLLM for testing.
        return_storage: If True, returns tuple of (engine, storage) for cleanup.
                       Defaults to False for backward compatibility.
        ranking_config: Optional dictionary with ranking configuration parameters.
                       Keys: alpha, beta, gamma, decay_constant, similarity_threshold,
                       score_threshold, namespace. If None, uses default values.
    
    Returns:
        ReasoningEngine: Fully configured reasoning engine with all dependencies
        Or tuple of (ReasoningEngine, SQLiteStorage) if return_storage=True
    
    Raises:
        Exception: If any initialization step fails
    
    Example:
        >>> # Initialize with default settings (StubLLM)
        >>> engine = initialize_application()
        >>> 
        >>> # Initialize with custom database path
        >>> engine = initialize_application(db_path="./custom/memory.db")
        >>> 
        >>> # Initialize with custom LLM
        >>> from luma.core.llm_interface import LLMInterface
        >>> class MyLLM(LLMInterface):
        ...     def generate_response(self, prompt: str, context: dict) -> str:
        ...         return "Custom response"
        >>> engine = initialize_application(llm=MyLLM())
        >>> 
        >>> # Initialize with custom ranking configuration
        >>> ranking_cfg = {
        ...     "alpha": 0.7,
        ...     "beta": 0.3,
        ...     "gamma": 0.0,
        ...     "decay_constant": 0.001,
        ...     "similarity_threshold": 0.5,
        ...     "score_threshold": 0.3
        ... }
        >>> engine = initialize_application(ranking_config=ranking_cfg)
    """
    logger.info("Starting application initialization...")
    
    storage = None
    try:
        # Step 1: Initialize storage layer
        logger.info(f"Initializing SQLiteStorage with db_path={db_path}")
        storage = SQLiteStorage(db_path=db_path)
        logger.info("SQLiteStorage initialized successfully")
        
        # Step 2: Initialize memory manager
        logger.info("Initializing MemoryManager with storage")
        memory_manager = MemoryManager(storage=storage)
        logger.info("MemoryManager initialized successfully")
        
        # Step 3: Create adapter
        logger.info("Creating SQLiteMemoryAdapter")
        memory_adapter = SQLiteMemoryAdapter(memory_manager=memory_manager)
        logger.info("SQLiteMemoryAdapter created successfully")
        
        # Step 4: Get LLM implementation
        if llm is None:
            try:
                logger.info("No LLM provided, initializing provider-based LLM from environment")
                
                # Load configuration from environment variables
                from luma.core.llm.config import load_llm_config_from_env
                llm_config = load_llm_config_from_env()
                
                # Create provider using factory
                from luma.core.llm.providers.provider_factory import ProviderFactory
                from luma.core.llm.llm_client import ProviderLLMClient
                from luma.core.llm.llm_client_adapter import LLMClientAdapter
                from luma.core.structured_logger import StructuredLogger
                
                provider = ProviderFactory.create(
                    provider_name=llm_config.provider_name,
                    config=llm_config.provider_config,
                    logger=StructuredLogger("provider_factory")
                )
                
                # Create LLM client with provider
                llm_client = ProviderLLMClient(
                    provider=provider,
                    config=llm_config,
                    logger=StructuredLogger("llm_client")
                )
                
                # Create adapter for ReasoningEngine
                llm = LLMClientAdapter(llm_client, llm_config)
                logger.info(f"Initialized provider-based LLM: {llm_config.provider_name}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize provider-based LLM: {e}, falling back to StubLLM")
                logger.warning("Check environment variables: LLM_PROVIDER, GEMINI_API_KEY, etc.")
                from luma.core.llm_interface import StubLLM
                llm = StubLLM()
        else:
            logger.info(f"Using provided LLM: {type(llm).__name__}")
        
        # Step 5: Create RankingEngine with configuration
        logger.info("Creating RankingEngine with configuration")
        if ranking_config is None:
            ranking_engine = create_ranking_engine()
        else:
            ranking_engine = create_ranking_engine(**ranking_config)
        logger.info("RankingEngine created successfully")
        
        # Step 6: Create reasoning engine with dependencies
        logger.info("Creating ReasoningEngine with LLM and memory adapter")
        reasoning_engine = ReasoningEngine(
            llm=llm,
            memory=memory_adapter
        )
        logger.info("ReasoningEngine created successfully")
        
        logger.info("Application initialized successfully")
        
        if return_storage:
            return reasoning_engine, storage
        return reasoning_engine
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}", exc_info=True)
        # Clean up storage if it was created
        if storage is not None:
            try:
                cleanup_application(storage)
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")
        raise


def verify_dependencies(reasoning_engine: ReasoningEngine) -> None:
    """
    Verify all required dependencies are properly configured.
    
    Checks that the reasoning engine has all required dependencies
    (LLM and memory) properly configured. Raises an error if LLM
    is missing (required), and logs a warning if memory is missing
    (optional feature).
    
    Verification Steps:
        1. Check if reasoning_engine.llm is not None
        2. Raise RuntimeError if LLM missing (required dependency)
        3. Check if reasoning_engine.memory is not None
        4. Log warning if memory missing (optional, not error)
        5. Log success if all dependencies present
    
    Args:
        reasoning_engine: The reasoning engine instance to verify
    
    Raises:
        RuntimeError: If LLM dependency is not configured (required)
    
    Example:
        >>> engine = initialize_application()
        >>> verify_dependencies(engine)  # Logs success
        >>> 
        >>> # Example with missing LLM (will raise error)
        >>> engine = ReasoningEngine(llm=None, memory=None)
        >>> verify_dependencies(engine)  # Raises RuntimeError
    """
    logger.info("Verifying dependencies...")
    
    # Check LLM dependency (required)
    if not reasoning_engine.llm:
        error_msg = "LLM dependency not configured - ReasoningEngine requires an LLM implementation"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info(f"LLM dependency verified: {type(reasoning_engine.llm).__name__}")
    
    # Check memory dependency (optional)
    if not reasoning_engine.memory:
        logger.warning(
            "Memory dependency not configured - memory features will be disabled. "
            "This is optional but limits functionality."
        )
    else:
        logger.info(f"Memory dependency verified: {type(reasoning_engine.memory).__name__}")
    
    logger.info("Dependency verification passed")


def cleanup_application(storage: SQLiteStorage) -> None:
    """
    Cleanup application resources, particularly database connections.
    
    This function should be called when shutting down the application
    to ensure all database connections are properly closed. This is
    especially important on Windows where open connections can prevent
    file deletion.
    
    Args:
        storage: The SQLiteStorage instance to cleanup
    
    Example:
        >>> engine, storage = initialize_application(return_storage=True)
        >>> # ... use the engine ...
        >>> cleanup_application(storage)
    """
    logger.info("Cleaning up application resources...")
    
    try:
        # Close all database connections in the pool
        if hasattr(storage, 'connection_pool'):
            storage.connection_pool.close_all()
            logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)
    
    logger.info("Application cleanup completed")
