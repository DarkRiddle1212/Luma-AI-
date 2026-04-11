"""
Example script demonstrating how to run the Luma Memory Module API server
with different uvicorn configurations.

This script shows various ways to start the server with different worker counts
and configurations for development and production use.
"""

from luma_memory.api.server import run_server


def run_development_server():
    """
    Run server in development mode with auto-reload.
    Single worker for easier debugging.
    """
    print("Starting development server with auto-reload...")
    run_server(
        host="127.0.0.1",
        port=8000,
        workers=1,
        log_level="DEBUG",
        reload=True
    )


def run_production_server():
    """
    Run server in production mode with multiple workers.
    Uses configuration from environment or defaults.
    """
    print("Starting production server with multiple workers...")
    run_server()  # Uses config defaults (4 workers)


def run_custom_workers():
    """
    Run server with custom number of workers.
    Useful for scaling based on CPU cores.
    """
    import multiprocessing
    
    # Use number of CPU cores
    workers = multiprocessing.cpu_count()
    print(f"Starting server with {workers} workers (one per CPU core)...")
    
    run_server(
        workers=workers,
        log_level="INFO"
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "dev":
            run_development_server()
        elif mode == "prod":
            run_production_server()
        elif mode == "auto":
            run_custom_workers()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python run_server_example.py [dev|prod|auto]")
            sys.exit(1)
    else:
        print("Usage: python run_server_example.py [dev|prod|auto]")
        print("  dev  - Development mode with auto-reload (1 worker)")
        print("  prod - Production mode with default workers (4 workers)")
        print("  auto - Auto-scale workers based on CPU cores")
        sys.exit(1)
