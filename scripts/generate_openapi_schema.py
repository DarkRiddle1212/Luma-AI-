#!/usr/bin/env python3
"""
Generate OpenAPI schema JSON file from the FastAPI application.

This script exports the OpenAPI/Swagger specification to a JSON file
that can be used for documentation, client generation, or API testing tools.

Usage:
    python scripts/generate_openapi_schema.py
    python scripts/generate_openapi_schema.py --output custom_path.json
"""

import json
import sys
from pathlib import Path
from argparse import ArgumentParser

# Add parent directory to path to import luma_memory
sys.path.insert(0, str(Path(__file__).parent.parent))

from luma_memory.api.server import create_app
from luma_memory.config import MemoryModuleConfig


def generate_openapi_schema(output_path: str = "openapi.json") -> None:
    """
    Generate OpenAPI schema and save to file.
    
    Args:
        output_path: Path where the OpenAPI JSON schema will be saved
    """
    print("Generating OpenAPI schema...")
    
    # Create a minimal config for schema generation (won't actually start server)
    config = MemoryModuleConfig(
        db_path=":memory:",  # Use in-memory database for schema generation
        encryption_key_path="./keys/encryption.key",
        api_host="0.0.0.0",
        api_port=8000
    )
    
    # Create FastAPI app (without starting the server)
    # Note: We can't use the lifespan context here, so we create a basic app
    from fastapi import FastAPI
    
    tags_metadata = [
        {
            "name": "Health",
            "description": "Health check and service status endpoints",
        },
        {
            "name": "Statistics",
            "description": "Storage and performance statistics endpoints",
        },
        {
            "name": "Memory Operations",
            "description": "Core memory entry operations including create, retrieve, and query",
        },
    ]
    
    app = FastAPI(
        title="Luma Memory Module API",
        description="""
## Luma Memory Module API

The Luma Memory Module provides persistent storage and retrieval of user actions and context summaries 
for the Luma personal AI system. It serves as the central memory layer for lightweight agents running 
on laptop and phone devices.

### Features

* **Store Memory Entries**: Persist user actions with context, metadata, and optional encryption
* **Query Memories**: Retrieve entries with flexible filtering by time, tags, and action type
* **Automatic Summarization**: Reduce storage overhead by consolidating similar entries
* **Local-First Storage**: All data stored locally using SQLite with optional encryption
* **Performance Optimized**: Sub-100ms storage, sub-200ms retrieval with LRU caching

### Authentication

Currently, the API does not require authentication. Future versions will support API key authentication.

### Rate Limiting

No rate limiting is currently enforced. Clients should implement their own throttling if needed.

### Error Handling

All endpoints return standard HTTP status codes:
- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters or validation error
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server-side error
- `503 Service Unavailable`: Service not ready

Error responses include a JSON body with `error` and optional `detail` fields.
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=tags_metadata,
        contact={
            "name": "Luma Memory Module",
            "url": "https://github.com/luma/memory-module",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
    )
    
    # Import and register routes
    from luma_memory.api.routes import (
        health_check, get_stats, create_memory, get_memory, query_memories
    )
    
    app.get("/api/v1/health")(health_check)
    app.get("/api/v1/stats")(get_stats)
    app.post("/api/v1/memory")(create_memory)
    app.get("/api/v1/memory/{entry_id}")(get_memory)
    app.post("/api/v1/memory/query")(query_memories)
    
    # Get OpenAPI schema
    openapi_schema = app.openapi()
    
    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(openapi_schema, f, indent=2, ensure_ascii=False)
    
    print(f"✓ OpenAPI schema generated successfully: {output_file}")
    print(f"  - Title: {openapi_schema['info']['title']}")
    print(f"  - Version: {openapi_schema['info']['version']}")
    print(f"  - Endpoints: {len(openapi_schema['paths'])}")
    print(f"  - Schemas: {len(openapi_schema.get('components', {}).get('schemas', {}))}")


def main():
    """Main entry point for the script."""
    parser = ArgumentParser(
        description="Generate OpenAPI schema from Luma Memory Module API"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="openapi.json",
        help="Output file path for OpenAPI schema (default: openapi.json)"
    )
    
    args = parser.parse_args()
    
    try:
        generate_openapi_schema(args.output)
    except Exception as e:
        print(f"✗ Error generating OpenAPI schema: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
