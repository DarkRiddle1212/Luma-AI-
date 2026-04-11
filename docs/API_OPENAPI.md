# OpenAPI/Swagger Documentation

This directory contains the OpenAPI specification for the Luma Memory Module API.

## Files

- `openapi.json` - Complete OpenAPI 3.1.0 specification in JSON format

## Viewing the Documentation

### Option 1: Interactive Swagger UI (Recommended)

When the API server is running, access the interactive documentation at:

```
http://localhost:8000/docs
```

This provides:
- Interactive API testing
- Request/response examples
- Schema validation
- Try-it-out functionality

### Option 2: ReDoc Documentation

Alternative documentation interface available at:

```
http://localhost:8000/redoc
```

This provides:
- Clean, readable documentation
- Searchable endpoints
- Code samples
- Schema references

### Option 3: External Tools

You can use the `openapi.json` file with various tools:

#### Swagger Editor
1. Go to https://editor.swagger.io/
2. File → Import File → Select `docs/openapi.json`
3. View and edit the specification

#### Postman
1. Open Postman
2. Import → Upload Files → Select `docs/openapi.json`
3. Creates a complete API collection

#### OpenAPI Generator
Generate client libraries in various languages:

```bash
# Install OpenAPI Generator
npm install -g @openapitools/openapi-generator-cli

# Generate Python client
openapi-generator-cli generate -i docs/openapi.json -g python -o clients/python

# Generate JavaScript client
openapi-generator-cli generate -i docs/openapi.json -g javascript -o clients/javascript

# Generate TypeScript client
openapi-generator-cli generate -i docs/openapi.json -g typescript-axios -o clients/typescript
```

## Regenerating the OpenAPI Schema

If you make changes to the API endpoints or models, regenerate the schema:

```bash
python scripts/generate_openapi_schema.py --output docs/openapi.json
```

## API Overview

### Base URL

```
http://localhost:8000/api/v1
```

### Endpoints

#### Health & Statistics
- `GET /api/v1/health` - Health check
- `GET /api/v1/stats` - Storage and performance statistics

#### Memory Operations
- `POST /api/v1/memory` - Create new memory entry
- `GET /api/v1/memory/{entry_id}` - Retrieve memory entry by ID
- `POST /api/v1/memory/query` - Query memory entries with filters

### Authentication

Currently, the API does not require authentication. Future versions will support API key authentication.

### Rate Limiting

No rate limiting is currently enforced. Clients should implement their own throttling if needed.

### Error Handling

All endpoints return standard HTTP status codes:
- `200 OK` - Successful request
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid request parameters or validation error
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server-side error
- `503 Service Unavailable` - Service not ready

Error responses include a JSON body with `error` and optional `detail` fields:

```json
{
  "error": "Validation failed",
  "detail": "Action cannot be empty"
}
```

## Example Usage

### Create Memory Entry

```bash
curl -X POST http://localhost:8000/api/v1/memory \
  -H "Content-Type: application/json" \
  -d '{
    "action": "User opened document",
    "context": {"file": "report.pdf", "page": 1},
    "device_id": "laptop-001",
    "sensitivity": "private",
    "tags": ["document", "work"]
  }'
```

Response:
```json
{
  "entry_id": "abc-123-def-456",
  "message": "Memory entry created successfully"
}
```

### Retrieve Memory Entry

```bash
curl http://localhost:8000/api/v1/memory/abc-123-def-456
```

### Query Memory Entries

```bash
curl -X POST http://localhost:8000/api/v1/memory/query \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-01-01T00:00:00Z",
    "end_time": "2024-01-31T23:59:59Z",
    "tags": ["work"],
    "limit": 50,
    "offset": 0
  }'
```

## Schema Validation

The OpenAPI schema includes comprehensive validation rules:

### CreateMemoryRequest
- `action` (required): Non-empty string
- `context` (required): JSON object
- `device_id` (required): Non-empty string
- `sensitivity` (optional): One of "public", "private", "sensitive" (default: "public")
- `tags` (optional): Array of strings (default: [])

### QueryMemoryRequest
- `start_time` (optional): ISO 8601 datetime string
- `end_time` (optional): ISO 8601 datetime string
- `tags` (optional): Array of strings
- `action_type` (optional): String for partial matching
- `limit` (optional): Integer 1-1000 (default: 100)
- `offset` (optional): Integer >= 0 (default: 0)

## Additional Resources

- [API Documentation](../API_DOCUMENTATION.md) - Detailed API guide
- [Configuration Guide](../CONFIG_GUIDE.md) - Server configuration
- [README](../README.md) - Project overview

## Support

For issues or questions about the API:
1. Check the interactive documentation at `/docs`
2. Review the API_DOCUMENTATION.md file
3. Open an issue on GitHub
