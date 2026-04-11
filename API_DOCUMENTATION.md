# Luma Memory Module API Documentation

## Overview

The Luma Memory Module provides a REST API for storing and retrieving user actions and context summaries. This API enables lightweight agents running on laptop and phone devices to maintain persistent memory across sessions.

**Base URL:** `http://localhost:8000`

**API Version:** v1

**Content Type:** `application/json`

---

## Table of Contents

1. [Authentication](#authentication)
2. [Endpoints](#endpoints)
   - [Health Check](#health-check)
   - [Get Statistics](#get-statistics)
   - [Create Memory Entry](#create-memory-entry)
   - [Get Memory Entry](#get-memory-entry)
   - [Query Memory Entries](#query-memory-entries)
3. [Data Models](#data-models)
4. [Error Handling](#error-handling)
5. [Examples](#examples)

---

## Authentication

### Current Status

The Luma Memory Module API currently operates in **local-only mode** without authentication requirements. This design is intentional for the initial release, as the API is designed to run locally on user devices (laptops and phones) and is not exposed to external networks.

**Security Model:**
- API binds to `localhost` (127.0.0.1) by default
- Only processes on the same device can access the API
- No network exposure unless explicitly configured
- Data is stored locally on the device filesystem

### Future Authentication Mechanisms

Future versions will implement authentication when cloud synchronization or multi-device access is enabled. The following authentication methods are planned:

#### 1. API Key Authentication

API keys will provide simple authentication for agent-to-module communication.

**How it works:**
- Each agent receives a unique API key during registration
- API key is included in the `X-API-Key` header for all requests
- Keys can be rotated or revoked as needed

**Example Request with API Key:**
```bash
curl -X POST http://localhost:8000/api/v1/memory \
  -H "Content-Type: application/json" \
  -H "X-API-Key: luma_sk_1234567890abcdef" \
  -d '{
    "action": "User opened document",
    "context": {"file": "report.pdf"},
    "device_id": "laptop-001"
  }'
```

**Error Response (Invalid API Key):**
```json
{
  "error": "Authentication failed",
  "detail": "Invalid or expired API key"
}
```

#### 2. JWT Token Authentication

JWT (JSON Web Token) authentication will be used for user-level authentication when accessing the API from web interfaces or third-party integrations.

**How it works:**
- User authenticates with credentials to obtain a JWT token
- Token is included in the `Authorization: Bearer <token>` header
- Tokens have configurable expiration times
- Refresh tokens enable long-lived sessions

**Example Authentication Flow:**

**Step 1: Obtain Token**
```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user@example.com",
    "password": "secure_password"
  }'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Step 2: Use Token in Requests**
```bash
curl -X GET http://localhost:8000/api/v1/memory/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Step 3: Refresh Token**
```bash
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }'
```

**Error Response (Invalid Token):**
```json
{
  "error": "Authentication failed",
  "detail": "Token expired or invalid"
}
```

#### 3. Device Authentication

Device-level authentication will ensure that only registered devices can access the memory module.

**How it works:**
- Each device generates a unique device certificate during setup
- Certificate is used to establish mutual TLS (mTLS) connection
- Device identity is verified before processing requests

**Configuration Example:**
```json
{
  "authentication": {
    "enabled": true,
    "method": "device_certificate",
    "certificate_path": "/path/to/device.crt",
    "private_key_path": "/path/to/device.key",
    "ca_certificate_path": "/path/to/ca.crt"
  }
}
```

### Authentication Configuration

When authentication is enabled, it will be configured through environment variables or the configuration file:

**Environment Variables:**
```bash
# Enable authentication
LUMA_AUTH_ENABLED=true

# Authentication method: api_key, jwt, device_cert
LUMA_AUTH_METHOD=api_key

# API key for simple authentication
LUMA_API_KEY=luma_sk_1234567890abcdef

# JWT configuration
LUMA_JWT_SECRET=your-secret-key-here
LUMA_JWT_ALGORITHM=HS256
LUMA_JWT_EXPIRATION_MINUTES=60

# Device certificate paths
LUMA_DEVICE_CERT_PATH=/path/to/device.crt
LUMA_DEVICE_KEY_PATH=/path/to/device.key
```

**Configuration File (config.yaml):**
```yaml
authentication:
  enabled: true
  method: api_key  # Options: api_key, jwt, device_cert
  
  api_key:
    key: luma_sk_1234567890abcdef
    header_name: X-API-Key
  
  jwt:
    secret: your-secret-key-here
    algorithm: HS256
    expiration_minutes: 60
    refresh_expiration_days: 30
  
  device_cert:
    certificate_path: /path/to/device.crt
    private_key_path: /path/to/device.key
    ca_certificate_path: /path/to/ca.crt
```

### Security Best Practices

When authentication is enabled, follow these security best practices:

#### For API Key Authentication:
1. **Keep keys secret:** Never commit API keys to version control
2. **Use environment variables:** Store keys in environment variables or secure vaults
3. **Rotate keys regularly:** Change API keys periodically (e.g., every 90 days)
4. **Use different keys per environment:** Separate keys for development, staging, and production
5. **Revoke compromised keys immediately:** If a key is exposed, revoke it and generate a new one

#### For JWT Authentication:
1. **Use strong secrets:** JWT secrets should be at least 256 bits of random data
2. **Set appropriate expiration:** Access tokens should expire within 1 hour
3. **Implement refresh tokens:** Use refresh tokens for long-lived sessions
4. **Validate token claims:** Always verify issuer, audience, and expiration
5. **Use HTTPS:** Never transmit tokens over unencrypted connections

#### For Device Certificates:
1. **Protect private keys:** Store private keys securely with appropriate file permissions
2. **Use strong key algorithms:** Use RSA 2048-bit or ECDSA P-256 keys
3. **Implement certificate rotation:** Rotate certificates before expiration
4. **Maintain certificate revocation list:** Track and revoke compromised certificates
5. **Verify certificate chain:** Always validate the full certificate chain

#### General Security:
1. **Use HTTPS in production:** Enable TLS/SSL for all API communications
2. **Implement rate limiting:** Prevent brute force attacks with rate limits
3. **Log authentication attempts:** Monitor failed authentication attempts
4. **Use secure defaults:** Authentication should be enabled by default in production
5. **Implement IP whitelisting:** Restrict API access to known IP addresses when possible
6. **Enable audit logging:** Log all authenticated requests for security auditing

### Testing Authentication

When authentication is enabled, you can test it using the following approaches:

**Test with curl:**
```bash
# Test with API key
curl -X GET http://localhost:8000/api/v1/health \
  -H "X-API-Key: luma_sk_1234567890abcdef"

# Test with JWT token
curl -X GET http://localhost:8000/api/v1/health \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Test with Python:**
```python
import requests

# API Key authentication
headers = {
    "X-API-Key": "luma_sk_1234567890abcdef"
}
response = requests.get("http://localhost:8000/api/v1/health", headers=headers)

# JWT authentication
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
response = requests.get("http://localhost:8000/api/v1/health", headers=headers)
```

**Test with JavaScript:**
```javascript
// API Key authentication
const response = await fetch('http://localhost:8000/api/v1/health', {
  headers: {
    'X-API-Key': 'luma_sk_1234567890abcdef'
  }
});

// JWT authentication
const response = await fetch('http://localhost:8000/api/v1/health', {
  headers: {
    'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'
  }
});
```

### Migration Path

When authentication is enabled in a future release, existing installations will need to migrate:

1. **Update configuration:** Add authentication settings to config file
2. **Generate credentials:** Create API keys or certificates for existing agents
3. **Update agent code:** Modify agents to include authentication headers
4. **Test thoroughly:** Verify all agents can authenticate successfully
5. **Enable authentication:** Set `LUMA_AUTH_ENABLED=true` in production

**Backward Compatibility:**
- Authentication will be disabled by default to maintain compatibility
- Existing agents will continue to work without modification
- A grace period will be provided before requiring authentication
- Clear migration documentation will be provided

### Troubleshooting Authentication

Common authentication issues and solutions:

**Issue: "Authentication failed: Invalid API key"**
- Verify the API key is correct and not expired
- Check that the key is included in the `X-API-Key` header
- Ensure no extra whitespace in the key value

**Issue: "Authentication failed: Token expired"**
- Obtain a new access token using the refresh token
- Check system clock synchronization (JWT validation is time-sensitive)
- Verify token expiration settings in configuration

**Issue: "Authentication failed: Certificate verification failed"**
- Verify certificate is not expired
- Check that certificate chain is complete
- Ensure CA certificate is trusted
- Verify file permissions on certificate and key files

**Issue: "403 Forbidden"**
- Authentication succeeded but authorization failed
- Verify the authenticated user/device has permission for the requested resource
- Check role-based access control (RBAC) settings if enabled

For additional support, enable debug logging:
```bash
LUMA_LOG_LEVEL=DEBUG python -m luma_memory.api.server
```

---

## Endpoints

### Health Check

Check the health status of the API server.

**Endpoint:** `GET /api/v1/health`

**Response:** `200 OK`

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Example Request:**

```bash
curl -X GET http://localhost:8000/api/v1/health
```

**Example Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### Get Statistics

Retrieve storage and performance statistics.

**Endpoint:** `GET /api/v1/stats`

**Response:** `200 OK`

```json
{
  "total_entries": 1000,
  "storage_size_bytes": 1048576,
  "encryption_enabled": true,
  "summarizer_enabled": true,
  "config": {
    "cache_size": 1000,
    "max_storage_size_mb": 1000
  },
  "performance": {
    "create_memory": {
      "count": 100,
      "avg_time_ms": 45.2
    }
  }
}
```

**Example Request:**

```bash
curl -X GET http://localhost:8000/api/v1/stats
```

**Example Response:**

```json
{
  "total_entries": 1523,
  "storage_size_bytes": 2097152,
  "encryption_enabled": true,
  "summarizer_enabled": true,
  "config": {
    "cache_size": 1000,
    "max_storage_size_mb": 1000,
    "similarity_threshold": 0.8
  },
  "performance": {
    "create_memory": {
      "count": 1523,
      "avg_time_ms": 42.7,
      "min_time_ms": 15.3,
      "max_time_ms": 98.4
    },
    "get_memory": {
      "count": 3045,
      "avg_time_ms": 18.2
    }
  }
}
```

---

### Create Memory Entry

Store a new memory entry with action, context, and metadata.

**Endpoint:** `POST /api/v1/memory`

**Request Body:**

```json
{
  "action": "string (required)",
  "context": {
    "key": "value"
  },
  "device_id": "string (required)",
  "sensitivity": "public|private|sensitive (optional, default: public)",
  "tags": ["string"] (optional)
}
```

**Response:** `201 Created`

```json
{
  "entry_id": "abc-123-def-456",
  "message": "Memory entry created successfully"
}
```

**Example Request:**

```bash
curl -X POST http://localhost:8000/api/v1/memory \
  -H "Content-Type: application/json" \
  -d '{
    "action": "User opened document",
    "context": {
      "file": "report.pdf",
      "page": 1,
      "application": "Adobe Reader"
    },
    "device_id": "laptop-001",
    "sensitivity": "private",
    "tags": ["document", "work", "pdf"]
  }'
```

**Example Response:**

```json
{
  "entry_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Memory entry created successfully"
}
```

**Error Responses:**

- `400 Bad Request` - Invalid request data or validation error
- `500 Internal Server Error` - Storage or processing error

**Example Error Response:**

```json
{
  "error": "Validation error",
  "detail": "Action cannot be empty"
}
```

---

### Get Memory Entry

Retrieve a specific memory entry by its unique identifier.

**Endpoint:** `GET /api/v1/memory/{entry_id}`

**Path Parameters:**
- `entry_id` (string, required) - Unique identifier of the memory entry

**Response:** `200 OK`

```json
{
  "id": "abc-123-def-456",
  "timestamp": "2024-01-15T10:30:00Z",
  "action": "User opened document",
  "context": {
    "file": "report.pdf"
  },
  "sensitivity": "private",
  "device_id": "laptop-001",
  "sync_status": "pending",
  "tags": ["document", "work"],
  "summary": null,
  "parent_id": null,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

**Example Request:**

```bash
curl -X GET http://localhost:8000/api/v1/memory/550e8400-e29b-41d4-a716-446655440000
```

**Example Response:**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-15T10:30:00Z",
  "action": "User opened document",
  "context": {
    "file": "report.pdf",
    "page": 1,
    "application": "Adobe Reader"
  },
  "sensitivity": "private",
  "device_id": "laptop-001",
  "sync_status": "pending",
  "tags": ["document", "work", "pdf"],
  "summary": null,
  "parent_id": null,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

**Error Responses:**

- `404 Not Found` - Entry with specified ID does not exist
- `500 Internal Server Error` - Storage or decryption error

**Example Error Response:**

```json
{
  "error": "Memory entry not found: 550e8400-e29b-41d4-a716-446655440000"
}
```

---

### Query Memory Entries

Query memory entries with filters and pagination.

**Endpoint:** `POST /api/v1/memory/query`

**Request Body:**

```json
{
  "start_time": "2024-01-01T00:00:00Z" (optional),
  "end_time": "2024-01-31T23:59:59Z" (optional),
  "tags": ["work", "document"] (optional),
  "action_type": "opened" (optional, partial match),
  "limit": 100 (optional, default: 100, max: 1000),
  "offset": 0 (optional, default: 0)
}
```

**Response:** `200 OK`

```json
{
  "entries": [
    {
      "id": "abc-123-def-456",
      "timestamp": "2024-01-15T10:30:00Z",
      "action": "User opened document",
      "context": {},
      "sensitivity": "private",
      "device_id": "laptop-001",
      "sync_status": "pending",
      "tags": ["document", "work"],
      "summary": null,
      "parent_id": null,
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

**Example Request - Query by Time Range:**

```bash
curl -X POST http://localhost:8000/api/v1/memory/query \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-01-01T00:00:00Z",
    "end_time": "2024-01-31T23:59:59Z",
    "limit": 50
  }'
```

**Example Request - Query by Tags:**

```bash
curl -X POST http://localhost:8000/api/v1/memory/query \
  -H "Content-Type: application/json" \
  -d '{
    "tags": ["work", "document"],
    "limit": 100,
    "offset": 0
  }'
```

**Example Request - Query by Action Type:**

```bash
curl -X POST http://localhost:8000/api/v1/memory/query \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "opened",
    "limit": 25
  }'
```

**Example Request - Combined Filters with Pagination:**

```bash
curl -X POST http://localhost:8000/api/v1/memory/query \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-01-01T00:00:00Z",
    "end_time": "2024-01-31T23:59:59Z",
    "tags": ["work"],
    "action_type": "document",
    "limit": 50,
    "offset": 0
  }'
```

**Example Response:**

```json
{
  "entries": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "timestamp": "2024-01-15T14:22:00Z",
      "action": "User opened document",
      "context": {
        "file": "quarterly_report.pdf",
        "page": 1
      },
      "sensitivity": "private",
      "device_id": "laptop-001",
      "sync_status": "pending",
      "tags": ["document", "work", "pdf"],
      "summary": null,
      "parent_id": null,
      "created_at": "2024-01-15T14:22:00Z",
      "updated_at": "2024-01-15T14:22:00Z"
    },
    {
      "id": "660e8400-e29b-41d4-a716-446655440001",
      "timestamp": "2024-01-15T10:30:00Z",
      "action": "User edited document",
      "context": {
        "file": "meeting_notes.docx",
        "changes": 127
      },
      "sensitivity": "private",
      "device_id": "laptop-001",
      "sync_status": "pending",
      "tags": ["document", "work", "editing"],
      "summary": null,
      "parent_id": null,
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 2,
  "limit": 50,
  "offset": 0
}
```

**Error Responses:**

- `400 Bad Request` - Invalid query parameters (e.g., malformed datetime)
- `500 Internal Server Error` - Storage error

---

## Data Models

### MemoryEntry

Represents a single memory entry in the system.

```json
{
  "id": "string (UUID)",
  "timestamp": "string (ISO 8601 datetime)",
  "action": "string",
  "context": {
    "key": "value"
  },
  "sensitivity": "public|private|sensitive",
  "device_id": "string",
  "sync_status": "pending|synced|conflict",
  "tags": ["string"],
  "summary": "string|null",
  "parent_id": "string|null",
  "created_at": "string (ISO 8601 datetime)|null",
  "updated_at": "string (ISO 8601 datetime)|null"
}
```

**Field Descriptions:**

- `id`: Unique identifier for the entry (UUID format)
- `timestamp`: When the action occurred (ISO 8601 format with Z suffix)
- `action`: Description of the user action
- `context`: Dictionary containing contextual information about the action
- `sensitivity`: Privacy level of the entry
  - `public`: Non-sensitive information
  - `private`: Personal but not highly sensitive
  - `sensitive`: Highly sensitive information (encrypted at rest)
- `device_id`: Identifier of the device that created the entry
- `sync_status`: Synchronization status
  - `pending`: Not yet synchronized
  - `synced`: Successfully synchronized
  - `conflict`: Sync conflict detected
- `tags`: List of tags for categorization and filtering
- `summary`: Optional summary text (used for summarized entries)
- `parent_id`: Reference to parent entry if this is a summarized entry
- `created_at`: When the entry was created in the system
- `updated_at`: When the entry was last updated

### Sensitivity Levels

The API supports three sensitivity levels:

1. **public**: Non-sensitive information that can be freely shared
   - Example: "User opened weather app"
   
2. **private**: Personal information that should be kept private
   - Example: "User opened document: work_report.pdf"
   
3. **sensitive**: Highly sensitive information (automatically encrypted)
   - Example: "User entered password for banking app"

---

## Error Handling

All error responses follow a consistent format:

```json
{
  "error": "Brief error message",
  "detail": "Detailed error information (optional)"
}
```

### HTTP Status Codes

- `200 OK` - Request successful
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid request data or parameters
- `404 Not Found` - Requested resource does not exist
- `500 Internal Server Error` - Server-side error
- `503 Service Unavailable` - Service not ready (e.g., during startup)

### Common Error Scenarios

**Validation Error:**
```json
{
  "error": "Validation error",
  "detail": "Action cannot be empty"
}
```

**Invalid Sensitivity Level:**
```json
{
  "error": "Invalid sensitivity level: confidential. Must be one of: public, private, sensitive"
}
```

**Entry Not Found:**
```json
{
  "error": "Memory entry not found: 550e8400-e29b-41d4-a716-446655440000"
}
```

**Invalid Datetime Format:**
```json
{
  "error": "Invalid start_time format: 2024-01-01. Use ISO format."
}
```

**Storage Error:**
```json
{
  "error": "Storage error: Database connection failed"
}
```

---

## Error Code Reference

This section provides a comprehensive reference of all error codes, their meanings, and recommended actions.

### HTTP Status Codes

| Status Code | Name | Description | When It Occurs |
|------------|------|-------------|----------------|
| 200 | OK | Request successful | Successful GET requests |
| 201 | Created | Resource created successfully | Successful POST /api/v1/memory |
| 400 | Bad Request | Invalid request data or parameters | Validation errors, malformed JSON, invalid parameters |
| 404 | Not Found | Requested resource does not exist | Entry ID not found |
| 500 | Internal Server Error | Server-side error | Storage errors, encryption errors, unexpected exceptions |
| 503 | Service Unavailable | Service not ready | Server starting up, database not initialized |

### Error Categories

#### 1. Validation Errors (400 Bad Request)

These errors occur when the request data does not meet validation requirements.

**Error Code:** `VALIDATION_ERROR`

**Common Scenarios:**

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `Action cannot be empty` | Missing or empty `action` field | Provide a non-empty action string |
| `Device ID cannot be empty` | Missing or empty `device_id` field | Provide a valid device identifier |
| `Context must be a dictionary` | Invalid `context` field type | Ensure context is a JSON object |
| `Tags must be a list` | Invalid `tags` field type | Ensure tags is a JSON array of strings |
| `Invalid sensitivity level: {value}` | Invalid sensitivity value | Use one of: `public`, `private`, `sensitive` |
| `Limit must be between 1 and 1000` | Invalid `limit` parameter | Set limit between 1 and 1000 |
| `Offset must be non-negative` | Negative `offset` parameter | Set offset to 0 or positive integer |

**Example Error Response:**
```json
{
  "error": "Validation error",
  "detail": "Action cannot be empty"
}
```

**Recommended Actions:**
1. Validate request data before sending
2. Check that all required fields are present
3. Ensure field types match the API specification
4. Review the Data Models section for correct formats

---

#### 2. Format Errors (400 Bad Request)

These errors occur when data is in an incorrect format.

**Error Code:** `FORMAT_ERROR`

**Common Scenarios:**

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `Invalid start_time format: {value}. Use ISO format.` | Malformed datetime string | Use ISO 8601 format: `2024-01-15T10:30:00Z` |
| `Invalid end_time format: {value}. Use ISO format.` | Malformed datetime string | Use ISO 8601 format: `2024-01-15T10:30:00Z` |
| `Invalid JSON in request body` | Malformed JSON | Validate JSON syntax before sending |
| `Invalid UUID format: {value}` | Malformed entry ID | Use valid UUID format |

**Example Error Response:**
```json
{
  "error": "Invalid start_time format: 2024-01-01. Use ISO format."
}
```

**Recommended Actions:**
1. Use ISO 8601 datetime format with Z suffix (UTC)
2. Validate JSON syntax before sending requests
3. Use proper UUID format for entry IDs
4. Test datetime parsing in your client code

**Correct Datetime Format Examples:**
- `2024-01-15T10:30:00Z` ✓
- `2024-01-15T10:30:00.123Z` ✓ (with milliseconds)
- `2024-01-15 10:30:00` ✗ (missing T separator and Z suffix)
- `01/15/2024` ✗ (not ISO format)

---

#### 3. Not Found Errors (404 Not Found)

These errors occur when a requested resource does not exist.

**Error Code:** `NOT_FOUND`

**Common Scenarios:**

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `Memory entry not found: {entry_id}` | Entry ID does not exist | Verify the entry ID is correct |
| `Endpoint not found` | Invalid API endpoint | Check the endpoint URL and method |

**Example Error Response:**
```json
{
  "error": "Memory entry not found: 550e8400-e29b-41d4-a716-446655440000"
}
```

**Recommended Actions:**
1. Verify the entry ID was created successfully
2. Check for typos in the entry ID
3. Ensure the entry was not deleted
4. Query entries to find the correct ID

---

#### 4. Storage Errors (500 Internal Server Error)

These errors occur when there is a problem with the storage backend.

**Error Code:** `STORAGE_ERROR`

**Common Scenarios:**

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `Storage error: Database connection failed` | Cannot connect to SQLite database | Check database file permissions and disk space |
| `Storage error: Database is locked` | Concurrent write conflict | Retry the operation after a short delay |
| `Storage error: Disk full` | Insufficient disk space | Free up disk space |
| `Storage error: Database corruption detected` | Corrupted database file | Restore from backup or reinitialize database |
| `Failed to create entry: {details}` | Error during entry creation | Check logs for detailed error information |
| `Failed to query entries: {details}` | Error during query execution | Verify query parameters and check logs |

**Example Error Response:**
```json
{
  "error": "Storage error: Database connection failed"
}
```

**Recommended Actions:**
1. Check database file permissions (should be readable/writable)
2. Verify sufficient disk space is available
3. Check system logs for detailed error information
4. Retry the operation after a short delay
5. If corruption is detected, restore from backup

---

#### 5. Encryption Errors (500 Internal Server Error)

These errors occur when there is a problem with encryption or decryption.

**Error Code:** `ENCRYPTION_ERROR`

**Common Scenarios:**

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `Encryption error: Key not found` | Encryption key file missing | Ensure encryption key file exists at configured path |
| `Encryption error: Invalid key format` | Corrupted or invalid key | Regenerate encryption key (will lose access to encrypted data) |
| `Decryption error: Invalid token` | Data corrupted or wrong key | Check data integrity and key configuration |
| `Encryption error: Permission denied` | Cannot read/write key file | Check file permissions on key file |

**Example Error Response:**
```json
{
  "error": "Encryption error: Key not found"
}
```

**Recommended Actions:**
1. Verify encryption key file exists at configured path
2. Check file permissions on encryption key (should be readable)
3. Ensure key file is not corrupted
4. If key is lost, encrypted data cannot be recovered
5. Consider implementing key backup procedures

**Key File Location:**
- Default: `./keys/encryption.key`
- Configurable via: `LUMA_ENCRYPTION_KEY_PATH` environment variable

---

#### 6. Service Unavailable Errors (503 Service Unavailable)

These errors occur when the service is not ready to handle requests.

**Error Code:** `SERVICE_UNAVAILABLE`

**Common Scenarios:**

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `Service starting up, please retry` | Server is initializing | Wait a few seconds and retry |
| `Database not initialized` | Database schema not created | Wait for initialization to complete |
| `Service shutting down` | Server is shutting down | Wait for restart or start a new instance |

**Example Error Response:**
```json
{
  "error": "Service starting up, please retry"
}
```

**Recommended Actions:**
1. Wait 2-5 seconds and retry the request
2. Check server logs for initialization progress
3. Verify server started successfully
4. Implement retry logic with exponential backoff in clients

---

### Error Response Format

All error responses follow this consistent format:

```json
{
  "error": "Brief error message describing what went wrong",
  "detail": "Optional detailed information about the error"
}
```

**Fields:**
- `error` (string, required): A brief, human-readable error message
- `detail` (string, optional): Additional context or technical details about the error

---

### Client Error Handling Best Practices

#### 1. Implement Retry Logic

For transient errors (500, 503), implement exponential backoff:

```python
import time
import requests

def create_memory_with_retry(data, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post("http://localhost:8000/api/v1/memory", json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [500, 503] and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                time.sleep(wait_time)
                continue
            raise
```

#### 2. Validate Before Sending

Validate data on the client side to catch errors early:

```python
def validate_memory_entry(action, context, device_id, sensitivity="public"):
    errors = []
    
    if not action or not action.strip():
        errors.append("Action cannot be empty")
    
    if not device_id or not device_id.strip():
        errors.append("Device ID cannot be empty")
    
    if not isinstance(context, dict):
        errors.append("Context must be a dictionary")
    
    if sensitivity not in ["public", "private", "sensitive"]:
        errors.append(f"Invalid sensitivity level: {sensitivity}")
    
    if errors:
        raise ValueError("; ".join(errors))
    
    return True
```

#### 3. Handle Specific Error Cases

Handle different error types appropriately:

```python
def handle_api_error(response):
    status_code = response.status_code
    error_data = response.json()
    
    if status_code == 400:
        # Validation error - fix the request data
        print(f"Validation error: {error_data.get('detail')}")
        # Don't retry, fix the data instead
        
    elif status_code == 404:
        # Not found - entry doesn't exist
        print(f"Entry not found: {error_data.get('error')}")
        # Don't retry, entry doesn't exist
        
    elif status_code == 500:
        # Server error - retry with backoff
        print(f"Server error: {error_data.get('error')}")
        # Implement retry logic
        
    elif status_code == 503:
        # Service unavailable - retry after delay
        print(f"Service unavailable: {error_data.get('error')}")
        # Wait and retry
```

#### 4. Log Errors for Debugging

Always log errors with sufficient context:

```python
import logging

logger = logging.getLogger(__name__)

try:
    response = requests.post(url, json=data)
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    logger.error(
        "API request failed",
        extra={
            "status_code": e.response.status_code,
            "error": e.response.json(),
            "request_data": data,
            "url": url
        }
    )
    raise
```

---

### Debugging Tips

#### Enable Debug Logging

Set log level to DEBUG for detailed error information:

```bash
LUMA_LOG_LEVEL=DEBUG python -m luma_memory.api.server
```

#### Check Server Logs

Server logs provide detailed error traces:

```bash
# View recent logs
tail -f /var/log/luma_memory/server.log

# Search for errors
grep "ERROR" /var/log/luma_memory/server.log
```

#### Test with curl

Use curl with verbose output to debug requests:

```bash
curl -v -X POST http://localhost:8000/api/v1/memory \
  -H "Content-Type: application/json" \
  -d '{"action": "test", "device_id": "test-001"}'
```

#### Verify Database State

Check database directly for debugging:

```bash
sqlite3 data/luma_memory.db "SELECT COUNT(*) FROM memory_entries;"
```

---

### Common Error Scenarios and Solutions

#### Scenario 1: "Action cannot be empty"

**Problem:** Trying to create a memory entry without an action.

**Solution:**
```python
# ✗ Wrong
data = {"context": {}, "device_id": "laptop-001"}

# ✓ Correct
data = {
    "action": "User opened document",
    "context": {},
    "device_id": "laptop-001"
}
```

#### Scenario 2: "Invalid start_time format"

**Problem:** Using incorrect datetime format.

**Solution:**
```python
# ✗ Wrong
data = {"start_time": "2024-01-01"}

# ✓ Correct
data = {"start_time": "2024-01-01T00:00:00Z"}
```

#### Scenario 3: "Memory entry not found"

**Problem:** Trying to retrieve an entry that doesn't exist.

**Solution:**
```python
# First, verify the entry was created
result = create_memory(...)
entry_id = result["entry_id"]

# Then retrieve it
entry = get_memory(entry_id)

# Or query to find existing entries
results = query_memories(tags=["work"])
```

#### Scenario 4: "Database is locked"

**Problem:** Concurrent write operations causing lock contention.

**Solution:**
```python
import time
import random

def create_with_retry(data, max_retries=3):
    for attempt in range(max_retries):
        try:
            return create_memory(data)
        except Exception as e:
            if "locked" in str(e) and attempt < max_retries - 1:
                time.sleep(random.uniform(0.1, 0.5))
                continue
            raise
```

#### Scenario 5: "Service starting up"

**Problem:** Sending requests before server is ready.

**Solution:**
```python
import time
import requests

def wait_for_service(max_wait=30):
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.get("http://localhost:8000/api/v1/health")
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    return False

# Wait for service before making requests
if wait_for_service():
    result = create_memory(...)
else:
    print("Service did not start in time")
```

---

## Examples

### Example 1: Basic Workflow

**Step 1: Create a memory entry**

```bash
curl -X POST http://localhost:8000/api/v1/memory \
  -H "Content-Type: application/json" \
  -d '{
    "action": "User started coding session",
    "context": {
      "project": "luma-memory",
      "language": "python",
      "editor": "vscode"
    },
    "device_id": "laptop-001",
    "sensitivity": "public",
    "tags": ["coding", "python", "work"]
  }'
```

**Response:**
```json
{
  "entry_id": "770e8400-e29b-41d4-a716-446655440002",
  "message": "Memory entry created successfully"
}
```

**Step 2: Retrieve the entry**

```bash
curl -X GET http://localhost:8000/api/v1/memory/770e8400-e29b-41d4-a716-446655440002
```

**Response:**
```json
{
  "id": "770e8400-e29b-41d4-a716-446655440002",
  "timestamp": "2024-01-15T15:45:00Z",
  "action": "User started coding session",
  "context": {
    "project": "luma-memory",
    "language": "python",
    "editor": "vscode"
  },
  "sensitivity": "public",
  "device_id": "laptop-001",
  "sync_status": "pending",
  "tags": ["coding", "python", "work"],
  "summary": null,
  "parent_id": null,
  "created_at": "2024-01-15T15:45:00Z",
  "updated_at": "2024-01-15T15:45:00Z"
}
```

---

### Example 2: Query Recent Work Activities

```bash
curl -X POST http://localhost:8000/api/v1/memory/query \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-01-15T00:00:00Z",
    "tags": ["work"],
    "limit": 10
  }'
```

**Response:**
```json
{
  "entries": [
    {
      "id": "770e8400-e29b-41d4-a716-446655440002",
      "timestamp": "2024-01-15T15:45:00Z",
      "action": "User started coding session",
      "context": {
        "project": "luma-memory",
        "language": "python"
      },
      "sensitivity": "public",
      "device_id": "laptop-001",
      "sync_status": "pending",
      "tags": ["coding", "python", "work"],
      "summary": null,
      "parent_id": null,
      "created_at": "2024-01-15T15:45:00Z",
      "updated_at": "2024-01-15T15:45:00Z"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

---

### Example 3: Sensitive Data Storage

```bash
curl -X POST http://localhost:8000/api/v1/memory \
  -H "Content-Type: application/json" \
  -d '{
    "action": "User logged into banking app",
    "context": {
      "app": "mobile_banking",
      "account_type": "checking"
    },
    "device_id": "phone-001",
    "sensitivity": "sensitive",
    "tags": ["banking", "authentication"]
  }'
```

**Note:** Entries with `sensitivity: "sensitive"` are automatically encrypted at rest using AES-256 encryption.

---

### Example 4: Pagination

**Get first page:**
```bash
curl -X POST http://localhost:8000/api/v1/memory/query \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 50,
    "offset": 0
  }'
```

**Get second page:**
```bash
curl -X POST http://localhost:8000/api/v1/memory/query \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 50,
    "offset": 50
  }'
```

---

### Example 5: Python Client

```python
import requests
from datetime import datetime

# Base URL
BASE_URL = "http://localhost:8000"

# Create a memory entry
def create_memory(action, context, device_id, sensitivity="public", tags=None):
    url = f"{BASE_URL}/api/v1/memory"
    payload = {
        "action": action,
        "context": context,
        "device_id": device_id,
        "sensitivity": sensitivity,
        "tags": tags or []
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

# Get a memory entry
def get_memory(entry_id):
    url = f"{BASE_URL}/api/v1/memory/{entry_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# Query memory entries
def query_memories(start_time=None, end_time=None, tags=None, limit=100):
    url = f"{BASE_URL}/api/v1/memory/query"
    payload = {
        "limit": limit
    }
    if start_time:
        payload["start_time"] = start_time
    if end_time:
        payload["end_time"] = end_time
    if tags:
        payload["tags"] = tags
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

# Example usage
if __name__ == "__main__":
    # Create an entry
    result = create_memory(
        action="User opened document",
        context={"file": "report.pdf", "page": 1},
        device_id="laptop-001",
        sensitivity="private",
        tags=["document", "work"]
    )
    print(f"Created entry: {result['entry_id']}")
    
    # Retrieve the entry
    entry = get_memory(result['entry_id'])
    print(f"Retrieved entry: {entry['action']}")
    
    # Query recent work entries
    results = query_memories(
        start_time="2024-01-01T00:00:00Z",
        tags=["work"],
        limit=10
    )
    print(f"Found {results['total']} entries")
```

---

### Example 6: JavaScript/Node.js Client

```javascript
const axios = require('axios');

const BASE_URL = 'http://localhost:8000';

// Create a memory entry
async function createMemory(action, context, deviceId, sensitivity = 'public', tags = []) {
  const response = await axios.post(`${BASE_URL}/api/v1/memory`, {
    action,
    context,
    device_id: deviceId,
    sensitivity,
    tags
  });
  return response.data;
}

// Get a memory entry
async function getMemory(entryId) {
  const response = await axios.get(`${BASE_URL}/api/v1/memory/${entryId}`);
  return response.data;
}

// Query memory entries
async function queryMemories({ startTime, endTime, tags, actionType, limit = 100, offset = 0 }) {
  const response = await axios.post(`${BASE_URL}/api/v1/memory/query`, {
    start_time: startTime,
    end_time: endTime,
    tags,
    action_type: actionType,
    limit,
    offset
  });
  return response.data;
}

// Example usage
(async () => {
  try {
    // Create an entry
    const result = await createMemory(
      'User opened document',
      { file: 'report.pdf', page: 1 },
      'laptop-001',
      'private',
      ['document', 'work']
    );
    console.log(`Created entry: ${result.entry_id}`);
    
    // Retrieve the entry
    const entry = await getMemory(result.entry_id);
    console.log(`Retrieved entry: ${entry.action}`);
    
    // Query recent work entries
    const results = await queryMemories({
      startTime: '2024-01-01T00:00:00Z',
      tags: ['work'],
      limit: 10
    });
    console.log(`Found ${results.total} entries`);
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
  }
})();
```

---

## Interactive API Documentation

The API provides interactive documentation through Swagger UI and ReDoc:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

These interfaces allow you to:
- Explore all available endpoints
- View request/response schemas
- Test API calls directly from the browser
- Download OpenAPI specification

---

## Rate Limiting

Currently, the API does not implement rate limiting. Future versions may add rate limiting to prevent abuse.

---

## Performance Considerations

- **Store operations:** Target < 100ms for typical entry sizes
- **Retrieve operations:** Target < 200ms for queries returning up to 100 entries
- **Memory usage:** System maintains < 100MB during normal operation
- **Caching:** Frequently accessed entries are cached in memory (LRU cache)
- **Pagination:** Use `limit` and `offset` parameters for large result sets

---

## Best Practices

1. **Use appropriate sensitivity levels:** Mark sensitive data appropriately to ensure encryption
2. **Add meaningful tags:** Tags improve query performance and organization
3. **Include rich context:** Store relevant contextual information for better retrieval
4. **Use pagination:** For large queries, use pagination to avoid memory issues
5. **Handle errors gracefully:** Always check HTTP status codes and handle errors appropriately
6. **Use ISO 8601 datetime format:** All timestamps should use ISO 8601 format with Z suffix
7. **Provide unique device IDs:** Use consistent device identifiers across sessions

---

## Support and Feedback

For issues, questions, or feedback, please refer to the project repository or contact the development team.
