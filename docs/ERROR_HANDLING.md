# Error Handling in ShopRec

ShopRec provides robust error handling throughout the API and recommendation system, ensuring graceful degradation and informative error messages.

## Error Types

### 1. Model Not Found (503 Service Unavailable)

**When it occurs:**
- Model files are missing from the specified directory
- Model directory doesn't exist
- Required model artifacts (SVD model, mappings) are incomplete

**Response example:**
```json
{
  "error": "Model not found",
  "message": "Model not found at 'models'. Please train a model first.",
  "details": "Please train a model first using the training script."
}
```

**How to fix:**
- Run the training script: `python scripts/train_model.py data/fake_purchases.csv`
- Ensure the model directory contains all required files:
  - `svd_model.joblib`
  - `user_id_mapping.joblib`
  - `product_id_mapping.joblib`

### 2. User Not Found (404 Not Found)

**When it occurs:**
- User ID is not in the training data
- `allow_cold_start=false` parameter is set

**Response example:**
```json
{
  "error": "User not found",
  "message": "User 999 not found in training data. Cannot generate personalized recommendations.",
  "details": "User not found in training data. Enable cold-start mode for fallback recommendations."
}
```

**How to handle:**
- Enable cold-start mode (default): `/recommend/999?allow_cold_start=true`
- Cold-start mode returns popular items for unknown users
- Retrain the model with updated user data

### 3. Validation Errors (422 Unprocessable Entity)

**When it occurs:**
- Invalid parameter types (e.g., non-numeric user_id)
- Invalid query parameters

**Response example:**
```json
{
  "error": "Validation error",
  "message": "Request validation failed",
  "details": [
    {
      "loc": ["path", "user_id"],
      "msg": "value is not a valid integer",
      "type": "type_error.integer"
    }
  ]
}
```

### 4. Internal Server Errors (500 Internal Server Error)

**When it occurs:**
- Unexpected exceptions during recommendation generation
- Model corruption or incompatibility
- System resource errors

**Response example:**
```json
{
  "error": "Internal server error",
  "message": "An unexpected error occurred. Please try again later.",
  "type": "ValueError"
}
```

**What happens:**
- Full traceback is logged for debugging
- Generic error message returned to user (security best practice)
- Request ID included for log correlation

## Error Handling Middleware

### Global Exception Handler

All unhandled exceptions are caught by the global exception handler, which:
- Logs the full error with traceback
- Returns a consistent JSON error response
- Includes request context (path, method, user agent)
- Assigns a unique request ID for tracking

### HTTP Exception Handler

FastAPI HTTP exceptions are handled with:
- Appropriate HTTP status codes
- Structured error messages
- Context-specific details
- Consistent JSON format

### Custom Exception Handlers

Specific handlers for:
- `ModelNotFoundError`: 503 Service Unavailable
- `UserNotFoundError`: 404 Not Found
- `RequestValidationError`: 422 Unprocessable Entity

## Logging

### Error Log Levels

- **ERROR**: Model not found, load failures, internal errors
- **WARNING**: User not found, validation issues, fallback scenarios
- **INFO**: Successful operations, request completions

### Log Structure

All error logs include:
```json
{
  "timestamp": "2026-01-06T21:34:04.500977Z",
  "level": "ERROR",
  "logger": "src.recommender.infer",
  "message": "Model files not found",
  "module": "infer",
  "function": "recommend_products_for_user",
  "line": 78,
  "error": "Model not found at 'non_existent_dir'",
  "error_type": "ModelNotFoundError",
  "user_id": 1,
  "model_path": "non_existent_dir",
  "traceback": "..."
}
```

## Cold-Start Handling

### Default Behavior

By default, `allow_cold_start=true`, which means:
- Unknown users receive fallback recommendations
- System returns the first N products from the catalog
- 200 OK response with recommendations

### Strict Mode

With `allow_cold_start=false`:
- Unknown users receive 404 Not Found
- Requires explicit user presence in training data
- Useful for testing or strict validation scenarios

## API Examples

### Graceful Error Handling

```bash
# Model not found - returns 503
curl http://localhost:8000/recommend/1?model_dir=invalid

# User not found with cold-start - returns 200 with fallback
curl http://localhost:8000/recommend/999?allow_cold_start=true

# User not found strict mode - returns 404
curl http://localhost:8000/recommend/999?allow_cold_start=false

# Invalid parameters - returns 422
curl http://localhost:8000/recommend/abc?top_n=invalid
```

### Checking System Health

```bash
# Health check (always available)
curl http://localhost:8000/ping

# Model status
curl http://localhost:8000/status

# Performance metrics
curl http://localhost:8000/metrics
```

## Error Recovery

### Automatic Recovery

- Model is cached after first load
- Subsequent requests use cached model
- Cache survives individual request failures

### Manual Recovery

```bash
# Reload model after fixing issues
curl -X POST http://localhost:8000/recommend/reload-model

# Or restart the API server
```

## Best Practices

### For API Consumers

1. **Check status endpoint** before making recommendation requests
2. **Handle 503 errors** by retrying after model is available
3. **Use cold-start mode** for new users
4. **Log request IDs** for debugging with support team
5. **Implement exponential backoff** for transient errors

### For Developers

1. **Always validate input** before expensive operations
2. **Use specific exceptions** for different error types
3. **Log with context** (user_id, paths, parameters)
4. **Include tracebacks** for unexpected errors
5. **Test error scenarios** in addition to happy paths

## Testing Error Handling

Run the error handling test suite:

```bash
pytest tests/test_error_handling.py -v
```

This tests:
- Model not found scenarios
- User not found with/without cold-start
- Invalid parameter validation
- Error response structure consistency
- Health check availability during errors

## Monitoring and Alerting

### Recommended Alerts

1. **High 5xx error rate** (> 1% of requests)
2. **Model load failures** (repeated 503 errors)
3. **High cold-start usage** (many unknown users)
4. **Validation error spikes** (client issues)

### Log Queries

Example queries for common error scenarios:

```bash
# Find all model not found errors
jq 'select(.error_type == "ModelNotFoundError")' logs.json

# Find all requests that failed
jq 'select(.level == "ERROR")' logs.json

# Count errors by type
jq -r '.error_type' logs.json | sort | uniq -c

# Find slow requests with errors
jq 'select(.duration_ms > 1000 and .level == "ERROR")' logs.json
```

## Troubleshooting

### Common Issues

**Problem:** API returns 503 Service Unavailable
- **Cause:** Model not trained or files missing
- **Solution:** Run `python scripts/train_model.py data/fake_purchases.csv`

**Problem:** All users return cold-start recommendations
- **Cause:** Model trained but user_id_mapping empty or corrupted
- **Solution:** Retrain model, check CSV data integrity

**Problem:** Recommendations fail with 500 error
- **Cause:** Model version mismatch or corrupted artifacts
- **Solution:** Clear model directory and retrain

**Problem:** Random 404 errors for valid users
- **Cause:** Model reloaded with different training data
- **Solution:** Ensure consistent training data or implement versioning

## Error Handling Architecture

```
┌─────────────┐
│   Request   │
└──────┬──────┘
       │
       v
┌──────────────────┐
│ Request Logging  │
│   Middleware     │
└──────┬───────────┘
       │
       v
┌──────────────────┐
│   Validation     │──── 422 Error ───> User
│   (FastAPI)      │
└──────┬───────────┘
       │
       v
┌──────────────────┐
│   Business       │
│   Logic          │
└──────┬───────────┘
       │
       ├─── ModelNotFoundError ──> 503 Response
       │
       ├─── UserNotFoundError ──> 404 Response
       │
       ├─── HTTPException ──────> Status Code Response
       │
       └─── Exception ──────────> 500 Response
                                    (logged with traceback)
```

## Future Improvements

Planned enhancements:
- [ ] Retry logic for transient errors
- [ ] Circuit breaker for repeated failures
- [ ] Rate limiting with informative errors
- [ ] Batch error handling for bulk requests
- [ ] Error metrics in Prometheus format
- [ ] Structured error codes for programmatic handling

