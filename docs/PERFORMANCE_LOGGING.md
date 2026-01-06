# Performance Logging

ShopRec includes comprehensive performance logging to track timing and help identify bottlenecks.

## Features

### 1. Structured JSON Logging
All logs are output in JSON format for easy parsing and aggregation:

```json
{
  "timestamp": "2026-01-06T21:07:04.662795Z",
  "level": "INFO",
  "logger": "src.recommender.infer",
  "message": "Model loaded",
  "module": "infer",
  "function": "recommend_products_for_user",
  "line": 53,
  "user_id": 1,
  "load_time_ms": 2.18,
  "num_users": 50,
  "num_products": 100
}
```

### 2. Model Loading Time
Tracks how long it takes to load the model from disk:

```json
{
  "message": "Model loaded",
  "load_time_ms": 2.18,
  "num_users": 50,
  "num_products": 100
}
```

### 3. Scoring Time
Measures the time to compute recommendations:

```json
{
  "message": "Recommendations generated",
  "scoring_time_ms": 0.19,
  "total_time_ms": 3.46
}
```

### 4. Hybrid Recommender Timing
Breaks down time for CF and content-based components:

```json
{
  "message": "Scores computed",
  "cf_time_ms": 0.15,
  "content_time_ms": 0.04,
  "num_cf_scores": 100,
  "num_content_scores": 100
}
```

### 5. HTTP Request Duration
FastAPI middleware logs request duration for all API calls:

```json
{
  "message": "Request completed",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "method": "GET",
  "path": "/recommend/42",
  "status_code": 200,
  "duration_ms": 12.34
}
```

## Usage

### Enable JSON Logging
JSON logging is enabled by default in the API. For CLI scripts:

```python
from src.api.logging_config import setup_logging

setup_logging(log_level="INFO")
```

### Parse Logs
Use `jq` or similar tools to parse JSON logs:

```bash
# Get all recommendation timings
python -m uvicorn src.api.main:app 2>&1 | jq 'select(.message == "Recommendations generated")'

# Calculate average request duration
python -m uvicorn src.api.main:app 2>&1 | jq 'select(.message == "Request completed") | .duration_ms' | awk '{sum+=$1; n++} END {print sum/n}'
```

### PowerShell Examples
```powershell
# Get timing logs
python scripts/predict_cli.py 1 --verbose 2>&1 | Select-String "time_ms"

# Parse specific log message
python -c "..." 2>&1 | Select-String "Model loaded" | ConvertFrom-Json | ConvertTo-Json
```

## Performance Metrics

Typical performance on standard hardware:

| Operation | Time (ms) | Notes |
|-----------|-----------|-------|
| Model Load | 2-5 | First load, includes file I/O |
| Model Load (cached) | 0.1 | Subsequent loads use cache |
| CF Scoring | 0.2-0.5 | For 100 products |
| Content Scoring | 0.1-0.3 | With embeddings |
| Total Inference | 3-10 | Cold start (first request) |
| Total Inference | 1-2 | Warm (cached model) |
| API Request | 10-50 | End-to-end including network |

## Monitoring Integration

The structured JSON logs can be easily integrated with monitoring tools:

- **ELK Stack**: Logstash can parse JSON logs directly
- **Splunk**: JSON format is natively supported
- **CloudWatch**: Use CloudWatch Logs Insights
- **Grafana**: Query logs and create dashboards

## Example Queries

### Find slow requests
```
jq 'select(.duration_ms > 100)'
```

### Average scoring time per user
```
jq 'select(.message == "Recommendations generated") | {user_id, scoring_time_ms}'
```

### Track model load frequency
```
jq 'select(.message == "Model loaded")' | wc -l
```

## Debug Mode

Enable debug logging for more detailed timing:

```python
setup_logging(log_level="DEBUG")
```

This adds additional timing for:
- Individual product score computation
- Normalization steps
- Filtering operations

