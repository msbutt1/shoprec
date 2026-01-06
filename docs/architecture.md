# ShopRec Architecture

Internal technical documentation for the ShopRec recommendation system.

## System Overview

ShopRec is a production-ready recommendation service that combines collaborative filtering (CF) and content-based filtering to generate product recommendations. The system is designed for e-commerce scale with support for implicit feedback data, cold-start scenarios, and hybrid recommendation strategies.

### High-Level Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP/REST
       v
┌─────────────────────────────────────┐
│         FastAPI Application          │
│  ┌───────────────────────────────┐   │
│  │  Request Logging Middleware  │   │
│  └───────────────┬──────────────┘   │
│                  │                    │
│  ┌───────────────▼───────────────┐   │
│  │   Recommendation Endpoints     │   │
│  │   /recommend/{user_id}        │   │
│  │   /status, /metrics, /ping     │   │
│  └───────────────┬───────────────┘   │
└──────────────────┼───────────────────┘
                   │
                   v
┌─────────────────────────────────────┐
│      Recommendation Engine          │
│  ┌──────────────────────────────┐  │
│  │   Hybrid Recommender          │  │
│  │   (CF + Content-Based)        │  │
│  └───────────┬──────────────────┘  │
│              │                      │
│  ┌───────────▼──────────┐          │
│  │  CF Module            │          │
│  │  (TruncatedSVD)       │          │
│  └───────────┬──────────┘          │
│              │                      │
│  ┌───────────▼──────────┐          │
│  │  Content Module       │          │
│  │  (Product Embeddings) │          │
│  └───────────────────────┘          │
└─────────────────────────────────────┘
                   │
                   v
┌─────────────────────────────────────┐
│      Model Artifacts (Disk)          │
│  - svd_model.joblib                 │
│  - user_id_mapping.joblib            │
│  - product_id_mapping.joblib        │
│  - product_embeddings.joblib        │
└─────────────────────────────────────┘
```

## Training Flow

### Overview

The training pipeline transforms raw purchase data into a trained recommendation model and supporting artifacts.

### Step-by-Step Flow

```
1. Data Ingestion
   └─> CSV file (user_id, product_id, timestamp)
       └─> Validation: column presence, non-empty, data types

2. Matrix Construction
   └─> load_csv_to_matrix()
       ├─> Extract unique users and products
       ├─> Create ID → index mappings
       ├─> Build sparse CSR matrix (binary or weighted)
       └─> Return: (matrix, user_id_to_idx, product_id_to_idx)

3. Model Training
   └─> train_svd_model()
       ├─> Validate matrix dimensions vs n_components
       ├─> Auto-adjust n_components if needed
       ├─> Fit TruncatedSVD on user-product matrix
       └─> Return: Trained TruncatedSVD model

4. Embedding Generation (Optional)
   └─> generate_embeddings()
       ├─> Method: "random" or "tfidf"
       ├─> Create ProductEmbeddings object
       └─> Save to disk

5. Artifact Persistence
   └─> save_model_artifacts()
       ├─> Save SVD model (joblib)
       ├─> Save user_id_to_idx mapping
       ├─> Save product_id_to_idx mapping
       └─> Save embeddings (if generated)
```

### Key Components

**TrainingConfig** (`src/recommender/train.py`)
- Dataclass holding all training parameters
- Validates configuration on initialization
- Supports embedding generation configuration

**Matrix Construction** (`src/recommender/utils.py::load_csv_to_matrix`)
- Handles binary (purchase/no-purchase) or weighted interactions
- Creates sparse CSR matrix for memory efficiency
- Generates ID mappings for O(1) lookups during inference

**SVD Training** (`src/recommender/train.py::train_svd_model`)
- Uses scikit-learn's TruncatedSVD
- Automatically adjusts `n_components` to `min(n_users, n_products) - 1`
- Tracks explained variance ratio for quality assessment

**Embedding Generation** (`src/recommender/embed.py`)
- Random embeddings: Fast, no data required
- TF-IDF embeddings: Requires product metadata (simulated if missing)
- Normalized to unit vectors for cosine similarity

### Training Artifacts

| Artifact | Format | Purpose |
|----------|--------|---------|
| `svd_model.joblib` | scikit-learn model | Core CF model |
| `user_id_mapping.joblib` | Dict[int, int] | Map user_id → matrix index |
| `product_id_mapping.joblib` | Dict[int, int] | Map product_id → matrix index |
| `product_embeddings.joblib` | ProductEmbeddings | Content-based features |
| `embedding_metadata.joblib` | Dict | Embedding configuration |

## Inference Flow

### Overview

The inference pipeline serves real-time recommendations via REST API, supporting both CF-only and hybrid modes.

### Step-by-Step Flow

```
1. API Request
   └─> GET /recommend/{user_id}?mode=hybrid&top_n=10
       ├─> Extract parameters (user_id, top_n, mode, explain)
       └─> Validate input (type checking, range validation)

2. Model Loading (Lazy, Cached)
   └─> load_model_if_needed()
       ├─> Check cache (_model_cache)
       ├─> If missing: load_model_artifacts()
       │   ├─> Load SVD model
       │   ├─> Load user_id_to_idx mapping
       │   ├─> Load product_id_to_idx mapping
       │   └─> Cache in memory
       └─> Return cached model artifacts

3. User Lookup
   └─> Check if user_id in user_id_to_idx
       ├─> If not found:
       │   ├─> If allow_cold_start=True: cold-start strategy
       │   └─> If allow_cold_start=False: raise UserNotFoundError (404)
       └─> If found: proceed to recommendation

4. Recommendation Generation
   ├─> Mode: "hybrid"
   │   └─> HybridRecommender.recommend()
   │       ├─> Compute CF scores (via SVD)
   │       ├─> Compute content scores (via embeddings)
   │       ├─> Normalize both score sets to [0, 1]
   │       ├─> Weighted combination: w_cf * CF + w_content * Content
   │       └─> Return top-N products
   │
   └─> Mode: "cf"
       └─> _compute_recommendations()
           ├─> Transform user vector to latent space
           ├─> Reconstruct product scores
           ├─> Filter out already-purchased products
           └─> Return top-N products

5. Response Construction
   └─> RecommendationResponse
       ├─> user_id
       ├─> recommendations: List[int]
       ├─> model_version: str
       └─> scores: Optional[Dict] (if explain=true)

6. Metrics Recording
   └─> metrics_service.record_inference(latency_ms)
       └─> Update: count, total_latency, min, max
```

### Key Components

**Model Caching** (`src/api/routes/recommend.py`)
- Global `_model_cache` stores loaded artifacts
- Persists across requests (process-level)
- Cleared on `/reload-model` endpoint

**Hybrid Recommender** (`src/recommender/hybrid.py`)
- Combines CF and content-based scores
- Configurable weights (default: 70% CF, 30% content)
- Falls back to CF-only if embeddings unavailable
- Supports explainability (score breakdown)

**Cold-Start Strategy** (`src/recommender/infer.py::_handle_cold_start_user`)
- Returns first N products from catalog (sorted by ID)
- Simple fallback for unknown users
- Can be disabled via `allow_cold_start=False`

**Error Handling** (`src/api/main.py`)
- Custom exception handlers for ModelNotFoundError, UserNotFoundError
- Global handler catches unexpected exceptions
- Structured logging with full tracebacks

## Key Modules and Interfaces

### Core Recommendation Modules

#### `src/recommender/train.py`

**Purpose**: Model training pipeline

**Key Functions**:
- `train_svd_model()`: Train TruncatedSVD on user-product matrix
- `train_with_config()`: High-level training with configuration
- `train_basic_cf_model()`: Simplified training interface

**Interfaces**:
```python
TrainingConfig(
    csv_path: str,
    output_dir: str = "models",
    n_components: int = 50,
    n_iter: int = 10,
    random_state: int = 42,
    generate_embeddings: bool = True,
    embedding_method: str = "random",
    embedding_dim: int = 50
)
```

**Dependencies**: `scipy.sparse`, `sklearn.decomposition.TruncatedSVD`

#### `src/recommender/infer.py`

**Purpose**: Recommendation inference

**Key Functions**:
- `recommend_products_for_user()`: Main inference function
- `_compute_recommendations()`: CF scoring logic
- `_handle_cold_start_user()`: Fallback for unknown users

**Interfaces**:
```python
recommend_products_for_user(
    user_id: int,
    model_path: str = "models",
    top_n: int = 5,
    user_product_matrix: Optional[csr_matrix] = None,
    allow_cold_start: bool = True
) -> List[int]
```

**Exceptions**:
- `ModelNotFoundError`: Model files missing
- `UserNotFoundError`: User not in training data (if cold-start disabled)

#### `src/recommender/hybrid.py`

**Purpose**: Hybrid recommendation combining CF and content-based

**Key Classes**:
- `HybridRecommender`: Main hybrid recommendation class

**Interfaces**:
```python
HybridRecommender(
    model: TruncatedSVD,
    user_id_to_idx: Dict[int, int],
    product_id_to_idx: Dict[int, int],
    embeddings: Optional[ProductEmbeddings] = None,
    cf_weight: float = 0.7,
    content_weight: float = 0.3
)

HybridRecommender.recommend(
    user_id: int,
    top_n: int = 10,
    user_product_matrix: Optional[csr_matrix] = None,
    return_scores: bool = False
) -> Union[List[int], Tuple[List[int], Dict]]
```

**Scoring Strategy**:
1. Compute CF scores (SVD-based)
2. Compute content scores (embedding similarity)
3. Normalize both to [0, 1]
4. Weighted combination: `w_cf * CF + w_content * Content`
5. Return top-N products

#### `src/recommender/embed.py`

**Purpose**: Product embedding generation and management

**Key Classes**:
- `ProductEmbeddings`: Manages product embeddings

**Interfaces**:
```python
ProductEmbeddings(
    product_id_to_idx: Dict[int, int],
    embeddings: np.ndarray,  # Shape: (n_products, embedding_dim)
    method: str = "random"
)

ProductEmbeddings.get_embedding(product_id: int) -> np.ndarray
ProductEmbeddings.compute_similarity(pid1: int, pid2: int) -> float
ProductEmbeddings.get_similar_products(product_id: int, top_n: int) -> List[int]
```

**Embedding Methods**:
- `random`: Random unit vectors (fast, no data required)
- `tfidf`: TF-IDF over product metadata (requires metadata)

#### `src/recommender/utils.py`

**Purpose**: Data loading and model persistence utilities

**Key Functions**:
- `load_csv_to_matrix()`: Convert CSV to sparse matrix
- `save_model_artifacts()`: Persist model and mappings
- `load_model_artifacts()`: Load model from disk
- `check_model_exists()`: Verify model files present

**Interfaces**:
```python
load_csv_to_matrix(
    csv_path: str,
    user_col: str = "user_id",
    item_col: str = "product_id",
    value_col: Optional[str] = None,
    binary: bool = True
) -> Tuple[csr_matrix, Dict[int, int], Dict[int, int]]
```

### API Modules

#### `src/api/main.py`

**Purpose**: FastAPI application setup and global error handling

**Key Components**:
- FastAPI app instance
- Request logging middleware
- Global exception handlers
- Health check endpoints (`/ping`, `/status`, `/metrics`)

**Exception Handlers**:
- `ModelNotFoundError` → 503 Service Unavailable
- `UserNotFoundError` → 404 Not Found
- `RequestValidationError` → 422 Unprocessable Entity
- `Exception` → 500 Internal Server Error (with traceback logging)

#### `src/api/routes/recommend.py`

**Purpose**: Recommendation API endpoints

**Key Functions**:
- `get_recommendations()`: Main recommendation endpoint
- `load_model_if_needed()`: Lazy model loading with caching
- `reload_model()`: Force model reload
- `get_model_status()`: Model status information

**Endpoints**:
- `GET /recommend/{user_id}`: Get recommendations
- `POST /recommend/reload-model`: Reload model from disk

**Query Parameters**:
- `top_n`: Number of recommendations (default: 10)
- `mode`: "hybrid" or "cf" (default: "hybrid")
- `explain`: Include score breakdown (default: false)
- `allow_cold_start`: Enable cold-start for unknown users (default: true)
- `cf_weight`, `content_weight`: Hybrid weights (default: 0.7, 0.3)

#### `src/api/logging_config.py`

**Purpose**: Structured JSON logging configuration

**Key Components**:
- `JSONFormatter`: Custom formatter for JSON logs
- `RequestLoggingMiddleware`: Logs all HTTP requests/responses
- `setup_logging()`: Configure logging system

**Log Structure**:
```json
{
  "timestamp": "ISO8601",
  "level": "INFO|WARNING|ERROR",
  "logger": "module.path",
  "message": "Human-readable message",
  "module": "module_name",
  "function": "function_name",
  "line": 123,
  "request_id": "UUID",
  "user_id": 42,
  "duration_ms": 12.5
}
```

#### `src/api/metrics.py`

**Purpose**: Application metrics collection

**Key Components**:
- `MetricsService`: Singleton for metrics tracking
- Thread-safe counters for inference count and latency

**Interfaces**:
```python
metrics_service.record_inference(latency_ms: float) -> None
metrics_service.get_metrics() -> Dict[str, float]
```

**Metrics Exposed**:
- `inference_count`: Total recommendations served
- `average_latency_ms`: Average latency
- `min_latency_ms`: Minimum latency observed
- `max_latency_ms`: Maximum latency observed

## Data Assumptions

### Input Data Format

**Training Data (CSV)**:
- Required columns: `user_id`, `product_id`
- Optional columns: `timestamp` (for temporal analysis, currently unused)
- Format: CSV with header row
- Encoding: UTF-8

**Data Characteristics**:
- **Implicit feedback only**: No explicit ratings (e.g., 1-5 stars)
- Binary interactions: Purchase = 1, No purchase = 0
- No negative feedback: Absence of interaction ≠ dislike
- Sparse matrix: Typically < 20% density for e-commerce

### User and Product IDs

- **Type**: Integer (int)
- **Uniqueness**: Must be unique within user/product sets
- **Range**: No restrictions (can be any positive integer)
- **Gaps allowed**: IDs don't need to be consecutive (e.g., [1, 5, 100] is valid)

### Matrix Assumptions

- **Shape**: (n_users, n_products)
- **Sparsity**: Expected to be sparse (< 50% density)
- **Storage**: CSR (Compressed Sparse Row) format for efficiency
- **Density**: Typically 5-20% for e-commerce purchase data

### Model Constraints

- **n_components**: Must be < min(n_users, n_products)
- **Auto-adjustment**: System automatically reduces n_components if too large
- **Minimum data**: Requires at least 1 user and 1 product
- **Non-empty matrix**: Must have at least 1 interaction

### Cold-Start Assumptions

- **New users**: Not in training data, no purchase history
- **New products**: Not in training data, no purchase history
- **Strategy**: Return first N products (sorted by ID) as fallback
- **No personalization**: Cold-start recommendations are not personalized

## Tradeoffs and Design Decisions

### Collaborative Filtering (CF)

**Advantages**:
- Captures user-item interaction patterns
- Works well with implicit feedback
- No need for product metadata
- Handles popularity bias naturally

**Limitations**:
- Cold-start problem (new users/products)
- Sparsity issues with large catalogs
- Limited explainability
- Requires sufficient interaction history

**Current Implementation**:
- Uses TruncatedSVD for matrix factorization
- Fixed n_components (default: 50, auto-adjusted)
- Binary interaction matrix (purchase/no-purchase)
- No temporal modeling (timestamp ignored)

### Content-Based Filtering

**Advantages**:
- Handles new products (no cold-start)
- Explainable (based on product features)
- Works for niche items
- Can incorporate product metadata

**Limitations**:
- Requires product metadata/embeddings
- May over-specialize (filter bubble)
- Limited by embedding quality
- Doesn't capture user preferences directly

**Current Implementation**:
- Random embeddings (no metadata required)
- TF-IDF embeddings (requires simulated metadata)
- Cosine similarity for product similarity
- Normalized to unit vectors

### Hybrid Approach

**Advantages**:
- Combines strengths of both methods
- Handles cold-start better than CF alone
- More robust to data sparsity
- Configurable weights for tuning

**Limitations**:
- More complex than single method
- Requires embeddings (optional but recommended)
- Weight tuning needed for optimal performance
- Higher computational cost

**Current Implementation**:
- Default: 70% CF, 30% content
- Falls back to CF-only if embeddings missing
- Normalizes scores before combination
- Supports explainability (score breakdown)

### Model Caching

**Decision**: Cache model in memory after first load

**Advantages**:
- Fast subsequent requests (< 1ms vs 2-5ms)
- Reduces disk I/O
- Lower latency for users

**Tradeoffs**:
- Memory usage (model + mappings + embeddings)
- Stale model if files updated on disk
- Requires `/reload-model` endpoint for updates

**Future Consideration**: Model versioning or file watching

### Error Handling

**Decision**: Comprehensive error handling with custom exceptions

**Advantages**:
- Clear error messages for debugging
- Appropriate HTTP status codes
- Structured logging for monitoring
- Graceful degradation

**Tradeoffs**:
- More code complexity
- Requires exception hierarchy maintenance
- Potential for over-engineering

### Cold-Start Strategy

**Decision**: Simple fallback (first N products by ID)

**Advantages**:
- Always returns recommendations
- No additional computation
- Predictable behavior

**Limitations**:
- Not personalized
- May not be relevant
- Doesn't consider product popularity

**Future Consideration**: Popularity-based or demographic-based cold-start

## Future Extensions

### Personalization Enhancements

**User Profiles**:
- Demographic data (age, location, preferences)
- Explicit preferences (categories, brands)
- Session-based behavior tracking

**Implementation Path**:
1. Extend `TrainingConfig` to accept user metadata
2. Create user profile embeddings
3. Incorporate into hybrid scoring
4. Add user profile endpoint

**Challenges**:
- Privacy concerns (GDPR compliance)
- Data collection infrastructure
- Profile update frequency

### Advanced Cold-Start

**Popularity-Based**:
- Rank products by purchase frequency
- Return top-N popular items
- Requires aggregate statistics

**Demographic-Based**:
- Cluster users by demographics
- Use cluster preferences for new users
- Requires demographic data

**Content-Based for Users**:
- Infer preferences from initial interactions
- Update recommendations as user engages
- Requires real-time model updates

**Implementation Path**:
1. Add popularity statistics to training
2. Implement popularity-based cold-start
3. Add demographic clustering
4. Create adaptive cold-start strategy

### Temporal Modeling

**Time-Aware Recommendations**:
- Incorporate purchase timestamps
- Weight recent interactions more
- Handle seasonality

**Implementation Path**:
1. Modify matrix construction to include temporal weights
2. Update SVD training to handle weighted interactions
3. Add time decay function
4. Test on time-series data

**Challenges**:
- Increased model complexity
- Requires historical data
- Computational overhead

### Real-Time Updates

**Incremental Learning**:
- Update model with new interactions
- Avoid full retraining
- Maintain model freshness

**Implementation Path**:
1. Implement incremental SVD updates
2. Add interaction streaming pipeline
3. Create model update scheduler
4. Add A/B testing framework

**Challenges**:
- Model consistency
- Update frequency vs. stability tradeoff
- Distributed system coordination

### Advanced Embeddings

**Deep Learning Embeddings**:
- Train neural embeddings on interaction data
- Capture complex product relationships
- Better than random/TF-IDF

**Implementation Path**:
1. Implement neural collaborative filtering
2. Train product embeddings via autoencoders
3. Integrate with hybrid recommender
4. Add GPU support for training

**Challenges**:
- Training time and resources
- Model complexity
- Hyperparameter tuning

### Explainability

**Current State**:
- Score breakdown (CF, content, hybrid)
- Basic explainability via `explain=true`

**Future Enhancements**:
- "Why this product?" explanations
- Feature importance for content scores
- Similar user reasoning for CF
- Visual explanations

**Implementation Path**:
1. Add explanation generation module
2. Create explanation templates
3. Integrate with API response
4. Add explanation UI components

### Scalability

**Current Limitations**:
- Single-process model caching
- No distributed inference
- Limited to in-memory matrices

**Future Enhancements**:
- Distributed model serving
- Sharding for large catalogs
- Caching layer (Redis)
- Load balancing

**Implementation Path**:
1. Add Redis for model caching
2. Implement model sharding
3. Create distributed inference service
4. Add load balancer configuration

### A/B Testing Framework

**Current State**:
- Single model version
- No experimentation support

**Future Enhancements**:
- Multiple model versions
- Traffic splitting
- Metrics collection per variant
- Statistical significance testing

**Implementation Path**:
1. Add model versioning
2. Implement traffic splitting
3. Create metrics collection per variant
4. Add analysis dashboard

## Performance Characteristics

### Training

- **Time Complexity**: O(n_users × n_products × n_components × n_iter)
- **Space Complexity**: O(n_users × n_products) for matrix, O(n_components × n_products) for model
- **Typical Performance**: 
  - 50 users, 100 products: < 1 second
  - 10K users, 1K products: ~10-30 seconds
  - 100K users, 10K products: ~5-15 minutes

### Inference

- **Time Complexity**: O(n_components × n_products) for CF, O(embedding_dim × n_products) for content
- **Space Complexity**: O(n_components × n_products) for model, O(n_products × embedding_dim) for embeddings
- **Typical Performance**:
  - Model load (first request): 2-5ms
  - Model load (cached): < 1ms
  - Recommendation generation: 0.5-2ms
  - Total API latency: 5-15ms

### Scalability Limits

- **Current**: ~100K users, ~10K products (in-memory)
- **Bottlenecks**: 
  - Matrix size (memory)
  - Model load time (disk I/O)
  - Embedding similarity computation
- **Mitigation**: 
  - Sparse matrix storage
  - Model caching
  - Vectorized operations

## Security Considerations

### Input Validation

- User IDs: Validated as integers
- Query parameters: Type-checked and range-validated
- File paths: Sanitized to prevent directory traversal

### Error Information

- Internal errors: Full traceback logged, generic message returned
- User-facing errors: Informative but not revealing system internals
- Logging: Structured logs exclude sensitive data

### Model Security

- Model files: Stored on filesystem (consider encryption for sensitive data)
- Model loading: Validates file existence and integrity
- No remote model loading: All models must be local

## Monitoring and Observability

### Metrics

- **Inference Count**: Total recommendations served
- **Latency**: Min, max, average (milliseconds)
- **Error Rates**: By error type (404, 500, 503)
- **Model Status**: Loaded/unloaded, last load time

### Logging

- **Structured JSON logs**: Easy parsing and aggregation
- **Request IDs**: Trace requests across logs
- **Performance timing**: Model load, scoring, total time
- **Error context**: Full tracebacks for debugging

### Health Checks

- `/ping`: Basic availability check
- `/status`: Model status and metadata
- `/metrics`: Performance metrics

## Conclusion

ShopRec provides a solid foundation for e-commerce recommendations with support for hybrid filtering, cold-start handling, and production-ready error handling. The architecture is designed for extensibility, with clear interfaces and modular components that enable future enhancements in personalization, temporal modeling, and scalability.

