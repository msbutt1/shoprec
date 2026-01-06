# ShopRec - Product Recommendation System

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Last Updated](https://img.shields.io/badge/last%20updated-2026--01--06-lightgrey)
![Status](https://img.shields.io/badge/status-production--ready-success)

A backend service that powers AI product recommendations for e-commerce platforms. Uses collaborative filtering and content-based filtering to suggest products that users might like.

## Overview

ShopRec is a recommendation engine built with:
- **FastAPI** for the REST API
- **scikit-learn** for machine learning (SVD-based collaborative filtering)
- **Docker** for easy deployment
- **Hybrid recommendations** that combine user behavior patterns with product similarities

The system learns from user purchase history and recommends products based on what similar users bought, plus what products are similar to what the user already likes.

## Features

- **Collaborative Filtering**: Learns user preferences from purchase patterns
- **Content-Based Filtering**: Recommends similar products using embeddings
- **Hybrid Mode**: Combines both methods for better recommendations
- **Cold-Start Handling**: Returns popular items for new users
- **REST API**: Easy to integrate with existing applications
- **CLI Tools**: Command-line scripts for training and testing
- **Docker Support**: Containerized for simple deployment

## Architecture

```
┌─────────────┐
│   Client    │
│ Application │
└──────┬──────┘
       │ HTTP Request
       ▼
┌─────────────────────────────────────────┐
│          FastAPI Server                 │
│  ┌─────────────────────────────────┐    │
│  │  Routes (recommend.py)          │    │
│  │  - GET /recommend/{user_id}     │    │
│  │  - GET /ping                    │    │
│  │  - GET /status                  │    │
│  └──────────┬──────────────────────┘    │
│             │                           │
│  ┌──────────▼──────────────────────┐    │
│  │  Hybrid Recommender             │    │
│  │  ┌────────────┐  ┌────────────┐ │    │
│  │  │ CF Model   │  │  Content   │ │    │
│  │  │   (SVD)    │  │ Embeddings │ │    │
│  │  └────────────┘  └────────────┘ │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────┐
│  Model Files    │
│  - svd_model    │
│  - mappings     │
│  - embeddings   │
└─────────────────┘
```

## Project Structure

```
shoprec/
├── src/
│   ├── api/              # FastAPI application
│   │   ├── main.py       # App entry point
│   │   └── routes/       # API endpoints
│   └── recommender/      # ML core
│       ├── train.py      # Model training
│       ├── infer.py      # Inference/prediction
│       ├── hybrid.py     # Hybrid recommendations
│       ├── embed.py      # Product embeddings
│       └── utils.py      # Helper functions
├── scripts/
│   ├── generate_fake_data.py  # Generate test data
│   ├── train_model.py         # Train model CLI
│   └── predict_cli.py         # Prediction CLI
├── data/                 # Purchase data (CSV)
├── models/               # Trained model files
├── tests/                # Unit tests
├── notebooks/            # Jupyter notebooks
├── Dockerfile            # Docker container config
└── requirements.txt      # Python dependencies
```

## Setup

### Option 1: Local Setup with pip/venv

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd shoprec
```

2. **Create a virtual environment**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Generate sample data** (optional, for testing)
```bash
python scripts/generate_fake_data.py
```

5. **Train the model**
```bash
python scripts/train_model.py data/fake_purchases.csv --output-dir models
```

6. **Run the API server**
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Option 2: Docker Setup

1. **Build the Docker image**
```bash
docker build -t shoprec:latest .
```

2. **Run the container**
```bash
docker run -p 8000:8000 \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/models:/app/models \
  shoprec:latest
```

**Note**: The container needs access to the `models/` directory. You can either:
- Mount the directory as a volume (as shown above)
- Train the model inside the container
- Build the image with pre-trained models

## Usage

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/ping
```

Response:
```json
{
  "status": "ok"
}
```

#### 2. Get Recommendations (Hybrid Mode)
```bash
curl http://localhost:8000/recommend/42?top_n=5
```

Response:
```json
{
  "user_id": 42,
  "recommendations": [9, 64, 61, 25, 66],
  "model_version": "0.1.0",
  "scores": null
}
```

#### 3. Get Recommendations (CF-only Mode)
```bash
curl "http://localhost:8000/recommend/42?top_n=5&mode=cf"
```

#### 4. Get Recommendations with Score Breakdown
```bash
curl "http://localhost:8000/recommend/42?top_n=5&explain=true"
```

Response:
```json
{
  "user_id": 42,
  "recommendations": [9, 64, 61, 25, 66],
  "model_version": "0.1.0",
  "scores": {
    "cf_scores": {
      "9": 1.0,
      "64": 0.9537,
      "61": 0.8713,
      "25": 0.7809,
      "66": 0.7749
    },
    "content_scores": {},
    "hybrid_scores": {
      "9": 1.0,
      "64": 0.9537,
      "61": 0.8713,
      "25": 0.7809,
      "66": 0.7749
    },
    "cf_weight": 0.7,
    "content_weight": 0.3
  }
}
```

#### 5. Check Model Status
```bash
curl http://localhost:8000/status
```

Response:
```json
{
  "model_loaded": true,
  "timestamp_last_loaded": "2025-01-06T10:30:45.123456Z",
  "num_users": 50,
  "num_products": 100
}
```

### CLI Tools

#### Generate Sample Data
```bash
python scripts/generate_fake_data.py
```

Creates `data/fake_purchases.csv` with 50 users, 100 products, and 1000 random purchases.

#### Train Model
```bash
# Basic training
python scripts/train_model.py data/fake_purchases.csv

# With custom settings
python scripts/train_model.py data/fake_purchases.csv \
  --output-dir models/prod \
  --n-components 30 \
  --embedding-method tfidf
```

#### Get Predictions (CLI)
```bash
# Get recommendations for user 42
python scripts/predict_cli.py 42

# Get 5 recommendations using CF-only mode
python scripts/predict_cli.py 42 --top-n 5 --mode cf

# Get recommendations with score breakdown
python scripts/predict_cli.py 42 --explain
```

## Training Configuration

You can customize training using command-line arguments:

```bash
python scripts/train_model.py data/purchases.csv \
  --output-dir models/experiment_1 \
  --n-components 50 \           # Number of latent features
  --n-iter 10 \                 # SVD iterations
  --random-state 42 \           # Random seed
  --embedding-method tfidf \    # Embedding method (random or tfidf)
  --embedding-dim 50            # Embedding dimensions
```

## Development

### Run Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_api.py -v
```

### Explore Data with Jupyter Notebook
```bash
jupyter notebook notebooks/explore.ipynb
```

The notebook includes:
- Dataset analysis
- User-product matrix visualization
- SVD component analysis
- Live inference testing

### Notebook Output Example

![Notebook Visualization](docs/notebook_example.png)

*The notebook shows the user-product interaction matrix as a heatmap, SVD component visualization, and recommendation results for multiple users.*

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Model Details

### Collaborative Filtering (CF)
- Uses **Truncated SVD** (matrix factorization)
- Learns latent features from user-product interactions
- Predicts scores based on similar users' preferences
- Good at finding patterns in user behavior

### Content-Based Filtering
- Uses **TF-IDF** or random vectors for product embeddings
- Recommends products similar to what user already likes
- Handles new products better than CF
- Requires product metadata (simulated in this demo)

### Hybrid Approach
- Combines CF (70%) and content (30%) scores
- Normalizes scores before combining
- Falls back to CF if embeddings unavailable
- Better overall performance and diversity

## Troubleshooting

### Model not found error
```
Error: Model not found in models. Please train a model first.
```

**Solution**: Train a model first using:
```bash
python scripts/train_model.py data/fake_purchases.csv
```

### Port already in use
```
Error: Address already in use
```

**Solution**: Either stop the other process or use a different port:
```bash
uvicorn src.api.main:app --port 8001
```

### Docker volume mount issues on Windows
Use PowerShell syntax for volume mounting:
```powershell
docker run -p 8000:8000 -v ${PWD}/models:/app/models shoprec:latest
```

## Performance

On a typical dataset:
- Training: ~1-5 seconds for 1000 interactions
- Inference: <10ms per user
- API response time: 20-50ms

## Future Enhancements

- [ ] Real product metadata integration
- [ ] A/B testing framework
- [ ] User feedback loop
- [ ] Real-time model updates
- [ ] Caching layer for frequent users
- [ ] Batch recommendation generation
- [ ] Performance metrics dashboard

## License

This project is for educational purposes.

## Contributing

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Contact

For questions or issues, please open an issue on GitHub.
