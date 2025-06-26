# FastAPI LLM Enhanced Toxic Classifier

A FastAPI-based REST API for toxic content classification using LLM and Qdrant search capabilities.

## 🚀 Features

- **RESTful API**: Easy-to-use HTTP endpoints for toxic content classification
- **Flexible Parameters**: Configurable confidence thresholds, search limits, and model options
- **Comprehensive Results**: Returns detected labels, search results, and LLM analysis
- **Health Monitoring**: Built-in health check endpoint
- **Auto-generated Documentation**: Interactive API docs with Swagger UI

## 📋 Prerequisites

1. **Python 3.8+**
2. **Qdrant Vector Database** (running locally)
3. **Jigsaw Dataset** loaded into Qdrant
4. **LLM Enhanced Classifier** from parent directory

## 🛠️ Installation

1. **Install Dependencies**:
   ```bash
   cd fastapi_classifier
   pip install -r requirements.txt
   ```

2. **Start Qdrant** (if not running):
   ```bash
   cd ..
   ./start_qdrant_simple.sh
   ```

3. **Load Jigsaw Data** (if not already done):
   ```bash
   python load_jigsaw_to_qdrant_enhanced.py
   ```

## 🎯 Usage

### Starting the API Server

```bash
cd fastapi_classifier
python main.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### API Endpoints

#### 1. Health Check
```bash
GET /health
```

Returns the health status of the API and Qdrant connection.

#### 2. Root Information
```bash
GET /
```

Returns API information and available endpoints.

#### 3. Simple Classification
```bash
POST /classify/simple
Content-Type: application/json

"Your text to classify"
```

Quick classification with default parameters.

#### 4. Full Classification
```bash
POST /classify
Content-Type: application/json

{
  "text": "Your text to classify",
  "confidence_threshold": 0.3,
  "search_limit": 3,
  "use_local_llm": true,
  "qdrant_host": "localhost",
  "qdrant_port": 6333,
  "collection_name": "jigsaw_toxic_comments_enhanced"
}
```

Full classification with customizable parameters.

## 📊 Request/Response Format

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to classify |
| `confidence_threshold` | float | 0.3 | Classification confidence threshold |
| `search_limit` | int | 3 | Number of similar examples to find |
| `use_local_llm` | boolean | true | Whether to use local LLM analysis |
| `qdrant_host` | string | "localhost" | Qdrant server host |
| `qdrant_port` | int | 6333 | Qdrant server port |
| `collection_name` | string | "jigsaw_toxic_comments_enhanced" | Qdrant collection name |

### Response Format

```json
{
  "user_input": "Your input text",
  "detected_labels": {
    "toxic": 0.856,
    "insult": 0.723,
    "threat": 0.612
  },
  "search_results": [
    {
      "rank": 1,
      "similarity_score": 0.892,
      "text": "Similar example from dataset",
      "labels": {
        "toxic": 1,
        "severe_toxic": 0,
        "obscene": 0,
        "threat": 1,
        "insult": 1,
        "identity_hate": 0
      },
      "primary_emotion": "anger",
      "toxicity_score": 0.85
    }
  ],
  "llm_response": {
    "analysis_type": "local_llm",
    "classification": "offensive",
    "llm_response": "The user text is offensive",
    "confidence": "medium"
  },
  "processing_info": {
    "confidence_threshold": 0.3,
    "search_limit": 3,
    "use_local_llm": true,
    "similar_examples_found": 3
  }
}
```

## 🧪 Testing

### Using the Test Script

```bash
python test_api.py
```

This will test all endpoints with various test cases.

### Manual Testing with curl

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Simple Classification
```bash
curl -X POST "http://localhost:8000/classify/simple" \
     -H "Content-Type: application/json" \
     -d '"You are a stupid idiot and I hope you die"'
```

#### Full Classification
```bash
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "You are a stupid idiot and I hope you die",
       "confidence_threshold": 0.3,
       "search_limit": 3,
       "use_local_llm": true
     }'
```

### Using Python Requests

```python
import requests

# Simple classification
response = requests.post(
    "http://localhost:8000/classify/simple",
    json="You are a stupid idiot and I hope you die"
)
result = response.json()
print(f"Classification: {result['llm_response']['classification']}")

# Full classification with parameters
response = requests.post(
    "http://localhost:8000/classify",
    json={
        "text": "You are a stupid idiot and I hope you die",
        "confidence_threshold": 0.2,
        "search_limit": 5,
        "use_local_llm": True
    }
)
result = response.json()
print(f"Detected Labels: {result['detected_labels']}")
print(f"Search Results: {len(result['search_results'])}")
```

## 🔧 Configuration

### Environment Variables

You can set these environment variables to customize the API:

```bash
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export COLLECTION_NAME=jigsaw_toxic_comments_enhanced
export USE_LOCAL_LLM=true
```

### API Configuration

The API can be configured by modifying `main.py`:

```python
# Change default parameters
class ClassificationRequest(BaseModel):
    text: str
    confidence_threshold: Optional[float] = 0.2  # Lower default
    search_limit: Optional[int] = 5  # More examples
    use_local_llm: Optional[bool] = True
    # ... other parameters
```

## 📈 Performance Considerations

### Optimization Tips

1. **Reduce Search Limit**: Lower `search_limit` for faster responses
2. **Disable LLM**: Set `use_local_llm=false` for rule-based analysis only
3. **Adjust Confidence**: Higher `confidence_threshold` reduces false positives
4. **Connection Pooling**: Use connection pooling for high-traffic scenarios

### Expected Response Times

- **Simple Classification**: 2-5 seconds
- **Full Classification (LLM)**: 3-8 seconds
- **Rule-based Only**: 1-3 seconds

## 🔍 Troubleshooting

### Common Issues

1. **Classifier Not Initialized**
   ```
   Error: Classifier not initialized
   ```
   - **Solution**: Check that all dependencies are installed
   - **Check**: Run `python main.py` and check logs

2. **Qdrant Connection Failed**
   ```
   Error: Classification failed: Connection refused
   ```
   - **Solution**: Start Qdrant with `./start_qdrant_simple.sh`
   - **Check**: Verify Qdrant is running on port 6333

3. **Collection Not Found**
   ```
   Error: Collection 'jigsaw_toxic_comments_enhanced' not found
   ```
   - **Solution**: Load data with `python load_jigsaw_to_qdrant_enhanced.py`
   - **Check**: Verify collection exists in Qdrant

### Debug Mode

Enable debug logging by modifying `main.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 🚀 Deployment

### Production Deployment

For production deployment, consider:

1. **Use Gunicorn**:
   ```bash
   pip install gunicorn
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Add Authentication**:
   ```python
   from fastapi import Depends, HTTPException, status
   from fastapi.security import HTTPBearer
   ```

3. **Rate Limiting**:
   ```python
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address
   ```

4. **Environment Variables**:
   ```bash
   export QDRANT_HOST=your-qdrant-host
   export QDRANT_PORT=6333
   ```

## 📚 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive documentation where you can:
- See all available endpoints
- Test the API directly
- View request/response schemas
- Understand parameter requirements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 