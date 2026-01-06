"""
Model Serving Documentation

This directory contains FastAPI application for serving the FrameModel for inference.

## Files

- `api.py`: Main FastAPI application with /infer endpoint

## Setup

### 1. Create .env file

Create `.env` in the `model-service/` directory:

```
MODEL_API_KEY=your-secret-api-key
```

### 2. Prepare Checkpoint

Ensure a checkpoint exists at `model-service/checkpoints/debug.pth`:

```bash
python src/train.py --data-dir data/sample --debug --output checkpoints/
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Server

From the root directory:

```bash
uvicorn model_service.src.serve.api:app --host 0.0.0.0 --port 8000
```

Or from the `model-service/` directory:

```bash
uvicorn src.serve.api:app --host 0.0.0.0 --port 8001 --reload
```

For production, use Gunicorn:

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker model_service.src.serve.api:app --bind 0.0.0.0:8000
```

## Access the API

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **Health Check**: http://localhost:8001/health
- **Config**: http://localhost:8001/config

## API Endpoints

### GET /

Root endpoint with API information.

**Response:**
```json
{
  "name": "Frame Classification API",
  "version": "1.0.0",
  "endpoints": {
    "infer": "POST /infer - Run inference on image file",
    "health": "GET /health - Health check",
    "docs": "GET /docs - Interactive API docs",
    "redoc": "GET /redoc - Alternative API docs"
  }
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "checkpoint_exists": true
}
```

### GET /config

Get server configuration (debug endpoint).

**Response:**
```json
{
  "device": "cuda",
  "cuda_available": true,
  "checkpoint_path": "...",
  "checkpoint_exists": true,
  "model_loaded": true,
  "max_file_size_mb": 10.0
}
```

### POST /infer

Run inference on an image file.

**Headers:**
- `X-API-KEY`: Required API key (from .env MODEL_API_KEY)

**Request:**
- Content-Type: `multipart/form-data`
- Body: File field named "file" (JPEG, PNG, etc.)

**Response (200 OK):**
```json
{
  "request_id": "abc12345",
  "fake_prob": 0.85,
  "real_prob": 0.15
}
```

**Error Responses:**
- `401`: Invalid or missing X-API-KEY header
- `400`: Invalid image format
- `413`: File too large (> 10 MB)
- `500`: Internal server error

## Example Usage

### Using cURL

```bash
# With API key
curl -X POST "http://localhost:8001/infer" \
  -H "X-API-KEY: your-secret-api-key" \
  -F "file=@path/to/image.jpg"

# Response
{
  "request_id": "abc12345",
  "fake_prob": 0.85,
  "real_prob": 0.15
}
```

### Using Python (requests)

```python
import requests

api_key = "your-secret-api-key"
files = {"file": open("image.jpg", "rb")}
headers = {"X-API-KEY": api_key}

response = requests.post(
    "http://localhost:8001/infer",
    files=files,
    headers=headers,
)

result = response.json()
print(f"Fake probability: {result['fake_prob']:.2%}")
```

### Using Python (httpx with async)

```python
import httpx

async def infer(image_path: str, api_key: str):
    with open(image_path, "rb") as f:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8001/infer",
                files={"file": f},
                headers={"X-API-KEY": api_key},
            )
            return response.json()
```

## API Key Management

### Development

Use a simple key in `.env`:

```
MODEL_API_KEY=dev-key-123
```

### Production

1. Generate a strong random key:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. Store securely in environment:
   ```bash
   export MODEL_API_KEY="your-strong-key-from-secrets-manager"
   ```

3. Never commit `.env` with real keys to git

## Troubleshooting

### Model not loading

Check that the checkpoint exists:

```bash
ls -la model-service/checkpoints/debug.pth
```

If missing, train the model:

```bash
python model-service/src/train.py --data-dir model-service/data/sample --debug --output model-service/checkpoints/
```

### API key issues

Verify the key is set correctly:

```python
# In Python
import os
from dotenv import load_dotenv
load_dotenv("model-service/.env")
print(os.getenv("MODEL_API_KEY"))
```

### CUDA out of memory

Run on CPU by setting environment variable:

```bash
export CUDA_VISIBLE_DEVICES=""
uvicorn model_service.src.serve.api:app --host 0.0.0.0 --port 8001
```

### Port already in use

Use a different port:

```bash
uvicorn model_service.src.serve.api:app --host 0.0.0.0 --port 8002
```

## Performance Considerations

- **Model caching**: Model is loaded once on startup and reused for all requests
- **Batch size**: Currently handles single image inference (batch_size=1)
- **GPU acceleration**: Automatic CUDA usage if available
- **Request logging**: Each request gets a unique ID for tracing

## Integration with Backend

The model serving API can be called from the FastAPI backend:

```python
# In backend/app/routes/inference.py
import httpx

async def run_inference(signed_url: str):
    # Download image from signed URL
    async with httpx.AsyncClient() as client:
        img_response = await client.get(signed_url)
        image_bytes = img_response.content
    
    # Send to model service
    files = {"file": image_bytes}
    headers = {"X-API-KEY": MODEL_API_KEY}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8001/infer",
            files=files,
            headers=headers,
        )
    
    return response.json()
```

## Next Steps

1. Train model with real data:
   ```bash
   python src/train.py --data-dir data/train --epochs 10 --output checkpoints/
   ```

2. Update checkpoint in api.py to use trained model:
   ```python
   CHECKPOINT_PATH = Path(__file__).parent.parent.parent / "checkpoints" / "epoch_010.pth"
   ```

3. Deploy to cloud (Google Cloud Run, AWS Lambda, etc.)

4. Add authentication middleware (OAuth, JWT tokens, etc.)

5. Add request rate limiting

6. Add monitoring and metrics collection
