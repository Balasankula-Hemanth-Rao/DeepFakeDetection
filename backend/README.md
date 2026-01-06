# Aura Veracity Backend

A fast, lightweight FastAPI backend for the Aura Veracity deepfake detection system.

## Overview

This backend provides REST APIs for the Aura Veracity webapp, integrating with the same Supabase project used by the frontend. It handles:

- **Authentication**: JWT token verification using Supabase auth
- **File uploads**: Generates signed URLs for direct uploads to Supabase Storage
- **Job management**: Creates and tracks video analysis jobs
- **Health checks**: Readiness and liveness probes for orchestration

## Features

✅ FastAPI (fast, async-ready, good for ML integration)  
✅ Supabase integration (same project as frontend)  
✅ JWT token verification  
✅ Signed URL generation for secure uploads  
✅ CORS-enabled for cross-origin requests  
✅ Production-ready Dockerfile  
✅ Kubernetes health checks  
✅ Structured logging  

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app factory
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py         # Configuration from env vars
│   ├── middleware/
│   │   ├── __init__.py
│   │   └── auth.py             # JWT verification dependencies
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py           # GET /health, /health/ready, /health/live
│   │   ├── auth.py             # GET /auth/me
│   │   └── uploads.py          # POST /uploads/signed-url, /uploads/init-job
│   └── services/
│       ├── __init__.py
│       └── supabase_service.py # Supabase client wrapper
├── main.py                     # ASGI entry point
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── Dockerfile                  # Multi-stage Docker build
└── README.md                   # This file
```

## Setup

### Prerequisites

- Python 3.10+
- pip or poetry
- Access to the Aura Veracity Supabase project

### Local Development

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/B-Hemanth-Rao/aura-veracity-lab.git
   cd aura-veracity-lab/backend
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create `.env` file** from the example:
   ```bash
   cp .env.example .env
   ```

5. **Edit `.env` with your Supabase credentials**:
   - Get `SUPABASE_SERVICE_ROLE_KEY` from Supabase project settings → API
   - Verify `SUPABASE_URL` and `SUPABASE_ANON_KEY` match the frontend

6. **Run the development server**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

   The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Run Tests

```bash
pytest
```

## API Endpoints

### Health Check

```http
GET /health
```

Returns status and service info.

**Response:**
```json
{
  "status": "healthy",
  "service": "aura-veracity-backend",
  "version": "1.0.0",
  "environment": "development"
}
```

---

### Get Current User

```http
GET /auth/me
Authorization: Bearer <JWT_TOKEN>
```

Returns authenticated user info.

**Request Headers:**
- `Authorization`: Bearer token from Supabase auth

**Response (200 OK):**
```json
{
  "id": "user-uuid-here",
  "email": "user@example.com",
  "authenticated": true
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid or missing token

---

### Generate Signed Upload URL

```http
POST /uploads/signed-url
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json
```

Generates a signed URL for uploading a file directly to Supabase Storage.

**Request Body:**
```json
{
  "filename": "video.mp4",
  "expires_in": 3600
}
```

**Response (200 OK):**
```json
{
  "signed_url": "https://...",
  "bucket": "videos",
  "expires_in": 3600
}
```

**Error Responses:**
- `400 Bad Request`: Missing filename
- `401 Unauthorized`: Invalid token
- `500 Internal Server Error`: Failed to generate URL

---

### Initialize Detection Job

```http
POST /uploads/init-job
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json
```

Creates a detection job record after uploading a file. Call this after the file is uploaded to Supabase Storage.

**Request Body:**
```json
{
  "original_filename": "video.mp4",
  "file_path": "user-id/timestamp/video.mp4"
}
```

**Response (200 OK):**
```json
{
  "job_id": "job-uuid-here",
  "status": "pending",
  "original_filename": "video.mp4",
  "upload_timestamp": "2024-01-01T12:00:00Z"
}
```

**Error Responses:**
- `400 Bad Request`: Missing required fields
- `401 Unauthorized`: Invalid token
- `500 Internal Server Error`: Failed to create job

---

## Environment Variables

See `.env.example` for all available configuration. Key variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `SUPABASE_URL` | Supabase project URL | `https://xxx.supabase.co` |
| `SUPABASE_ANON_KEY` | Supabase public key | (long JWT string) |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase admin key | (long JWT string) |
| `SUPABASE_STORAGE_BUCKET` | Storage bucket name | `videos` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `DEBUG` | Debug mode | `true` or `false` |
| `ALLOW_ORIGINS` | CORS origins | `http://localhost:5173,*` |

## Docker

### Build

```bash
docker build -t aura-veracity-backend .
```

### Run Locally

```bash
docker run -p 8000:8000 \
  -e SUPABASE_URL=https://xxx.supabase.co \
  -e SUPABASE_ANON_KEY=... \
  -e SUPABASE_SERVICE_ROLE_KEY=... \
  aura-veracity-backend
```

Or with an `.env` file:

```bash
docker run -p 8000:8000 --env-file .env aura-veracity-backend
```

### Push to Registry

```bash
docker tag aura-veracity-backend your-registry/aura-veracity-backend
docker push your-registry/aura-veracity-backend
```

## Deployment

### Google Cloud Run

```bash
gcloud run deploy aura-veracity-backend \
  --image gcr.io/your-project/aura-veracity-backend \
  --platform managed \
  --memory 1Gi \
  --port 8000 \
  --allow-unauthenticated \
  --set-env-vars "SUPABASE_URL=...,SUPABASE_ANON_KEY=...,SUPABASE_SERVICE_ROLE_KEY=..."
```

### AWS Lambda (with API Gateway)

Use a serverless framework like Mangum or Zappa to wrap the FastAPI app.

### Kubernetes

```bash
kubectl apply -f deployment.yaml
```

Example deployment manifest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aura-veracity-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aura-veracity-backend
  template:
    metadata:
      labels:
        app: aura-veracity-backend
    spec:
      containers:
      - name: backend
        image: your-registry/aura-veracity-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: SUPABASE_URL
          valueFrom:
            secretKeyRef:
              name: supabase-secrets
              key: url
        - name: SUPABASE_SERVICE_ROLE_KEY
          valueFrom:
            secretKeyRef:
              name: supabase-secrets
              key: service-role-key
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Integration with Frontend

The frontend (React + Vite) already has the Supabase client configured. To use the backend:

1. **Get JWT token** from Supabase auth (already available in frontend)
2. **Call backend endpoints** with `Authorization: Bearer <token>` header
3. **Use signed URLs** for file uploads instead of frontend storage client

Example frontend integration:

```typescript
// Get signed URL from backend
const response = await fetch('http://localhost:8000/uploads/signed-url', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${session.access_token}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    filename: file.name,
    expires_in: 3600,
  }),
});

const { signed_url } = await response.json();

// Upload file directly to signed URL
await fetch(signed_url, {
  method: 'PUT',
  body: file,
  headers: { 'Content-Type': file.type },
});
```

## Security Considerations

⚠️ **Never commit `.env` or the service role key to git!**

- Use GitHub Secrets or CI/CD platform secrets to store `SUPABASE_SERVICE_ROLE_KEY`
- Rotate keys regularly in the Supabase dashboard
- Use signed URLs with expiration times (default 1 hour)
- Verify JWT tokens on every request
- Use HTTPS in production
- Set `DEBUG=false` in production

## Troubleshooting

### "Invalid or expired token"
- Ensure the token is valid and not expired
- Check that the Authorization header format is `Bearer <token>`
- Verify `SUPABASE_ANON_KEY` in `.env` matches the frontend

### "Failed to generate signed URL"
- Ensure `SUPABASE_SERVICE_ROLE_KEY` is set in `.env`
- Check that the service role key is correct in Supabase settings
- Verify the storage bucket exists and is named `videos`

### "CORS errors"
- Check `ALLOW_ORIGINS` in `.env` includes your frontend URL
- Use `ALLOW_ORIGINS=*` for development (not recommended for production)

### Logs
- Set `DEBUG=true` in `.env` to see detailed logs
- Check Docker logs with `docker logs <container_id>`

## Future Enhancements

- [ ] WebSocket support for real-time job updates
- [ ] Batch upload API
- [ ] Job retry logic with exponential backoff
- [ ] Metrics and monitoring (Prometheus, OpenTelemetry)
- [ ] Rate limiting and throttling
- [ ] Admin endpoints for job management
- [ ] Webhook support for job completion notifications
- [ ] Integration with ML inference services

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request

## License

MIT

## Support

For issues or questions, please open a GitHub issue or contact the team.

---

**Built with ❤️ for AI-powered deepfake detection**
