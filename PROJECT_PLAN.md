# **AURA VERACITY LAB — COMPREHENSIVE PROJECT PLAN**

## **A. High-Level Overview**

**Purpose:** Aura Veracity Lab is an AI-powered deepfake detection system comprising a React web application, Python backend API, and machine learning model service. The platform enables users to upload videos, analyze them for signs of deepfake manipulation using multimodal (audio + video) deep learning, and receive detailed forensic reports with confidence scores and frame-level anomalies.

**Key Constraints:**
- **Tech Stack Lock:** React 18 + TypeScript (frontend), FastAPI (backend), PyTorch (ML service), Supabase (database + auth + storage)
- **Single Database:** Shared Supabase project across all services; no independent backend database
- **ML Latency:** Model inference is CPU/GPU-bound; async job processing is implicit, not implemented
- **Security:** JWT-based auth only; signed URLs for file access; no end-to-end encryption visible
- **Cost:** Supabase storage bandwidth and database operations are metered
- **Performance:** No caching, no CDN; all static assets served from built dist/

**Assumptions Made from Codebase:**
1. Frontend and backend are **separate deployment targets** (vite builds to `dist/`, backend is Python ASGI)
2. Model service is **stateless inference only**; no persistent model state between requests
3. Supabase project credentials are **shared** (frontend and backend point to same URL/keys)
4. Users are **authenticated via Supabase Auth**; JWT tokens are forwarded from frontend → backend → Supabase verification
5. Video uploads follow a **two-step flow:** get signed URL from backend, upload directly to Supabase, then notify backend
6. Detection results are **eventually consistent**; polling is the primary mechanism for job status
7. No RabbitMQ, no queuing system; jobs are stored in `detection_jobs` table with status field
8. Model checkpoints are **baked into the model-service Docker image** (at `/checkpoints/debug.pth`)

---

## **B. Architecture Breakdown**

### **High-Level Data Flow**

```
User (Browser)
   ↓ [Authenticate]
   ├→ Supabase Auth (email/password, Google OAuth, OTP)
   ├→ Frontend receives JWT + session
   │
   ├→ [Upload Video]
   │  ├→ POST /uploads/signed-url to Backend
   │  ├→ Backend verifies JWT, calls Supabase to generate signed URL
   │  ├→ Frontend receives signed URL, PUT file directly to Supabase Storage
   │  │
   │  └→ [Create Job]
   │     ├→ POST /uploads/init-job to Backend with video path
   │     ├→ Backend creates row in `detection_jobs` table (status: pending)
   │     └→ Returns job_id to frontend
   │
   └→ [Poll Results]
      ├→ Frontend periodically GET `/results/{jobId}` from Supabase or backend
      ├→ [Model Service processes video async - implicit, no endpoint]
      ├→ Model inference returns predictions
      ├→ Backend (or async worker) stores results in `detection_results` table
      └→ Frontend renders analysis with confidence scores, timestamps, video frames
```

### **Frontend Responsibilities**
- **Authentication:** Supabase auth UI (sign up, sign in, password reset, OTP, Google OAuth)
- **Pages:** Landing (Index), Auth, Dashboard (upload), Results (analysis view), History (past jobs), Compare (side-by-side)
- **Components:** Video uploader, progress bar, results visualization, analytics charts
- **State Management:** React Query for server state; local React state for UI state (theme, settings)
- **Styling:** Tailwind CSS + shadcn/ui component library (60+ pre-built components)
- **Theming:** Light/dark mode with ThemeContext
- **Error Handling:** Toast notifications, HTTP error handling, user feedback

### **Backend API Responsibilities**
- **Authentication:** Verify JWT tokens, extract user ID/email
- **Signed URL Generation:** Create expiring URLs for direct Supabase Storage uploads
- **Job Tracking:** CRUD operations on `detection_jobs` table
- **Result Querying:** Retrieve `detection_results` from database
- **Health Checks:** Kubernetes-ready liveness/readiness probes
- **CORS:** Cross-origin request handling for frontend

### **Model Service Responsibilities**
- **Inference:** Accept image/video frames, run through multimodal model, return probabilities
- **Data Preprocessing:** Extract frames, normalize, batch processing
- **Model Loading:** Load checkpoint from disk at startup
- **API:** FastAPI with `/infer` endpoint for frame-level predictions
- **Configuration:** YAML-based config for model paths, device selection, batch size
- **Security:** API key validation from headers

### **Database (Supabase) Responsibilities**
- **Authentication:** User credentials, JWT signing
- **Storage:** Video file storage with signed URLs, direct browser uploads
- **Tables:** `detection_jobs`, `detection_results` (schema assumed, not visible in code)
- **Real-time:** Subscriptions for live updates (optional, not used currently)
- **Polling:** Clients poll `/results/{jobId}` to check status

### **External Dependencies**
- **Supabase:** Auth, Postgres DB, Storage (S3-compatible)
- **Google OAuth:** OAuth provider for sign-in
- **PyTorch/TorchVision:** ML model training and inference
- **EfficientNet-B3:** Pre-trained backbone (via `timm` library)
- **Framer Motion:** React animations
- **React Router:** Client-side routing
- **Radix UI:** Unstyled component primitives (wrapped by shadcn/ui)

---

## **C. Component-Level Responsibilities**

### **Frontend (`src/`)**

| Component | Responsibility | Ownership | Should NOT Own |
|-----------|---|---|---|
| **App.tsx** | Route definitions, auth context, query client | Router setup | Auth state (delegated to useAuth) |
| **pages/Index.tsx** | Landing page (Hero, Features, HowItWorks, Pricing, About) | Marketing content | None (page) |
| **pages/Auth.tsx** | Sign up, sign in, OTP verification, Google OAuth flow | Auth UI | Token storage (Supabase SDK) |
| **pages/Dashboard.tsx** | Video upload, job creation, polling job status, showing progress | Core upload workflow | Model inference |
| **pages/Results.tsx** | Display detection results, confidence scores, frame analysis, export PDF | Result visualization | Video processing |
| **pages/History.tsx** | List past jobs, filtering, pagination, re-run analysis | Job history | Backend APIs |
| **pages/Compare.tsx** | Side-by-side analysis of two jobs | Comparison UI | Analysis computation |
| **hooks/useAuth.tsx** | Auth context provider, sign in/up/out, session management, OTP | Auth orchestration | JWT generation (Supabase) |
| **components/dashboard/VideoUploader.tsx** | File drag-drop, validation, chunked upload progress | Upload UX | Signed URL generation |
| **components/dashboard/AnalysisProgress.tsx** | Job status polling, ETA, animated progress | Polling loop | Backend APIs |
| **components/dashboard/AnalyticsCharts.tsx** | Confidence graphs, frame-level timelines, anomaly heatmaps | Chart rendering | Data aggregation |
| **contexts/ThemeContext.tsx** | Light/dark mode state, CSS variables | Theme switching | UI rendering |
| **integrations/supabase/client.ts** | Supabase client initialization, auth config | SDK bootstrap | Query logic |
| **lib/utils.ts** | Utility functions (classname merging, formatting) | Helper functions | Business logic |

### **Backend (`backend/app/`)**

| Module | Responsibility | Ownership | Should NOT Own |
|--------|---|---|---|
| **main.py** | FastAPI app factory, CORS, route registration | App initialization | Route logic |
| **config/settings.py** | Environment variable loading, validation | Configuration | Secrets (should use secrets manager) |
| **middleware/auth.py** | JWT verification, Authorization header parsing | Auth dependency | Token generation |
| **routes/health.py** | `/health`, `/health/ready`, `/health/live` probes | Service health | App startup logic |
| **routes/auth.py** | `GET /auth/me` endpoint | Auth verification | User database |
| **routes/uploads.py** | `POST /uploads/signed-url`, `POST /uploads/init-job` | Upload orchestration | Supabase client |
| **services/supabase_service.py** | JWT decode, signed URL generation, DB CRUD | Supabase operations | Route handling |

### **Model Service (`model-service/src/`)**

| Module | Responsibility | Ownership | Should NOT Own |
|--------|---|---|---|
| **serve/api.py** | FastAPI `/infer` endpoint, request/response handling | Inference API | Model training |
| **models/multimodal_model.py** | Model architecture, forward pass, feature extraction | Model definition | Data loading |
| **models/frame_model.py** | EfficientNet-B3 + classifier head | Frame classification | Preprocessing |
| **data/** | Dataset classes, batch loading | Data pipeline | Model selection |
| **preprocess/extract_frames.py** | Video → frame extraction, normalization | Frame preparation | Model inference |
| **train.py** | Training loop, checkpoint saving, evaluation | Model training | Serving |
| **eval/multimodal_eval.py** | Ablation studies, modality analysis | Evaluation experiments | Core model |
| **config.py** | YAML config loader, environment overrides | Configuration | Model logic |
| **logging_config.py** | Structured logging setup | Logging configuration | Application logic |

---

## **D. Execution Phases**

### **Phase 1: Core Functionality (MVP)** ✅ Currently Implemented

**Objective:** Enable authenticated users to upload videos and receive AI-powered deepfake detection results.

**Deliverables:**
1. **Frontend:**
   - Landing page (marketing material)
   - Auth pages (sign in, sign up, OTP, Google OAuth)
   - Dashboard (video uploader with progress)
   - Results page (detection results display)
   - History page (past job list)

2. **Backend:**
   - JWT token verification endpoint
   - Signed URL generation for uploads
   - Job creation in database
   - Health checks (Kubernetes-ready)

3. **Model Service:**
   - FastAPI inference endpoint
   - Multimodal model (video + audio)
   - Frame-level predictions
   - API key authentication

4. **Database:**
   - `detection_jobs` table (job metadata)
   - `detection_results` table (predictions, scores)
   - User authentication (Supabase Auth)
   - File storage (Supabase Storage)

**Assumptions & Gaps:**
- ⚠️ **No explicit async job processing:** Jobs are marked "pending" but no queue system (Celery, AWS SQS) drives the inference. Unclear how `detection_results` gets populated after upload.
- ⚠️ **Polling mechanism:** Frontend polls job status; no webhook notifications or WebSocket updates.
- ⚠️ **Single model instance:** Model service has no load balancing; can handle only one inference at a time.

---

### **Phase 2: Reliability & Scaling** 🚧 Partially Needed

**Objective:** Handle concurrent users, large files, long-running inference without timeouts.

**Deliverables:**
1. **Async Job Processing:**
   - Implement Celery or AWS SQS for job queue
   - Separate inference worker pods
   - Database updates triggered by workers

2. **Horizontal Scaling:**
   - Docker Compose → Kubernetes manifests
   - Multiple model service replicas
   - Load balancer for backend API

3. **Performance Optimization:**
   - Model quantization (FP16, INT8)
   - Batch inference (multiple frames at once)
   - Redis caching for token verification
   - CDN for static assets (frontend)

4. **Monitoring & Observability:**
   - Structured logging (JSON format)
   - Prometheus metrics (inference latency, job queue depth)
   - Sentry or similar for error tracking
   - Health check dashboards

5. **Database Optimization:**
   - Indexing on `detection_jobs.user_id`, `detection_jobs.status`
   - Partitioning `detection_results` by date
   - Connection pooling (pgbouncer)

---

### **Phase 3: Security, Monitoring, Optimization** 🚧 Partially Needed

**Objective:** Production-hardened system with audit trails, rate limiting, secrets management.

**Deliverables:**
1. **Security Hardening:**
   - Secrets manager (AWS Secrets Manager, HashiCorp Vault)
   - Rate limiting (IP-based, user-based)
   - Request validation (file size, MIME type, virus scanning)
   - HTTPS enforcement, HSTS headers
   - CSRF protection if needed
   - Role-based access control (RBAC) for admin endpoints

2. **Audit Logging:**
   - Log all file uploads, deletions
   - Track API endpoint usage per user
   - Store audit trail in separate table
   - GDPR compliance (data deletion hooks)

3. **Failure Handling:**
   - Retry logic for failed model inference
   - Dead letter queue for unprocessable jobs
   - Circuit breaker pattern for Supabase calls
   - Graceful degradation (fallback responses)

4. **Compliance:**
   - Data residency (GDPR, CCPA)
   - Video retention policies
   - User consent management
   - Regular security audits

---

### **Phase 4: Optional Enhancements** 🎯 Nice-to-Have

**Objective:** Competitive features and user retention.

**Deliverables:**
1. **Comparative Analysis:**
   - Side-by-side video comparison (Compare page exists but may be stub)
   - Temporal anomaly heatmaps
   - Modality contribution charts (video vs. audio confidence)

2. **Advanced Exports:**
   - PDF report generation
   - JSON exports for forensic tools
   - Video overlay with confidence timeline
   - Frame-by-frame breakdowns

3. **User Experience:**
   - Keyboard shortcuts (provider exists, implementation pending)
   - Dark mode (context exists, CSS not finalized)
   - Settings panel (exists, purpose unclear)
   - Batch upload/analysis

4. **ML Enhancements:**
   - Fine-tuning on user-supplied datasets
   - Model versioning and A/B testing
   - Uncertainty quantification
   - Adversarial robustness analysis

---

## **E. Risk & Gap Analysis**

### **Critical Gaps**

| Gap | Impact | Severity | Mitigation |
|-----|--------|----------|-----------|
| **No explicit job queue** | Unclear how model inference is triggered after job creation; jobs may never transition from "pending" | 🔴 CRITICAL | Implement Celery or AWS SQS; define worker process |
| **Synchronous model inference** | Model service `/infer` endpoint is synchronous; blocks requests if inference is slow | 🔴 CRITICAL | Implement async inference (async FastAPI, background tasks) |
| **No retry logic** | Failed uploads or model crashes are not retried | 🔴 CRITICAL | Add exponential backoff, circuit breaker, DLQ |
| **Polling-only job status** | Frontend polls every N seconds; inefficient, poor UX on slow connections | 🟠 HIGH | Implement WebSocket or Server-Sent Events (SSE) for push updates |
| **Single model instance** | Model service can only process one frame/video at a time | 🟠 HIGH | Implement batch inference, multi-GPU support |

### **Design Flaws**

| Flaw | Consequence | Mitigation |
|------|---|---|
| **JWT verification done locally** | Backend decodes JWT manually instead of trusting Supabase; no signature validation | Use Supabase's native JWT verification or verify signature against public key |
| **Secrets in environment variables** | `.env` file checked into git (though .gitignore should prevent it); plain-text secrets in CI/CD logs | Use secrets manager (AWS Secrets Manager, Vault); rotate keys regularly |
| **No input validation on file uploads** | Potential for large files, malicious payloads, MIME type spoofing | Validate file size, MIME type, scan for malware; use Supabase Storage's built-in restrictions |
| **CORS allows all origins** | `allow_origins=["*"]` in backend; any website can call your API | Restrict to known frontend domains; use environment-specific CORS config |

### **Performance Risks**

| Risk | Likelihood | Impact |
|------|------------|--------|
| **Model loading on every request** | Low (checkpoint loaded at startup) | High (first request slow) |
| **Large video files crash model service** | High (no file size limit) | High (OOM, service restart) |
| **Database connection pool exhaustion** | Medium (no pgbouncer) | Medium (connection timeouts under load) |
| **Frontend polling hammers backend** | Medium (no request throttling) | Medium (DDoS-like behavior, degraded UX) |
| **Supabase storage bandwidth limits** | Medium (no upload chunking) | Medium (slow uploads, timeouts) |

### **Security Risks**

| Risk | Severity | Impact |
|------|----------|--------|
| **JWT token leakage in logs** | 🔴 HIGH | Authenticated access to user accounts |
| **Supabase keys in frontend code** | 🔴 HIGH | Direct access to storage, database (mitigated by row-level security) |
| **No rate limiting** | 🟠 MEDIUM | Brute force attacks, resource exhaustion |
| **No file type validation** | 🟠 MEDIUM | Malware distribution, code injection |
| **Missing HTTPS/TLS** | 🟠 MEDIUM | Man-in-the-middle attacks during development |

### **Operational Gaps**

| Gap | Mitigation |
|-----|-----------|
| **No logs aggregation** | Use ELK stack, Datadog, or cloud provider logs (CloudWatch, Stackdriver) |
| **No metrics** | Add Prometheus, StatsD, or cloud monitoring |
| **No alerting** | Set up PagerDuty or similar for critical failures |
| **No backup strategy** | Supabase backups are automated; verify retention policy |
| **No disaster recovery plan** | Document RTO/RPO; test failover scenarios |

---

## **F. Developer Onboarding Guide**

### **First 30 Minutes: Project Understanding**

**Goal:** Understand what this project does and its overall structure.

1. **Read this plan** (10 min)
2. **Explore folder structure** (5 min)
   ```
   aura-veracity-lab/
   ├── src/                    ← React frontend
   ├── backend/                ← FastAPI backend
   ├── model-service/          ← PyTorch ML service
   ├── supabase/               ← DB schema (if managed here)
   ├── package.json            ← Frontend deps
   └── docker-compose.yml      ← Local dev setup
   ```
3. **Run the project locally** (15 min)
   ```bash
   # Terminal 1: Frontend
   npm install && npm run dev
   # Terminal 2: Backend
   cd backend && pip install -r requirements.txt && uvicorn main:app --reload
   # Terminal 3: Model service
   cd model-service && pip install -r requirements.txt && uvicorn src.serve.api:app --reload --port 8001
   ```

**Mental Model:** User uploads video → Backend creates job → Model service processes → Results stored → Frontend shows results

---

### **Next 60 Minutes: Component Deep Dive**

**Goal:** Understand how each piece fits together.

**Order to Read:**
1. **`frontend/App.tsx`** (2 min) — Routes and context providers
2. **`frontend/pages/Auth.tsx`** (5 min) — Auth flow, Supabase integration
3. **`frontend/pages/Dashboard.tsx`** (10 min) — Video upload, job polling
4. **`backend/app/main.py`** (3 min) — FastAPI setup, middleware
5. **`backend/app/services/supabase_service.py`** (10 min) — Database/storage operations
6. **`backend/app/routes/uploads.py`** (5 min) — Signed URL and job creation
7. **`model-service/src/serve/api.py`** (10 min) — Inference endpoint
8. **`model-service/src/models/multimodal_model.py`** (15 min) — Model architecture

**Questions to Answer:**
- How does a JWT token flow from frontend → backend?
- How does a video get uploaded to Supabase?
- How does the backend know when the model finishes processing?
- What happens if a job fails?

---

### **Next 120 Minutes: Execution & Testing**

**Goal:** Make a small change and see it work end-to-end.

**Exercise 1: Add a new health check**
1. Add endpoint in `backend/app/routes/health.py` that returns model service status
2. Call model service `/health` from backend health check
3. Test via frontend or curl
4. **Deliverable:** New `GET /health/model-service` returns `{"status": "ok", "model": "multimodal_v1"}`

**Exercise 2: Add caching to JWT verification**
1. Add Redis cache check in `backend/app/middleware/auth.py`
2. Cache verified tokens for 5 minutes
3. Test login → dashboard → repeat calls
4. **Deliverable:** First call hits Supabase, subsequent calls use cache

**Exercise 3: Improve error handling**
1. Catch model inference timeout in `model-service/src/serve/api.py`
2. Return sensible error instead of 500
3. Log to structured logger
4. **Deliverable:** `POST /infer` with large file returns 422 with helpful message

---

### **Key Files to Understand (In Priority Order)**

```
UNDERSTAND FIRST (Day 1):
├── src/App.tsx                                (Routes)
├── src/pages/Dashboard.tsx                    (Upload workflow)
├── backend/app/main.py                        (API setup)
├── backend/app/services/supabase_service.py   (Database)
└── model-service/src/serve/api.py             (Inference)

UNDERSTAND NEXT (Day 2):
├── src/hooks/useAuth.tsx                      (Auth context)
├── src/pages/Results.tsx                      (Results display)
├── backend/app/routes/uploads.py              (API endpoints)
├── backend/app/config/settings.py             (Configuration)
└── model-service/src/models/multimodal_model.py (Model)

UNDERSTAND EVENTUALLY (Week 2+):
├── src/pages/History.tsx                      (Job history)
├── src/pages/Compare.tsx                      (Comparison)
├── model-service/src/train.py                 (Training)
├── model-service/src/eval/                    (Evaluation)
└── Dockerfile, docker-compose.yml             (Deployment)
```

---

### **Common Tasks & Where to Make Changes**

| Task | Where | How |
|------|-------|-----|
| Add new API endpoint | `backend/app/routes/*.py` | Create router, add dependency |
| Change authentication flow | `src/hooks/useAuth.tsx` | Modify sign in/up methods |
| Add new page | `src/pages/*.tsx` | Create component, add to App.tsx routes |
| Modify model architecture | `model-service/src/models/multimodal_model.py` | Edit forward() method |
| Add database table | Supabase dashboard or schema file | Update types in `src/integrations/supabase/types.ts` |
| Change styling | `src/*.css`, Tailwind config | Use Tailwind classes or modify postcss.config.js |
| Add environment variable | `.env.example`, `backend/app/config/settings.py` | Add to Settings class, document |

---

### **Debugging Checklist**

- **Frontend doesn't connect to backend?**
  - Check CORS in `backend/app/main.py` (allow_origins)
  - Verify backend is running on correct port (8000)
  - Check browser console for CORS errors

- **Upload fails after getting signed URL?**
  - Check Supabase Storage bucket name in `.env`
  - Verify service role key has storage permissions
  - Check file size limits in Supabase dashboard

- **Model inference times out?**
  - Check model checkpoint path in `.env`
  - Verify GPU/CUDA is available if expected
  - Check model-service logs for memory errors

- **Job stays in "pending" forever?**
  - Check if model-service is running
  - Look for errors in model-service logs
  - Verify database write permissions
  - Check if job status is being updated (query `detection_jobs` table)

---

## **G. Open Questions & Clarifications Needed**

1. **Job Processing Pipeline:** How does `detection_jobs` transition from "pending" to "completed"? Is there a background worker, a cron job, or does the model service push results back?
2. **Model Service Integration:** Does the backend call the model service directly, or is there an async queue? Is there a deployment assumption (same pod, separate service, external API)?
3. **Database Schema:** What fields exist in `detection_jobs` and `detection_results` tables? This is not visible in code.
4. **Supabase Row-Level Security (RLS):** Are RLS policies enabled? Can users see other users' jobs?
5. **File Size Limits:** Are there enforced limits on video file size? Supabase Storage has a 5GB per file limit.
6. **Audio Processing:** The model is "multimodal" but how is audio extracted from video files? Who does this (backend, model service, frontend)?
7. **Inference Latency:** What's the expected inference time per video? Is it seconds, minutes, or hours?
8. **Concurrent Users:** What's the target scale (10 users, 1000 users, 1M users)?
9. **Model Update Cadence:** How often is the model retrained? Who deploys new checkpoints?
10. **Compliance & Data Retention:** How long are videos and results stored? What's the GDPR/CCPA deletion mechanism?

---

**END OF PLAN**

This plan provides a complete mental model of the system, clear ownership boundaries, identified gaps, and a structured path for new developers to ramp up. Each phase is actionable and risk-aware.
