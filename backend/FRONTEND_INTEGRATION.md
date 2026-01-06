# Frontend-to-Backend Integration Guide

This document shows how the frontend can integrate with the new backend API.

## Overview

The backend provides three main endpoints:

1. **GET /auth/me** — Verify authenticated user
2. **POST /uploads/signed-url** — Get a signed URL for file uploads
3. **POST /uploads/init-job** — Create a detection job after upload

---

## 1. Authenticate & Get Current User

Use this to verify the user is logged in and fetch their info from the backend.

### Request

```typescript
const response = await fetch('http://localhost:8000/auth/me', {
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${session.access_token}`,
    'Content-Type': 'application/json',
  },
});

const user = await response.json();
// Response: { "id": "user-uuid", "email": "user@example.com", "authenticated": true }
```

### Error Handling

```typescript
if (response.status === 401) {
  // Token invalid or expired
  // Redirect to login
  navigate('/auth');
}
```

---

## 2. Upload Video (Two-Step Flow)

Instead of uploading directly to Supabase from the frontend, use the backend:

### Step 1: Get Signed URL

```typescript
const getSignedUrl = async (filename: string) => {
  const response = await fetch('http://localhost:8000/uploads/signed-url', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${session.access_token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      filename: filename,
      expires_in: 3600, // 1 hour
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to get signed URL: ${response.statusText}`);
  }

  const data = await response.json();
  return data.signed_url; // Use this URL for upload
};
```

### Step 2: Upload File

```typescript
const uploadFile = async (file: File, signedUrl: string) => {
  const response = await fetch(signedUrl, {
    method: 'PUT',
    body: file,
    headers: {
      'Content-Type': file.type,
    },
  });

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.statusText}`);
  }

  console.log('✓ File uploaded successfully');
};
```

### Step 3: Create Detection Job

After upload, create a detection job record:

```typescript
const createDetectionJob = async (filename: string, filePath: string) => {
  const response = await fetch('http://localhost:8000/uploads/init-job', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${session.access_token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      original_filename: filename,
      file_path: filePath, // e.g., "user-id/1234567890/video.mp4"
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to create job: ${response.statusText}`);
  }

  const job = await response.json();
  return job.job_id; // Use this for polling results
};
```

---

## Complete Upload Flow Example

Here's a complete example for integrating into the `Dashboard` component:

```typescript
import { useState } from 'react';
import { useAuth } from '@/hooks/useAuth';

const Dashboard = () => {
  const { user, session } = useAuth();
  const [uploading, setUploading] = useState(false);
  const [currentJob, setCurrentJob] = useState<string | null>(null);

  const handleFileUpload = async (file: File) => {
    try {
      setUploading(true);

      // Step 1: Get signed URL
      const signedUrlResponse = await fetch(
        `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/uploads/signed-url`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${session?.access_token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            filename: file.name,
            expires_in: 3600,
          }),
        }
      );

      if (!signedUrlResponse.ok) {
        throw new Error('Failed to get signed URL');
      }

      const { signed_url } = await signedUrlResponse.json();

      // Step 2: Upload file directly to signed URL
      const uploadResponse = await fetch(signed_url, {
        method: 'PUT',
        body: file,
        headers: {
          'Content-Type': file.type,
        },
      });

      if (!uploadResponse.ok) {
        throw new Error('Failed to upload file');
      }

      // Step 3: Create detection job
      const timestamp = Date.now();
      const jobResponse = await fetch(
        `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/uploads/init-job`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${session?.access_token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            original_filename: file.name,
            file_path: `${user?.id}/${timestamp}/${file.name}`,
          }),
        }
      );

      if (!jobResponse.ok) {
        throw new Error('Failed to create detection job');
      }

      const { job_id } = await jobResponse.json();
      setCurrentJob(job_id);

      // Now poll or subscribe to job status...
      pollJobStatus(job_id);
    } catch (error) {
      console.error('Upload failed:', error);
      // Show error toast
    } finally {
      setUploading(false);
    }
  };

  // Rest of component...
  return (
    // Your existing JSX
  );
};
```

---

## Environment Variables

Add to your frontend `.env`:

```bash
# Backend API URL (adjust based on environment)
VITE_API_URL=http://localhost:8000
# Or in production:
# VITE_API_URL=https://api.aura-veracity.com
```

---

## Advantages of This Approach

✅ **Secure**: Signed URLs are time-limited and user-specific  
✅ **Efficient**: Files bypass the backend (cheaper bandwidth)  
✅ **Scalable**: Backend doesn't process large files  
✅ **Flexible**: Easy to add pre-upload validation, rate limiting  
✅ **Logs**: All uploads tracked in Supabase database  

---

## Backward Compatibility

The frontend can still use the direct Supabase client for:
- User authentication
- Reading results
- Realtime subscriptions

The backend complements the existing code without breaking anything.

---

## Troubleshooting

### "Invalid token"
- Ensure `session.access_token` is valid and not expired
- Check the Authorization header format: `Bearer <token>`

### "Failed to get signed URL"
- Verify the backend is running on `http://localhost:8000`
- Check that `SUPABASE_SERVICE_ROLE_KEY` is set in backend `.env`

### CORS errors
- If running frontend on a different port, update backend `ALLOW_ORIGINS` in `.env`
- Default is `*` which allows all origins in dev mode

### File upload fails
- Check file size (max 50MB)
- Ensure the signed URL hasn't expired (default 1 hour)
- Verify the storage bucket exists (should be `videos`)

---

## What's Next

1. Update `src/pages/Dashboard.tsx` to use backend endpoints
2. Move signed URL generation from frontend to backend
3. Add real-time job status via WebSocket (future enhancement)
4. Add admin panel for viewing all jobs (optional)

---

For detailed API docs, visit: `http://localhost:8000/docs` when the backend is running.
