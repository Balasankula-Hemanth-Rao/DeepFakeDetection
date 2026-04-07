import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.53.0'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

/**
 * Call the real model analysis service for cross-dataset validation
 * 
 * This connects to the Python model service (model-service/src/serve/service.py)
 * which runs multimodal deepfake detection with real inference or simulation fallback.
 * 
 * The service is designed to support:
 * - Real model inference (if checkpoints available)
 * - Graceful fallback to simulation (for demo/testing)
 * - Cross-dataset validation (FaceForensics++ → FakeAVCeleb)
 */
async function callModelService(
  jobId: string,
  filePath: string,
  supabaseClient: any
): Promise<any> {
  // Get the model service URL from environment or use default
  const modelServiceUrl = Deno.env.get('MODEL_SERVICE_URL') || 'http://localhost:8001'
  
  console.log(`[${jobId}] Calling model service at ${modelServiceUrl}`)
  console.log(`[${jobId}] Analyzing: ${filePath}`)
  
  try {
    // Submit analysis request to model service
    const response = await fetch(`${modelServiceUrl}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        job_id: jobId,
        video_path: filePath,
      }),
    })
    
    if (!response.ok) {
      const error = await response.text()
      throw new Error(`Model service error (${response.status}): ${error}`)
    }
    
    const analysisResult = await response.json()
    console.log(`[${jobId}] ✓ Analysis received from model service`)
    
    return analysisResult
    
  } catch (error) {
    console.error(`[${jobId}] Model service call failed: ${error.message}`)
    console.log(`[${jobId}] Falling back to mock results for demo`)
    
    // Fallback: generate mock results for demo
    // This ensures the pipeline works even if model service is not available
    return generateMockResults(filePath)
  }
}

/**
 * Generate mock results for demonstration
 * Used when model service is unavailable
 */
function generateMockResults(filePath: string): any {
  const filename = filePath.split('/').pop()?.toLowerCase() || ''
  const isSuspicious = filename.includes('fake') || filename.includes('deepfake') || Math.random() > 0.6
  
  const baseConfidence = Math.random() * 0.3 + 0.7
  const prediction = isSuspicious ? 'FAKE' : 'REAL'
  const confidence = isSuspicious ? baseConfidence : (1 - baseConfidence)
  const visualConfidence = Math.random() * 0.2 + (confidence - 0.1)
  const audioConfidence = Math.random() * 0.2 + (confidence - 0.1)
  
  return {
    prediction,
    confidence_score: Math.max(0, Math.min(1, confidence)),
    visual_confidence: Math.max(0, Math.min(1, visualConfidence)),
    audio_confidence: Math.max(0, Math.min(1, audioConfidence)),
    analysis_duration_seconds: Math.random() * 30 + 15,
    anomaly_timestamps: isSuspicious ? [
      { timestamp: 12.3, type: 'facial_inconsistency', severity: 0.8 },
      { timestamp: 28.7, type: 'lip_sync_mismatch', severity: 0.6 },
    ] : [],
    visual_analysis: {
      facial_regions_analyzed: 847,
      temporal_consistency_score: visualConfidence,
      suspicious_frames: isSuspicious ? [156, 342, 578, 723] : [],
      compression_artifacts: Math.random() * 0.3,
      edge_inconsistencies: isSuspicious ? Math.random() * 0.5 + 0.3 : Math.random() * 0.2,
    },
    audio_analysis: {
      frequency_analysis_score: audioConfidence,
      spectral_anomalies: isSuspicious ? Math.random() * 0.6 + 0.2 : Math.random() * 0.2,
      voice_consistency: audioConfidence,
      background_noise_patterns: Math.random() * 0.3,
      synthetic_markers: isSuspicious ? Math.random() * 0.4 + 0.3 : Math.random() * 0.1,
    },
    is_simulation: true, // Flag indicating this is simulated
  }
}

Deno.serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders })
  }

  try {
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    const { jobId } = await req.json()
    
    if (!jobId) {
      throw new Error('Job ID is required')
    }

    console.log(`[${jobId}] ===== AI Detection Pipeline Started =====`)
    console.log(`[${jobId}] Cross-dataset validation mode enabled`)

    // Get the job details
    const { data: job, error: jobError } = await supabaseClient
      .from('detection_jobs')
      .select('*')
      .eq('id', jobId)
      .single()

    if (jobError || !job) {
      throw new Error(`Job not found: ${jobError?.message}`)
    }

    console.log(`[${jobId}] Job loaded: ${job.original_filename} at ${job.file_path}`)

    // Update job status to processing
    await supabaseClient
      .from('detection_jobs')
      .update({
        status: 'processing',
        analysis_start_time: new Date().toISOString()
      })
      .eq('id', jobId)

    console.log(`[${jobId}] Status: processing`)

    // Call the real model service (with fallback to mock)
    const results = await callModelService(jobId, job.file_path, supabaseClient)

    console.log(`[${jobId}] Analysis result: ${results.prediction} (confidence: ${results.confidence_score.toFixed(3)})`)
    console.log(`[${jobId}] Using ${results.is_simulation ? 'SIMULATION' : 'REAL MODEL'}`)

    // Save results to database
    const { error: resultError } = await supabaseClient
      .from('detection_results')
      .insert({
        job_id: jobId,
        ...results
      })

    if (resultError) {
      throw new Error(`Failed to save results: ${resultError.message}`)
    }

    console.log(`[${jobId}] Results saved to detection_results`)

    // Update job status to completed
    await supabaseClient
      .from('detection_jobs')
      .update({
        status: 'completed',
        analysis_end_time: new Date().toISOString()
      })
      .eq('id', jobId)

    console.log(`[${jobId}] Status: completed`)
    console.log(`[${jobId}] ===== AI Detection Pipeline Finished (${results.analysis_duration_seconds.toFixed(1)}s) =====`)

    return new Response(
      JSON.stringify({ 
        success: true, 
        jobId,
        prediction: results.prediction,
        confidence: results.confidence_score,
        is_simulation: results.is_simulation,
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )

  } catch (error: unknown) {
    console.error('Error in AI detection:', error)
    
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'

    return new Response(
      JSON.stringify({ 
        error: errorMessage,
        message: 'Please ensure MODEL_SERVICE_URL env var points to running model service'
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )
  }
})