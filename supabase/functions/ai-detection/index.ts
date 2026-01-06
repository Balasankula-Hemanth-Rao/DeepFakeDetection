import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.53.0'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

// Simulate AI detection processing
function simulateAIDetection(filename: string) {
  console.log(`Starting AI analysis for file: ${filename}`)
  
  // Simulate analysis based on filename for demo purposes
  const isSuspicious = filename.toLowerCase().includes('fake') || 
                      filename.toLowerCase().includes('deepfake') ||
                      Math.random() > 0.6 // Random factor for demo
  
  const baseConfidence = Math.random() * 0.3 + 0.7 // 0.7-1.0 range
  const prediction = isSuspicious ? 'FAKE' : 'REAL'
  const confidence = isSuspicious ? baseConfidence : (1 - baseConfidence)
  
  // Generate realistic analysis data
  const visualConfidence = Math.random() * 0.2 + (confidence - 0.1)
  const audioConfidence = Math.random() * 0.2 + (confidence - 0.1)
  
  const anomalyTimestamps = isSuspicious ? [
    { timestamp: 12.3, type: 'facial_inconsistency', severity: 0.8 },
    { timestamp: 28.7, type: 'lip_sync_mismatch', severity: 0.6 },
    { timestamp: 45.1, type: 'audio_artifact', severity: 0.7 }
  ] : []
  
  const visualAnalysis = {
    facial_regions_analyzed: 847,
    temporal_consistency_score: visualConfidence,
    suspicious_frames: isSuspicious ? [156, 342, 578, 723] : [],
    compression_artifacts: Math.random() * 0.3,
    edge_inconsistencies: isSuspicious ? Math.random() * 0.5 + 0.3 : Math.random() * 0.2
  }
  
  const audioAnalysis = {
    frequency_analysis_score: audioConfidence,
    spectral_anomalies: isSuspicious ? Math.random() * 0.6 + 0.2 : Math.random() * 0.2,
    voice_consistency: audioConfidence,
    background_noise_patterns: Math.random() * 0.3,
    synthetic_markers: isSuspicious ? Math.random() * 0.4 + 0.3 : Math.random() * 0.1
  }
  
  return {
    prediction,
    confidence_score: Math.max(0, Math.min(1, confidence)),
    visual_confidence: Math.max(0, Math.min(1, visualConfidence)),
    audio_confidence: Math.max(0, Math.min(1, audioConfidence)),
    analysis_duration_seconds: Math.random() * 30 + 15, // 15-45 seconds
    anomaly_timestamps,
    visual_analysis,
    audio_analysis
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

    console.log(`Processing detection job: ${jobId}`)

    // Get the job details
    const { data: job, error: jobError } = await supabaseClient
      .from('detection_jobs')
      .select('*')
      .eq('id', jobId)
      .single()

    if (jobError || !job) {
      throw new Error(`Job not found: ${jobError?.message}`)
    }

    // Update job status to processing
    await supabaseClient
      .from('detection_jobs')
      .update({
        status: 'processing',
        analysis_start_time: new Date().toISOString()
      })
      .eq('id', jobId)

    // Simulate AI processing delay
    await new Promise(resolve => setTimeout(resolve, 3000 + Math.random() * 2000))

    // Run AI detection simulation
    const results = simulateAIDetection(job.original_filename)

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

    // Update job status to completed
    await supabaseClient
      .from('detection_jobs')
      .update({
        status: 'completed',
        analysis_end_time: new Date().toISOString()
      })
      .eq('id', jobId)

    console.log(`AI detection completed for job: ${jobId}`)

    return new Response(
      JSON.stringify({ 
        success: true, 
        jobId,
        prediction: results.prediction,
        confidence: results.confidence_score
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )

  } catch (error: unknown) {
    console.error('Error in AI detection:', error)
    
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'

    return new Response(
      JSON.stringify({ error: errorMessage }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )
  }
})