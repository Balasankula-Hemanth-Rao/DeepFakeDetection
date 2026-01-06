import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

Deno.serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders })
  }

  try {
    // Get the authorization header
    const authHeader = req.headers.get('Authorization')
    if (!authHeader) {
      return new Response(
        JSON.stringify({ error: 'Missing authorization header' }),
        { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // Create Supabase client with user's token for initial verification
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!
    const supabaseAnonKey = Deno.env.get('SUPABASE_ANON_KEY')!
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!

    const supabaseUser = createClient(supabaseUrl, supabaseAnonKey, {
      global: { headers: { Authorization: authHeader } }
    })

    // Get the authenticated user
    const { data: { user }, error: userError } = await supabaseUser.auth.getUser()
    
    if (userError || !user) {
      console.error('Auth error:', userError)
      return new Response(
        JSON.stringify({ error: 'Unauthorized' }),
        { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const userId = user.id
    console.log(`Starting account deletion for user: ${userId}`)

    // Create admin client for deletion operations
    const supabaseAdmin = createClient(supabaseUrl, supabaseServiceKey)

    try {
      // Step 1: Delete all files from storage (videos bucket)
      console.log('Deleting user files from storage...')
      try {
        const { data: files, error: listError } = await supabaseAdmin.storage
          .from('videos')
          .list(userId)

        if (listError) {
          console.error('Error listing files:', listError)
        } else if (files && files.length > 0) {
          const filePaths = files.map(file => `${userId}/${file.name}`)
          const { error: deleteFilesError } = await supabaseAdmin.storage
            .from('videos')
            .remove(filePaths)
          
          if (deleteFilesError) {
            console.error('Error deleting files:', deleteFilesError)
          } else {
            console.log(`Deleted ${filePaths.length} files from storage`)
          }
        }
      } catch (storageError) {
        console.error('Storage deletion failed, continuing:', storageError)
      }

      // Step 2: Delete detection results (linked to jobs)
      console.log('Deleting detection results...')
      try {
        const { data: jobs } = await supabaseAdmin
          .from('detection_jobs')
          .select('id')
          .eq('user_id', userId)

        if (jobs && jobs.length > 0) {
          const jobIds = jobs.map(job => job.id)
          const { error: resultsError } = await supabaseAdmin
            .from('detection_results')
            .delete()
            .in('job_id', jobIds)

          if (resultsError) {
            console.error('Error deleting detection results:', resultsError)
          } else {
            console.log(`Deleted detection results for ${jobIds.length} jobs`)
          }
        }
      } catch (resultsError) {
        console.error('Results deletion failed, continuing:', resultsError)
      }

      // Step 3: Delete detection jobs
      console.log('Deleting detection jobs...')
      try {
        const { error: jobsError } = await supabaseAdmin
          .from('detection_jobs')
          .delete()
          .eq('user_id', userId)

        if (jobsError) {
          console.error('Error deleting detection jobs:', jobsError)
        } else {
          console.log('Deleted detection jobs')
        }
      } catch (jobsError) {
        console.error('Jobs deletion failed, continuing:', jobsError)
      }

      // Step 4: Delete user profile
      console.log('Deleting user profile...')
      try {
        const { error: profileError } = await supabaseAdmin
          .from('profiles')
          .delete()
          .eq('user_id', userId)

        if (profileError) {
          console.error('Error deleting profile:', profileError)
        } else {
          console.log('Deleted user profile')
        }
      } catch (profileError) {
        console.error('Profile deletion failed, continuing:', profileError)
      }
    } catch (dataError) {
      console.error('Data deletion phase error:', dataError)
    }

    // Step 5: Delete auth user (this is the final step)
    console.log('Deleting auth user...')
    try {
      const { error: authDeleteError } = await supabaseAdmin.auth.admin.deleteUser(userId)

      if (authDeleteError) {
        console.error('Error deleting auth user:', authDeleteError)
        return new Response(
          JSON.stringify({ 
            error: 'Failed to delete authentication account', 
            details: authDeleteError.message 
          }),
          { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        )
      }

      console.log(`Successfully deleted account for user: ${userId}`)

      return new Response(
        JSON.stringify({ 
          success: true, 
          message: 'Account deleted successfully',
          userId: userId
        }),
        { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    } catch (authError) {
      console.error('Auth deletion error:', authError)
      return new Response(
        JSON.stringify({ 
          error: 'Failed to delete account',
          details: authError instanceof Error ? authError.message : 'Unknown error'
        }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

  } catch (error) {
    console.error('Unexpected error:', error)
    return new Response(
      JSON.stringify({ 
        error: 'Internal server error',
        details: error instanceof Error ? error.message : 'Unknown error'
      }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )
  }
})
