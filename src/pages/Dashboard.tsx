import { useState, useCallback } from 'react';
import { Navigate, useNavigate } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { useAuth } from '@/hooks/useAuth';
import { supabase } from '@/integrations/supabase/client';
import { toast } from '@/hooks/use-toast';
import { History } from 'lucide-react';
import { PageHeader } from '@/components/ui/page-header';
import { PageContainer } from '@/components/ui/page-container';
import { SettingsPanel } from '@/components/SettingsPanel';
import { AnalyzingAnimation } from '@/components/AnalyzingAnimation';
import { VideoUploader } from '@/components/dashboard/VideoUploader';
import { AnalysisProgress } from '@/components/dashboard/AnalysisProgress';
import Navbar from '@/components/landing/Navbar';

interface DetectionJob {
  id: string;
  original_filename: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  upload_timestamp: string;
}

const Dashboard = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [uploading, setUploading] = useState(false);
  const [currentJob, setCurrentJob] = useState<DetectionJob | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [analyzing, setAnalyzing] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  if (!user) {
    return <Navigate to="/auth" replace />;
  }

  const pollJobStatus = useCallback(async (jobId: string) => {
    let attempts = 0;
    const maxAttempts = 60;
    
    const poll = async () => {
      try {
        const { data: job, error } = await supabase
          .from('detection_jobs')
          .select('*')
          .eq('id', jobId)
          .single();

        if (error) throw error;

        setCurrentJob(job as DetectionJob);

        if (job.status === 'completed') {
          setAnalyzing(true);
          setTimeout(() => {
            navigate(`/results/${jobId}`);
          }, 2000);
          return;
        } else if (job.status === 'failed') {
          toast({
            variant: "destructive",
            title: "Analysis Failed",
            description: "There was an error processing your video. Please try again.",
          });
          return;
        }

        attempts++;
        if (attempts < maxAttempts && (job.status === 'pending' || job.status === 'processing')) {
          setTimeout(poll, 5000);
        } else if (attempts >= maxAttempts) {
          toast({
            variant: "destructive",
            title: "Timeout",
            description: "Analysis is taking longer than expected. Please try again.",
          });
          setCurrentJob(null);
        }
      } catch (error: any) {
        console.error('Polling error:', error);
        toast({
          variant: "destructive",
          title: "Error",
          description: "Failed to check analysis status.",
        });
        setCurrentJob(null);
      }
    };

    poll();
  }, [navigate]);

  const handleFileUpload = async (file: File) => {
    // Validate file type
    if (!file.type.startsWith('video/')) {
      toast({
        variant: "destructive",
        title: "Invalid file type",
        description: "Please upload a video file (MP4, AVI, MOV, WebM).",
      });
      return;
    }

    // Validate file size (50MB limit)
    if (file.size > 50 * 1024 * 1024) {
      toast({
        variant: "destructive",
        title: "File too large",
        description: "Please upload a video smaller than 50MB.",
      });
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    try {
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90));
      }, 200);

      // Upload file to Supabase Storage
      const fileExt = file.name.split('.').pop();
      const fileName = `${user.id}/${Date.now()}.${fileExt}`;
      
      const { error: uploadError } = await supabase.storage
        .from('videos')
        .upload(fileName, file);

      clearInterval(progressInterval);

      if (uploadError) throw uploadError;

      // Create detection job
      const { data: job, error: jobError } = await supabase
        .from('detection_jobs')
        .insert({
          user_id: user.id,
          original_filename: file.name,
          file_path: fileName,
          status: 'pending'
        })
        .select()
        .single();

      if (jobError) throw jobError;

      setCurrentJob(job as DetectionJob);
      setUploadProgress(100);

      // Start AI processing
      await supabase.functions.invoke('ai-detection', {
        body: { jobId: job.id }
      });

      // Start polling for results
      pollJobStatus(job.id);

      toast({
        title: "Upload successful!",
        description: "Your video is now being analyzed by our AI.",
      });

    } catch (error: any) {
      console.error('Upload error:', error);
      toast({
        variant: "destructive",
        title: "Upload failed",
        description: error.message || "Failed to upload video. Please try again.",
      });
      setCurrentJob(null);
    } finally {
      setUploading(false);
    }
  };

  const resetUpload = () => {
    setCurrentJob(null);
    setUploadProgress(0);
  };

  return (
    <>
      <AnimatePresence>
        {analyzing && currentJob && (
          <AnalyzingAnimation fileName={currentJob.original_filename} />
        )}
      </AnimatePresence>

      <SettingsPanel isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />

      <Navbar />
      <div className="min-h-screen pt-20 bg-gradient-to-br from-background via-background to-primary/5">
        <PageContainer maxWidth="2xl">
          {!currentJob ? (
            <VideoUploader 
              onUpload={handleFileUpload}
              isUploading={uploading}
              uploadProgress={uploadProgress}
            />
          ) : (
            <AnalysisProgress 
              fileName={currentJob.original_filename}
              status={currentJob.status}
              onReset={resetUpload}
            />
          )}
        </PageContainer>
      </div>
    </>
  );
};

export default Dashboard;
