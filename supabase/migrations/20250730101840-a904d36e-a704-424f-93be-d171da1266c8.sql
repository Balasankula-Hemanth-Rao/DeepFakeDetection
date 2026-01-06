-- Create user profiles table
CREATE TABLE public.profiles (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
  username TEXT,
  full_name TEXT,
  avatar_url TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS on profiles
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Create policies for profiles
CREATE POLICY "Users can view their own profile" 
ON public.profiles 
FOR SELECT 
USING (auth.uid() = user_id);

CREATE POLICY "Users can update their own profile" 
ON public.profiles 
FOR UPDATE 
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own profile" 
ON public.profiles 
FOR INSERT 
WITH CHECK (auth.uid() = user_id);

-- Create detection jobs table
CREATE TABLE public.detection_jobs (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  original_filename TEXT NOT NULL,
  file_path TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
  upload_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  analysis_start_time TIMESTAMP WITH TIME ZONE,
  analysis_end_time TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS on detection jobs
ALTER TABLE public.detection_jobs ENABLE ROW LEVEL SECURITY;

-- Create policies for detection jobs
CREATE POLICY "Users can view their own jobs" 
ON public.detection_jobs 
FOR SELECT 
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own jobs" 
ON public.detection_jobs 
FOR INSERT 
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own jobs" 
ON public.detection_jobs 
FOR UPDATE 
USING (auth.uid() = user_id);

-- Create detection results table
CREATE TABLE public.detection_results (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  job_id UUID NOT NULL UNIQUE REFERENCES public.detection_jobs(id) ON DELETE CASCADE,
  prediction TEXT NOT NULL CHECK (prediction IN ('REAL', 'FAKE')),
  confidence_score FLOAT NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
  visual_confidence FLOAT NOT NULL CHECK (visual_confidence >= 0 AND visual_confidence <= 1),
  audio_confidence FLOAT NOT NULL CHECK (audio_confidence >= 0 AND audio_confidence <= 1),
  analysis_duration_seconds FLOAT NOT NULL,
  anomaly_timestamps JSONB DEFAULT '[]',
  visual_analysis JSONB DEFAULT '{}',
  audio_analysis JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS on detection results
ALTER TABLE public.detection_results ENABLE ROW LEVEL SECURITY;

-- Create policies for detection results
CREATE POLICY "Users can view results for their own jobs" 
ON public.detection_results 
FOR SELECT 
USING (
  EXISTS (
    SELECT 1 FROM public.detection_jobs 
    WHERE detection_jobs.id = detection_results.job_id 
    AND detection_jobs.user_id = auth.uid()
  )
);

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_profiles_updated_at
BEFORE UPDATE ON public.profiles
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_detection_jobs_updated_at
BEFORE UPDATE ON public.detection_jobs
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();

-- Create storage bucket for video uploads
INSERT INTO storage.buckets (id, name, public) VALUES ('videos', 'videos', false);

-- Create storage policies for video uploads
CREATE POLICY "Users can upload their own videos" 
ON storage.objects 
FOR INSERT 
WITH CHECK (bucket_id = 'videos' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can view their own videos" 
ON storage.objects 
FOR SELECT 
USING (bucket_id = 'videos' AND auth.uid()::text = (storage.foldername(name))[1]);

-- Create function to handle new user registration
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (user_id, full_name, username)
  VALUES (
    NEW.id,
    NEW.raw_user_meta_data ->> 'full_name',
    NEW.raw_user_meta_data ->> 'username'
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create trigger for automatic profile creation
CREATE TRIGGER on_auth_user_created
AFTER INSERT ON auth.users
FOR EACH ROW
EXECUTE FUNCTION public.handle_new_user();