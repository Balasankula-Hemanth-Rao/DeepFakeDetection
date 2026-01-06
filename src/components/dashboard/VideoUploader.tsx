import { useState, useCallback, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileVideo, X, CheckCircle2, AlertCircle, Play, Info } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { toast } from '@/hooks/use-toast';

interface VideoUploaderProps {
  onUpload: (file: File) => Promise<void>;
  isUploading: boolean;
  uploadProgress: number;
}

interface FileValidation {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

const SUPPORTED_FORMATS = ['video/mp4', 'video/avi', 'video/mov', 'video/quicktime', 'video/webm', 'video/x-msvideo'];
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

export const VideoUploader = ({ onUpload, isUploading, uploadProgress }: VideoUploaderProps) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [videoDuration, setVideoDuration] = useState<number | null>(null);
  const [validation, setValidation] = useState<FileValidation | null>(null);
  const [uploadStage, setUploadStage] = useState<'idle' | 'validating' | 'uploading' | 'processing' | 'complete'>('idle');
  const videoRef = useRef<HTMLVideoElement>(null);

  const validateFile = (file: File): FileValidation => {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Check file type
    if (!SUPPORTED_FORMATS.includes(file.type)) {
      errors.push(`Unsupported format: ${file.type || 'unknown'}. Use MP4, AVI, MOV, or WebM.`);
    }

    // Check file size
    if (file.size > MAX_FILE_SIZE) {
      errors.push(`File too large: ${(file.size / (1024 * 1024)).toFixed(1)}MB. Max is 50MB.`);
    }

    // Warnings
    if (file.size > 30 * 1024 * 1024) {
      warnings.push('Large file may take longer to upload and analyze.');
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
    };
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const processFile = useCallback((file: File) => {
    setUploadStage('validating');
    
    const fileValidation = validateFile(file);
    setValidation(fileValidation);

    if (!fileValidation.valid) {
      setUploadStage('idle');
      toast({
        variant: "destructive",
        title: "Invalid file",
        description: fileValidation.errors[0],
      });
      return;
    }

    // Show warnings if any
    if (fileValidation.warnings.length > 0) {
      toast({
        title: "Heads up",
        description: fileValidation.warnings[0],
      });
    }

    setSelectedFile(file);

    // Create video preview
    const previewUrl = URL.createObjectURL(file);
    setVideoPreview(previewUrl);

    // Get video duration
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.onloadedmetadata = () => {
      setVideoDuration(video.duration);
      URL.revokeObjectURL(video.src);
    };
    video.src = previewUrl;

    setUploadStage('idle');
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  }, [processFile]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      processFile(e.target.files[0]);
    }
  };

  const startUpload = async () => {
    if (!selectedFile) return;
    setUploadStage('uploading');
    await onUpload(selectedFile);
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setVideoPreview(null);
    setVideoDuration(null);
    setValidation(null);
    setUploadStage('idle');
  };

  // Update stage based on upload progress
  useEffect(() => {
    if (isUploading && uploadProgress > 0 && uploadProgress < 100) {
      setUploadStage('uploading');
    } else if (uploadProgress === 100) {
      setUploadStage('processing');
    }
  }, [isUploading, uploadProgress]);

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const getProgressLabel = () => {
    switch (uploadStage) {
      case 'validating':
        return 'Validating file...';
      case 'uploading':
        return `Uploading... ${Math.round(uploadProgress)}%`;
      case 'processing':
        return 'Processing video...';
      case 'complete':
        return 'Complete!';
      default:
        return '';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="glass-strong border-border/50 overflow-hidden">
        <CardHeader className="text-center pb-4">
          <CardTitle className="text-2xl">Upload Video for Analysis</CardTitle>
          <CardDescription>
            Drag and drop or browse to detect deepfakes with our advanced AI
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <AnimatePresence mode="wait">
            {!selectedFile ? (
              <motion.div
                key="dropzone"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className={`relative border-2 border-dashed rounded-xl p-12 text-center transition-all duration-300 ${
                  dragActive 
                    ? 'border-primary bg-primary/5 scale-[1.02]' 
                    : 'border-border/50 hover:border-primary/50 hover:bg-primary/5'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleFileSelect}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  disabled={isUploading}
                />
                
                <motion.div
                  animate={dragActive ? { scale: 1.1, y: -10 } : { scale: 1, y: 0 }}
                  transition={{ duration: 0.2 }}
                  className="pointer-events-none"
                >
                  <motion.div 
                    className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-primary/10 flex items-center justify-center"
                    animate={{ y: [0, -5, 0] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    <FileVideo className="w-10 h-10 text-primary" />
                  </motion.div>
                  
                  <h3 className="text-lg font-semibold mb-2">
                    {dragActive ? 'Drop your video here' : 'Drag and drop your video here'}
                  </h3>
                  <p className="text-muted-foreground mb-6">
                    or click anywhere to browse
                  </p>
                  
                  <Button variant="outline" className="pointer-events-none">
                    <Upload className="w-4 h-4 mr-2" />
                    Choose Video File
                  </Button>
                </motion.div>
              </motion.div>
            ) : (
              <motion.div 
                key="preview"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="space-y-4"
              >
                {/* Video Preview */}
                <div className="relative rounded-xl overflow-hidden bg-black/50 aspect-video max-h-64">
                  {videoPreview && (
                    <video
                      ref={videoRef}
                      src={videoPreview}
                      className="w-full h-full object-contain"
                      controls={!isUploading}
                      muted
                    />
                  )}
                  {!isUploading && (
                    <div className="absolute top-2 right-2">
                      <Button 
                        variant="secondary" 
                        size="icon" 
                        onClick={clearSelection}
                        className="h-8 w-8 bg-background/80 backdrop-blur-sm"
                      >
                        <X className="w-4 h-4" />
                      </Button>
                    </div>
                  )}
                </div>

                {/* File Info */}
                <div className="flex items-center gap-4 p-4 rounded-xl bg-muted/30">
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                    <FileVideo className="w-6 h-6 text-primary" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium truncate">{selectedFile.name}</p>
                    <div className="flex items-center gap-3 text-sm text-muted-foreground">
                      <span>{formatFileSize(selectedFile.size)}</span>
                      {videoDuration && (
                        <>
                          <span>•</span>
                          <span>{formatDuration(videoDuration)}</span>
                        </>
                      )}
                    </div>
                  </div>
                  {validation?.valid && !isUploading && (
                    <Badge variant="outline" className="text-success border-success/50">
                      <CheckCircle2 className="w-3 h-3 mr-1" />
                      Ready
                    </Badge>
                  )}
                </div>

                {/* Validation Warnings */}
                {validation?.warnings && validation.warnings.length > 0 && (
                  <div className="flex items-start gap-2 p-3 rounded-lg bg-warning/10 text-warning text-sm">
                    <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
                    <span>{validation.warnings[0]}</span>
                  </div>
                )}
                
                {/* Upload Progress */}
                {isUploading && (
                  <motion.div 
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="space-y-3"
                  >
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">{getProgressLabel()}</span>
                      <span className="font-medium">{Math.round(uploadProgress)}%</span>
                    </div>
                    <Progress value={uploadProgress} className="h-2" />
                    
                    {/* Stage Indicators */}
                    <div className="flex items-center justify-center gap-8 pt-2">
                      <div className={`flex items-center gap-2 text-xs ${uploadStage === 'uploading' || uploadStage === 'processing' || uploadStage === 'complete' ? 'text-primary' : 'text-muted-foreground'}`}>
                        <div className={`w-2 h-2 rounded-full ${uploadStage === 'uploading' ? 'bg-primary animate-pulse' : uploadStage === 'processing' || uploadStage === 'complete' ? 'bg-success' : 'bg-muted'}`} />
                        Upload
                      </div>
                      <div className={`flex items-center gap-2 text-xs ${uploadStage === 'processing' || uploadStage === 'complete' ? 'text-primary' : 'text-muted-foreground'}`}>
                        <div className={`w-2 h-2 rounded-full ${uploadStage === 'processing' ? 'bg-primary animate-pulse' : uploadStage === 'complete' ? 'bg-success' : 'bg-muted'}`} />
                        Process
                      </div>
                      <div className={`flex items-center gap-2 text-xs ${uploadStage === 'complete' ? 'text-success' : 'text-muted-foreground'}`}>
                        <div className={`w-2 h-2 rounded-full ${uploadStage === 'complete' ? 'bg-success' : 'bg-muted'}`} />
                        Analyze
                      </div>
                    </div>
                  </motion.div>
                )}
                
                {uploadProgress === 100 && (
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex items-center gap-2 text-success justify-center"
                  >
                    <CheckCircle2 className="w-5 h-5" />
                    <span className="text-sm font-medium">Upload complete, analyzing...</span>
                  </motion.div>
                )}

                {/* Start Analysis Button */}
                {!isUploading && validation?.valid && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.2 }}
                  >
                    <Button 
                      onClick={startUpload} 
                      className="w-full h-12 text-base"
                      size="lg"
                    >
                      <Play className="w-5 h-5 mr-2" />
                      Start Analysis
                    </Button>
                  </motion.div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
          
          <p className="mt-6 text-xs text-muted-foreground text-center">
            Supported: MP4, AVI, MOV, WebM • Max size: 50MB
          </p>
        </CardContent>
      </Card>
    </motion.div>
  );
};