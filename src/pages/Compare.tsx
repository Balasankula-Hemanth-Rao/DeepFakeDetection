import { useState, useEffect } from 'react';
import { Navigate, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useAuth } from '@/hooks/useAuth';
import { supabase } from '@/integrations/supabase/client';
import { toast } from '@/hooks/use-toast';
import { PageHeader } from '@/components/ui/page-header';
import { LoadingState } from '@/components/ui/loading-state';
import { EmptyState } from '@/components/ui/empty-state';
import { SettingsPanel } from '@/components/SettingsPanel';
import Navbar from '@/components/landing/Navbar';
import { 
  FileVideo, 
  ArrowLeftRight,
  Shield,
  AlertTriangle,
  Eye,
  Volume2,
  Clock
} from 'lucide-react';

interface AnalysisRecord {
  id: string;
  original_filename: string;
  upload_timestamp: string;
  result?: {
    prediction: string;
    confidence_score: number;
    visual_confidence: number;
    audio_confidence: number;
    analysis_duration_seconds: number;
  };
}

const Compare = () => {
  const { user, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  const [analyses, setAnalyses] = useState<AnalysisRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [leftId, setLeftId] = useState<string>('');
  const [rightId, setRightId] = useState<string>('');

  useEffect(() => {
    const fetchAnalyses = async () => {
      if (!user) return;

      try {
        const { data: jobs, error: jobsError } = await supabase
          .from('detection_jobs')
          .select('*')
          .eq('user_id', user.id)
          .eq('status', 'completed')
          .order('upload_timestamp', { ascending: false });

        if (jobsError) throw jobsError;

        const analysesWithResults = await Promise.all(
          (jobs || []).map(async (job) => {
            const { data: result } = await supabase
              .from('detection_results')
              .select('prediction, confidence_score, visual_confidence, audio_confidence, analysis_duration_seconds')
              .eq('job_id', job.id)
              .maybeSingle();

            return {
              ...job,
              result: result || undefined
            };
          })
        );

        const completedAnalyses = analysesWithResults.filter(a => a.result);
        setAnalyses(completedAnalyses);
        
        // Auto-select first two if available
        if (completedAnalyses.length >= 2) {
          setLeftId(completedAnalyses[0].id);
          setRightId(completedAnalyses[1].id);
        } else if (completedAnalyses.length === 1) {
          setLeftId(completedAnalyses[0].id);
        }
      } catch (error: any) {
        console.error('Error fetching analyses:', error);
        toast({
          variant: "destructive",
          title: "Error",
          description: "Failed to load analysis history.",
        });
      } finally {
        setLoading(false);
      }
    };

    if (user) {
      fetchAnalyses();
    }
  }, [user]);

  const leftAnalysis = analyses.find(a => a.id === leftId);
  const rightAnalysis = analyses.find(a => a.id === rightId);

  if (authLoading) {
    return <LoadingState message="Loading..." />;
  }

  if (!user) {
    return <Navigate to="/auth" replace />;
  }

  const renderComparisonCard = (analysis: AnalysisRecord | undefined, side: 'left' | 'right') => {
    if (!analysis || !analysis.result) {
      return (
        <Card className="glass h-full flex items-center justify-center min-h-[400px]">
          <CardContent className="text-center">
            <FileVideo className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
            <p className="text-muted-foreground">Select an analysis to compare</p>
          </CardContent>
        </Card>
      );
    }

    const isFake = analysis.result.prediction === 'FAKE';
    const confidence = Math.round(analysis.result.confidence_score * 100);

    return (
      <Card className={`glass border-2 ${isFake ? 'border-destructive/30' : 'border-success/30'}`}>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg truncate max-w-[200px]" title={analysis.original_filename}>
              {analysis.original_filename}
            </CardTitle>
            <Badge 
              variant={isFake ? "destructive" : "default"}
              className={`${!isFake ? 'bg-success hover:bg-success/80' : ''}`}
            >
              {isFake ? (
                <><AlertTriangle className="w-3 h-3 mr-1" /> Fake</>
              ) : (
                <><Shield className="w-3 h-3 mr-1" /> Real</>
              )}
            </Badge>
          </div>
          <CardDescription>
            {new Date(analysis.upload_timestamp).toLocaleString()}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Main Confidence */}
          <div className="text-center">
            <p className="text-4xl font-bold mb-2">{confidence}%</p>
            <p className="text-sm text-muted-foreground">Overall Confidence</p>
            <Progress value={confidence} className="h-2 mt-2" />
          </div>

          {/* Detailed Metrics */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Eye className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm">Visual Analysis</span>
              </div>
              <span className="font-medium">
                {Math.round(analysis.result.visual_confidence * 100)}%
              </span>
            </div>
            <Progress value={analysis.result.visual_confidence * 100} className="h-1.5" />

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Volume2 className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm">Audio Analysis</span>
              </div>
              <span className="font-medium">
                {Math.round(analysis.result.audio_confidence * 100)}%
              </span>
            </div>
            <Progress value={analysis.result.audio_confidence * 100} className="h-1.5" />

            <div className="flex items-center justify-between pt-2 border-t border-border/50">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm">Processing Time</span>
              </div>
              <span className="font-medium">
                {Math.round(analysis.result.analysis_duration_seconds)}s
              </span>
            </div>
          </div>

          {/* View Details Button */}
          <Button 
            variant="outline" 
            className="w-full"
            onClick={() => navigate(`/results/${analysis.id}`)}
          >
            View Full Results
          </Button>
        </CardContent>
      </Card>
    );
  };

  return (
    <>
      <SettingsPanel isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <Navbar />
      
      <div className="min-h-screen pt-20 bg-gradient-to-br from-background via-background to-primary/5 transition-smooth">
        <PageHeader
          title="Compare Analyses"
          onSettingsClick={() => setSettingsOpen(true)}
        />

        <main className="container mx-auto px-4 py-8">
          {loading ? (
            <LoadingState message="Loading analyses..." />
          ) : analyses.length < 2 ? (
            <EmptyState
              icon={ArrowLeftRight}
              title="Not enough analyses"
              description="You need at least 2 completed analyses to compare. Upload more videos to get started."
              action={{
                label: "Start Analysis",
                onClick: () => navigate('/dashboard')
              }}
            />
          ) : (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="max-w-6xl mx-auto space-y-6"
            >
              {/* Selection Row */}
              <Card className="glass">
                <CardContent className="p-4">
                  <div className="flex flex-col md:flex-row items-center gap-4">
                    <Select value={leftId} onValueChange={setLeftId}>
                      <SelectTrigger className="flex-1">
                        <SelectValue placeholder="Select first analysis" />
                      </SelectTrigger>
                      <SelectContent>
                        {analyses.map(a => (
                          <SelectItem key={a.id} value={a.id} disabled={a.id === rightId}>
                            {a.original_filename}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>

                    <div className="flex items-center justify-center w-10 h-10 rounded-full bg-muted">
                      <ArrowLeftRight className="w-5 h-5 text-muted-foreground" />
                    </div>

                    <Select value={rightId} onValueChange={setRightId}>
                      <SelectTrigger className="flex-1">
                        <SelectValue placeholder="Select second analysis" />
                      </SelectTrigger>
                      <SelectContent>
                        {analyses.map(a => (
                          <SelectItem key={a.id} value={a.id} disabled={a.id === leftId}>
                            {a.original_filename}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </CardContent>
              </Card>

              {/* Comparison Grid */}
              <div className="grid md:grid-cols-2 gap-6">
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  {renderComparisonCard(leftAnalysis, 'left')}
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 }}
                >
                  {renderComparisonCard(rightAnalysis, 'right')}
                </motion.div>
              </div>

              {/* Difference Summary */}
              {leftAnalysis?.result && rightAnalysis?.result && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                >
                  <Card className="glass">
                    <CardHeader>
                      <CardTitle className="text-base">Comparison Summary</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-3 gap-4 text-center">
                        <div>
                          <p className="text-sm text-muted-foreground mb-1">Confidence Difference</p>
                          <p className="text-xl font-bold">
                            {Math.abs(
                              Math.round(leftAnalysis.result.confidence_score * 100) -
                              Math.round(rightAnalysis.result.confidence_score * 100)
                            )}%
                          </p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground mb-1">Visual Difference</p>
                          <p className="text-xl font-bold">
                            {Math.abs(
                              Math.round(leftAnalysis.result.visual_confidence * 100) -
                              Math.round(rightAnalysis.result.visual_confidence * 100)
                            )}%
                          </p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground mb-1">Audio Difference</p>
                          <p className="text-xl font-bold">
                            {Math.abs(
                              Math.round(leftAnalysis.result.audio_confidence * 100) -
                              Math.round(rightAnalysis.result.audio_confidence * 100)
                            )}%
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              )}
            </motion.div>
          )}
        </main>
      </div>
    </>
  );
};

export default Compare;
