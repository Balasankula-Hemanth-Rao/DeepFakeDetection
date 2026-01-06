import { useState, useEffect } from 'react';
import { Navigate, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useAuth } from '@/hooks/useAuth';
import { supabase } from '@/integrations/supabase/client';
import { toast } from '@/hooks/use-toast';
import { PageHeader } from '@/components/ui/page-header';
import { LoadingState } from '@/components/ui/loading-state';
import { EmptyState } from '@/components/ui/empty-state';
import Navbar from '@/components/landing/Navbar';
import { AnalyticsCharts } from '@/components/dashboard/AnalyticsCharts';
import { 
  FileVideo, 
  Calendar, 
  Search, 
  Filter,
  ArrowUpDown,
  Eye,
  Trash2,
  Clock,
  Shield,
  AlertTriangle,
  ArrowLeftRight
} from 'lucide-react';

interface AnalysisRecord {
  id: string;
  original_filename: string;
  upload_timestamp: string;
  status: string;
  result?: {
    prediction: string;
    confidence_score: number;
  };
}

const History = () => {
  const { user, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  const [analyses, setAnalyses] = useState<AnalysisRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortOrder, setSortOrder] = useState<'newest' | 'oldest'>('newest');
  const [filterStatus, setFilterStatus] = useState<'all' | 'real' | 'fake'>('all');

  useEffect(() => {
    const fetchAnalyses = async () => {
      if (!user) return;

      try {
        const { data: jobs, error: jobsError } = await supabase
          .from('detection_jobs')
          .select('*')
          .eq('user_id', user.id)
          .order('upload_timestamp', { ascending: sortOrder === 'oldest' });

        if (jobsError) throw jobsError;

        // Fetch results for each job
        const analysesWithResults = await Promise.all(
          (jobs || []).map(async (job) => {
            const { data: result } = await supabase
              .from('detection_results')
              .select('prediction, confidence_score')
              .eq('job_id', job.id)
              .maybeSingle();

            return {
              ...job,
              result: result || undefined
            };
          })
        );

        setAnalyses(analysesWithResults);
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
  }, [user, sortOrder]);

  const filteredAnalyses = analyses.filter((analysis) => {
    const matchesSearch = analysis.original_filename
      .toLowerCase()
      .includes(searchQuery.toLowerCase());
    
    if (filterStatus === 'all') return matchesSearch;
    if (filterStatus === 'real') return matchesSearch && analysis.result?.prediction === 'REAL';
    if (filterStatus === 'fake') return matchesSearch && analysis.result?.prediction === 'FAKE';
    
    return matchesSearch;
  });

  const handleDelete = async (id: string) => {
    try {
      const { error } = await supabase
        .from('detection_jobs')
        .delete()
        .eq('id', id);

      if (error) throw error;

      setAnalyses(analyses.filter(a => a.id !== id));
      toast({
        title: "Deleted",
        description: "Analysis record has been deleted.",
      });
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to delete analysis.",
      });
    }
  };

  if (authLoading) {
    return <LoadingState message="Loading..." />;
  }

  if (!user) {
    return <Navigate to="/auth" replace />;
  }

  return (
    <>
      <Navbar />
      
      <div className="min-h-screen pt-20 bg-gradient-to-br from-background via-background to-primary/5 transition-smooth">
        <main className="container mx-auto px-4 py-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="max-w-5xl mx-auto space-y-6"
          >
            {/* Filters */}
            <Card className="glass">
              <CardContent className="p-4">
                <div className="flex flex-col md:flex-row gap-4">
                  <div className="relative flex-1">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input
                      placeholder="Search by filename..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                  
                  <Select value={filterStatus} onValueChange={(value: 'all' | 'real' | 'fake') => setFilterStatus(value)}>
                    <SelectTrigger className="w-full md:w-40">
                      <Filter className="w-4 h-4 mr-2" />
                      <SelectValue placeholder="Filter" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Results</SelectItem>
                      <SelectItem value="real">Authentic Only</SelectItem>
                      <SelectItem value="fake">Deepfake Only</SelectItem>
                    </SelectContent>
                  </Select>

                  <Select value={sortOrder} onValueChange={(value: 'newest' | 'oldest') => setSortOrder(value)}>
                    <SelectTrigger className="w-full md:w-40">
                      <ArrowUpDown className="w-4 h-4 mr-2" />
                      <SelectValue placeholder="Sort" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="newest">Newest First</SelectItem>
                      <SelectItem value="oldest">Oldest First</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>

            {/* Compare Button */}
            {analyses.length >= 2 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
              >
                <Button 
                  onClick={() => navigate('/compare')}
                  className="w-full"
                  variant="outline"
                >
                  <ArrowLeftRight className="w-4 h-4 mr-2" />
                  Compare Analyses Side-by-Side
                </Button>
              </motion.div>
            )}

            {/* Analytics Charts */}
            {analyses.length > 0 && (
              <AnalyticsCharts analyses={analyses} />
            )}
            {loading ? (
              <LoadingState message="Loading history..." />
            ) : filteredAnalyses.length === 0 ? (
              <EmptyState
                icon={FileVideo}
                title="No analyses found"
                description={searchQuery ? "Try adjusting your search or filters" : "Upload a video to get started with deepfake detection"}
                action={{
                  label: "Start Analysis",
                  onClick: () => navigate('/dashboard')
                }}
              />
            ) : (
              <div className="space-y-4">
                {filteredAnalyses.map((analysis, index) => {
                  const isFake = analysis.result?.prediction === 'FAKE';
                  const confidencePercent = analysis.result 
                    ? Math.round(analysis.result.confidence_score * 100) 
                    : null;

                  return (
                    <motion.div
                      key={analysis.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05, duration: 0.3 }}
                    >
                      <Card className="glass hover:shadow-glow-primary transition-all duration-300 group">
                        <CardContent className="p-4">
                          <div className="flex items-center gap-4">
                            {/* Thumbnail */}
                            <div className="flex-shrink-0 w-16 h-16 bg-muted rounded-lg flex items-center justify-center">
                              <FileVideo className="w-8 h-8 text-muted-foreground" />
                            </div>

                            {/* Info */}
                            <div className="flex-1 min-w-0">
                              <h3 className="font-semibold truncate">{analysis.original_filename}</h3>
                              <div className="flex items-center gap-4 text-sm text-muted-foreground mt-1">
                                <span className="flex items-center">
                                  <Calendar className="w-3 h-3 mr-1" />
                                  {new Date(analysis.upload_timestamp).toLocaleDateString()}
                                </span>
                                <span className="flex items-center">
                                  <Clock className="w-3 h-3 mr-1" />
                                  {new Date(analysis.upload_timestamp).toLocaleTimeString()}
                                </span>
                              </div>
                            </div>

                            {/* Status/Result */}
                            <div className="flex-shrink-0 flex items-center gap-3">
                              {analysis.result ? (
                                <>
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
                                  <span className="text-sm font-medium">{confidencePercent}%</span>
                                </>
                              ) : (
                                <Badge variant="secondary">
                                  {analysis.status === 'processing' ? 'Processing...' : 'Pending'}
                                </Badge>
                              )}
                            </div>

                            {/* Actions */}
                            <div className="flex-shrink-0 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                              {analysis.result && (
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => navigate(`/results/${analysis.id}`)}
                                  className="transition-smooth hover:shadow-glow-primary"
                                >
                                  <Eye className="w-4 h-4 mr-1" />
                                  View
                                </Button>
                              )}
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleDelete(analysis.id)}
                                className="text-destructive hover:text-destructive hover:bg-destructive/10"
                              >
                                <Trash2 className="w-4 h-4" />
                              </Button>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </motion.div>
                  );
                })}
              </div>
            )}

            {/* Stats */}
            {!loading && analyses.length > 0 && (
              <Card className="glass">
                <CardHeader>
                  <CardTitle className="text-lg">Summary</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <p className="text-2xl font-bold">{analyses.length}</p>
                      <p className="text-sm text-muted-foreground">Total Analyses</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-success">
                        {analyses.filter(a => a.result?.prediction === 'REAL').length}
                      </p>
                      <p className="text-sm text-muted-foreground">Authentic</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-destructive">
                        {analyses.filter(a => a.result?.prediction === 'FAKE').length}
                      </p>
                      <p className="text-sm text-muted-foreground">Deepfakes</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </motion.div>
        </main>
      </div>
    </>
  );
};

export default History;
