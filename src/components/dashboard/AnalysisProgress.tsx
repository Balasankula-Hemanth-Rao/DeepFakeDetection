import { motion } from 'framer-motion';
import { Loader2, Brain, Eye, Waves, RotateCcw } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface AnalysisProgressProps {
  fileName: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  onReset: () => void;
}

const statusMessages = {
  pending: 'Preparing analysis...',
  processing: 'AI Analysis in Progress',
  completed: 'Analysis Complete!',
  failed: 'Analysis Failed',
};

const analysisSteps = [
  { icon: Eye, label: 'Scanning facial regions and artifacts', delay: 0 },
  { icon: Waves, label: 'Analyzing audio-visual synchronization', delay: 0.4 },
  { icon: Brain, label: 'Processing with neural networks', delay: 0.8 },
];

export const AnalysisProgress = ({ fileName, status, onReset }: AnalysisProgressProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="glass-strong border-border/50 overflow-hidden">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl">{statusMessages[status]}</CardTitle>
          <CardDescription className="truncate max-w-sm mx-auto">
            {fileName}
          </CardDescription>
        </CardHeader>
        
        <CardContent className="space-y-8">
          {status === 'processing' && (
            <>
              {/* Main spinner */}
              <div className="flex justify-center">
                <motion.div
                  className="relative w-24 h-24"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
                >
                  <div className="absolute inset-0 rounded-full border-4 border-primary/10" />
                  <motion.div
                    className="absolute inset-0 rounded-full border-4 border-transparent border-t-primary border-r-primary/50"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                  />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Loader2 className="w-10 h-10 text-primary" />
                  </div>
                </motion.div>
              </div>

              {/* Analysis steps */}
              <div className="space-y-3 max-w-sm mx-auto">
                {analysisSteps.map((step, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: step.delay, duration: 0.4 }}
                    className="flex items-center gap-3 p-3 rounded-lg bg-muted/20"
                  >
                    <motion.div
                      animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
                      transition={{ duration: 2, repeat: Infinity, delay: step.delay }}
                    >
                      <step.icon className="w-5 h-5 text-primary" />
                    </motion.div>
                    <span className="text-sm text-muted-foreground flex-1">{step.label}</span>
                    <motion.div
                      className="w-2 h-2 rounded-full bg-primary"
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ duration: 1.5, repeat: Infinity, delay: step.delay }}
                    />
                  </motion.div>
                ))}
              </div>
            </>
          )}

          {status === 'failed' && (
            <div className="text-center py-4">
              <p className="text-destructive mb-4">
                There was an error processing your video. Please try again.
              </p>
            </div>
          )}

          <div className="flex justify-center">
            <Button variant="outline" onClick={onReset} className="gap-2">
              <RotateCcw className="w-4 h-4" />
              Upload Another Video
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};
