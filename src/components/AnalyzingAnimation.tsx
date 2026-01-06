import { motion } from 'framer-motion';
import { Loader2, Brain, Eye, Waves } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';

interface AnalyzingAnimationProps {
  fileName: string;
}

export const AnalyzingAnimation = ({ fileName }: AnalyzingAnimationProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ duration: 0.4 }}
      className="fixed inset-0 bg-background/95 backdrop-blur-xl z-50 flex items-center justify-center"
    >
      <Card className="glass-strong max-w-md w-full mx-4">
        <CardContent className="pt-6 space-y-6">
          {/* Main Loading Animation */}
          <div className="text-center space-y-4">
            <motion.div
              className="relative w-24 h-24 mx-auto"
              animate={{ rotate: 360 }}
              transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
            >
              <div className="absolute inset-0 rounded-full border-4 border-primary/20" />
              <motion.div
                className="absolute inset-0 rounded-full border-4 border-transparent border-t-primary"
                animate={{ rotate: 360 }}
                transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
              />
              <div className="absolute inset-0 flex items-center justify-center">
                <Loader2 className="w-12 h-12 text-primary" />
              </div>
            </motion.div>

            <div>
              <h3 className="text-2xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent mb-2">
                Analyzing Video
              </h3>
              <p className="text-sm text-muted-foreground">
                {fileName}
              </p>
            </div>
          </div>

          {/* Analysis Steps */}
          <div className="space-y-3">
            {[
              { icon: Eye, label: 'Scanning facial regions', delay: 0 },
              { icon: Waves, label: 'Analyzing audio-visual sync', delay: 0.3 },
              { icon: Brain, label: 'Processing neural patterns', delay: 0.6 },
            ].map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: step.delay, duration: 0.4 }}
                className="flex items-center space-x-3 p-3 rounded-lg bg-muted/30"
              >
                <motion.div
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 2, repeat: Infinity, delay: step.delay }}
                >
                  <step.icon className="w-5 h-5 text-primary" />
                </motion.div>
                <span className="text-sm text-foreground">{step.label}</span>
                <motion.div
                  className="ml-auto"
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ duration: 1.5, repeat: Infinity, delay: step.delay }}
                >
                  <div className="w-2 h-2 rounded-full bg-primary" />
                </motion.div>
              </motion.div>
            ))}
          </div>

          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-primary to-secondary"
                initial={{ width: '0%' }}
                animate={{ width: '100%' }}
                transition={{ duration: 2, ease: "easeInOut" }}
              />
            </div>
            <p className="text-xs text-center text-muted-foreground">
              Deep learning analysis in progress...
            </p>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};
