import { motion } from 'framer-motion';
import { Loader2 } from 'lucide-react';

interface LoadingStateProps {
  message?: string;
  size?: 'sm' | 'md' | 'lg';
  fullScreen?: boolean;
}

export const LoadingState = ({ 
  message = 'Loading...', 
  size = 'md',
  fullScreen = false 
}: LoadingStateProps) => {
  const sizeClasses = {
    sm: { container: 'w-8 h-8', icon: 'w-4 h-4' },
    md: { container: 'w-16 h-16', icon: 'w-8 h-8' },
    lg: { container: 'w-24 h-24', icon: 'w-12 h-12' },
  };

  const content = (
    <motion.div 
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      className="flex flex-col items-center justify-center gap-4"
    >
      <div className={`relative ${sizeClasses[size].container}`}>
        {/* Outer ring */}
        <motion.div
          className="absolute inset-0 rounded-full border-2 border-primary/20"
          animate={{ rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
        />
        {/* Inner spinner */}
        <motion.div
          className="absolute inset-0 rounded-full border-2 border-transparent border-t-primary"
          animate={{ rotate: -360 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
        />
        {/* Center icon */}
        <div className="absolute inset-0 flex items-center justify-center">
          <Loader2 className={`${sizeClasses[size].icon} text-primary animate-spin`} />
        </div>
      </div>
      {message && (
        <motion.p 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="text-sm text-muted-foreground"
        >
          {message}
        </motion.p>
      )}
    </motion.div>
  );

  if (fullScreen) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5 flex items-center justify-center">
        {content}
      </div>
    );
  }

  return content;
};
