import { motion } from 'framer-motion';
import { ReactNode } from 'react';

interface PageContainerProps {
  children: ReactNode;
  className?: string;
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | '2xl' | 'full';
}

const maxWidthClasses = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  '2xl': 'max-w-2xl',
  full: 'max-w-4xl',
};

export const PageContainer = ({ 
  children, 
  className = '',
  maxWidth = 'full' 
}: PageContainerProps) => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5"
    >
      <div className={`container mx-auto px-4 py-8 ${className}`}>
        <div className={`${maxWidthClasses[maxWidth]} mx-auto`}>
          {children}
        </div>
      </div>
    </motion.div>
  );
};
