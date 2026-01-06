import { Moon, Sun } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useTheme, Theme } from '@/contexts/ThemeContext';
import { motion } from 'framer-motion';

export const ThemeToggle = () => {
  const { theme, setTheme } = useTheme();
  
  const isDark = theme === 'dark' || theme === 'cinematic' || theme === 'neon';
  
  const toggleTheme = () => {
    // Simple toggle between light and current dark variant
    if (isDark) {
      setTheme('light');
    } else {
      // Return to previous dark theme or default dark
      const stored = localStorage.getItem('aura-dark-variant') as Theme;
      setTheme(stored || 'dark');
    }
    
    // Store current dark variant for later
    if (isDark) {
      localStorage.setItem('aura-dark-variant', theme);
    }
  };

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={toggleTheme}
      className="text-muted-foreground hover:text-foreground relative overflow-hidden"
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      <motion.div
        initial={false}
        animate={{
          rotate: isDark ? 0 : 180,
          scale: 1
        }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
      >
        {isDark ? (
          <Sun className="w-5 h-5" />
        ) : (
          <Moon className="w-5 h-5" />
        )}
      </motion.div>
    </Button>
  );
};
