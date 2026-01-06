import { useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useTheme, Theme } from '@/contexts/ThemeContext';
import { toast } from 'sonner';

const ROUTES = ['/', '/dashboard', '/history'];

export const useKeyboardShortcuts = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { theme, setTheme } = useTheme();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      
      // Ignore if user is typing in an input/textarea
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target.isContentEditable
      ) {
        return;
      }

      // Escape: Close modals/dialogs (trigger click on overlay or close button)
      if (e.key === 'Escape') {
        const overlay = document.querySelector('[data-radix-dialog-overlay]') as HTMLElement;
        const closeBtn = document.querySelector('[data-radix-dialog-close]') as HTMLElement;
        if (overlay) overlay.click();
        else if (closeBtn) closeBtn.click();
        return;
      }

      // Ctrl+U: Go to upload/dashboard
      if (e.ctrlKey && e.key === 'u') {
        e.preventDefault();
        navigate('/dashboard');
        toast.info('Upload', { duration: 1500 });
      }

      // Ctrl+H: Go to history
      if (e.ctrlKey && e.key === 'h') {
        e.preventDefault();
        navigate('/history');
        toast.info('History', { duration: 1500 });
      }

      // Ctrl+T: Toggle theme (dark <-> light)
      if (e.ctrlKey && e.key === 't') {
        e.preventDefault();
        const newTheme: Theme = theme === 'dark' ? 'light' : 'dark';
        setTheme(newTheme);
        toast.info(`Theme: ${newTheme}`, { duration: 1500 });
      }

      // Arrow Left/Right: Navigate between main routes
      if (e.altKey && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
        e.preventDefault();
        const currentIndex = ROUTES.indexOf(location.pathname);
        if (currentIndex === -1) return;
        
        const newIndex = e.key === 'ArrowRight' 
          ? Math.min(currentIndex + 1, ROUTES.length - 1)
          : Math.max(currentIndex - 1, 0);
        
        if (newIndex !== currentIndex) {
          navigate(ROUTES[newIndex]);
          toast.info(`${ROUTES[newIndex] === '/' ? 'Home' : ROUTES[newIndex].slice(1)}`, { duration: 1500 });
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [navigate, theme, setTheme, location.pathname]);
};
