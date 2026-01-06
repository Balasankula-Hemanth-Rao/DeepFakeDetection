import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Settings, LogOut, User } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/hooks/useAuth';
import { supabase } from '@/integrations/supabase/client';

interface PageHeaderProps {
  title?: string;
  subtitle?: string;
  onSettingsClick?: () => void;
  actions?: React.ReactNode;
  showBackButton?: boolean;
  backTo?: string;
}

export const PageHeader = ({ 
  title,
  subtitle,
  onSettingsClick,
  actions,
  showBackButton = false,
  backTo = '/dashboard'
}: PageHeaderProps) => {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const [profileName, setProfileName] = useState<string | null>(null);

  useEffect(() => {
    const fetchProfile = async () => {
      if (!user) return;
      
      const { data } = await supabase
        .from('profiles')
        .select('full_name, username')
        .eq('user_id', user.id)
        .maybeSingle();
      
      if (data) {
        setProfileName(data.full_name || data.username || null);
      }
    };

    fetchProfile();
  }, [user]);

  const displayName = profileName || user?.email?.split('@')[0] || 'User';

  return (
    <motion.header 
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.4 }}
      className="border-b border-border/50 bg-card/80 backdrop-blur-lg"
    >
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            {showBackButton && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => navigate(backTo)}
                className="transition-smooth hover:bg-primary/10"
              >
                ← Back
              </Button>
            )}
            {(title || subtitle) && (
              <div>
                {title && (
                  <h1 className="text-xl font-semibold text-foreground">
                    {title}
                  </h1>
                )}
                {subtitle && (
                  <span className="text-sm text-muted-foreground">{subtitle}</span>
                )}
              </div>
            )}
          </div>
          
          <div className="flex items-center gap-3">
            {user && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <User className="w-4 h-4" />
                <span className="hidden sm:inline">{displayName}</span>
              </div>
            )}
            {actions}
            {onSettingsClick && (
              <Button 
                variant="ghost" 
                size="icon"
                onClick={onSettingsClick}
                className="transition-smooth hover:bg-primary/10 hover:text-primary"
              >
                <Settings className="w-5 h-5" />
              </Button>
            )}
            {user && (
              <Button 
                variant="ghost" 
                size="icon"
                onClick={signOut}
                className="transition-smooth hover:bg-destructive/10 hover:text-destructive"
              >
                <LogOut className="w-5 h-5" />
              </Button>
            )}
          </div>
        </div>
      </div>
    </motion.header>
  );
};
