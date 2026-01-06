import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Shield, Menu, X, Settings, LogOut, User, History } from 'lucide-react';
import { SettingsPanel } from '@/components/SettingsPanel';
import { ThemeToggle } from '@/components/ThemeToggle';
import { useAuth } from '@/hooks/useAuth';
import { supabase } from '@/integrations/supabase/client';

const navLinks = [
  { label: 'Features', href: '#features' },
  { label: 'How it Works', href: '#how-it-works' },
  { label: 'Pricing', href: '#pricing' },
  { label: 'About', href: '#about' },
];

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [profileName, setProfileName] = useState<string | null>(null);
  const navigate = useNavigate();
  const location = useLocation();
  const { user, loading, signOut } = useAuth();
  
  const isHomePage = location.pathname === '/';
  const isDashboard = location.pathname === '/dashboard';
  const isHistory = location.pathname === '/history';

  useEffect(() => {
    const fetchProfile = async () => {
      if (!user) {
        setProfileName(null);
        return;
      }
      
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

  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, href: string) => {
    e.preventDefault();
    
    if (!isHomePage) {
      navigate('/' + href);
      return;
    }
    
    const targetId = href.replace('#', '');
    const element = document.getElementById(targetId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    setIsOpen(false);
  };

  const handleSignOut = async () => {
    await signOut();
    navigate('/');
    setIsOpen(false);
  };

  return (
    <>
      <SettingsPanel isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
      
      <motion.nav
        className="fixed top-0 left-0 right-0 z-50 glass-strong border-b border-border/30"
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <motion.button
              className="flex items-center gap-2 cursor-pointer"
              onClick={() => navigate('/')}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Shield className="w-7 h-7 sm:w-8 sm:h-8 text-primary" />
              <span className="text-xl sm:text-2xl font-bold text-gradient">
                Aura Veracity
              </span>
            </motion.button>

            {/* Desktop navigation */}
            {isHomePage && (
              <div className="hidden lg:flex items-center gap-8">
                {navLinks.map((item) => (
                  <motion.a
                    key={item.label}
                    href={item.href}
                    onClick={(e) => handleNavClick(e, item.href)}
                    className="relative text-muted-foreground hover:text-foreground transition-smooth cursor-pointer"
                    whileHover={{ y: -2 }}
                  >
                    {item.label}
                    <motion.div
                      className="absolute -bottom-1 left-0 right-0 h-0.5 bg-primary origin-left"
                      initial={{ scaleX: 0 }}
                      whileHover={{ scaleX: 1 }}
                      transition={{ duration: 0.2 }}
                    />
                  </motion.a>
                ))}
              </div>
            )}

            {/* Desktop CTA */}
            <div className="hidden lg:flex items-center gap-3">
              <ThemeToggle />
              <Button 
                variant="ghost" 
                size="icon"
                onClick={() => setSettingsOpen(true)}
                className="text-muted-foreground hover:text-foreground"
              >
                <Settings className="w-5 h-5" />
              </Button>
              
              {loading ? (
                <div className="w-20 h-9 bg-muted/50 rounded animate-pulse" />
              ) : user ? (
                <>
                  <div className="flex items-center gap-2 text-sm text-muted-foreground px-3">
                    <User className="w-4 h-4" />
                    <span>{displayName}</span>
                  </div>
                  {(isDashboard || isHistory) && (
                    <Button 
                      variant="ghost"
                      size="sm"
                      className="gap-2"
                      onClick={() => navigate(isDashboard ? '/history' : '/dashboard')}
                    >
                      {isDashboard ? (
                        <>
                          <History className="w-4 h-4" />
                          <span className="hidden sm:inline">History</span>
                        </>
                      ) : (
                        <>Dashboard</>
                      )}
                    </Button>
                  )}
                  {!isDashboard && !isHistory && (
                    <Button 
                      variant="outline"
                      onClick={() => navigate('/dashboard')}
                    >
                      Dashboard
                    </Button>
                  )}
                  <Button 
                    variant="ghost"
                    size="icon"
                    onClick={handleSignOut}
                    className="text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                  >
                    <LogOut className="w-5 h-5" />
                  </Button>
                </>
              ) : (
                <>
                  <Button 
                    variant="ghost" 
                    onClick={() => navigate('/auth?mode=signin')}
                    className="text-muted-foreground hover:text-foreground"
                  >
                    Sign In
                  </Button>
                  <Button 
                    variant="default"
                    onClick={() => navigate('/auth?mode=signup')}
                  >
                    Get Started
                  </Button>
                </>
              )}
            </div>

            {/* Mobile menu button */}
            <div className="flex items-center gap-2 lg:hidden">
              <ThemeToggle />
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setSettingsOpen(true)}
              >
                <Settings className="w-5 h-5" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsOpen(!isOpen)}
                aria-label="Toggle menu"
              >
                {isOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </Button>
            </div>
          </div>

          {/* Mobile menu */}
          <AnimatePresence>
            {isOpen && (
              <motion.div
                className="lg:hidden mt-4 pt-4 border-t border-border/30"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.2 }}
              >
                <div className="flex flex-col gap-3">
                  {isHomePage && navLinks.map((item, index) => (
                    <motion.a
                      key={item.label}
                      href={item.href}
                      onClick={(e) => handleNavClick(e, item.href)}
                      className="text-muted-foreground hover:text-foreground transition-smooth py-2 cursor-pointer"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                    >
                      {item.label}
                    </motion.a>
                  ))}
                  <div className="flex flex-col gap-2 pt-4 border-t border-border/30">
                    {loading ? (
                      <div className="h-10 bg-muted/50 rounded animate-pulse" />
                    ) : user ? (
                      <>
                        <div className="flex items-center gap-2 text-sm text-muted-foreground py-2">
                          <User className="w-4 h-4" />
                          <span>{displayName}</span>
                        </div>
                        {(isDashboard || isHistory) && (
                          <Button 
                            variant="ghost"
                            className="justify-start gap-2"
                            onClick={() => {
                              navigate(isDashboard ? '/history' : '/dashboard');
                              setIsOpen(false);
                            }}
                          >
                            {isDashboard ? (
                              <>
                                <History className="w-4 h-4" />
                                History
                              </>
                            ) : (
                              <>Dashboard</>
                            )}
                          </Button>
                        )}
                        {!isDashboard && !isHistory && (
                          <Button 
                            variant="default"
                            className="justify-start"
                            onClick={() => {
                              navigate('/dashboard');
                              setIsOpen(false);
                            }}
                          >
                            Dashboard
                          </Button>
                        )}
                        <Button 
                          variant="ghost"
                          className="justify-start text-destructive hover:text-destructive hover:bg-destructive/10"
                          onClick={handleSignOut}
                        >
                          <LogOut className="w-4 h-4 mr-2" />
                          Sign Out
                        </Button>
                      </>
                    ) : (
                      <>
                        <Button 
                          variant="ghost" 
                          className="justify-start"
                          onClick={() => {
                            navigate('/auth?mode=signin');
                            setIsOpen(false);
                          }}
                        >
                          Sign In
                        </Button>
                        <Button 
                          variant="default"
                          className="justify-start"
                          onClick={() => {
                            navigate('/auth?mode=signup');
                            setIsOpen(false);
                          }}
                        >
                          Get Started
                        </Button>
                      </>
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.nav>
    </>
  );
};

export default Navbar;
