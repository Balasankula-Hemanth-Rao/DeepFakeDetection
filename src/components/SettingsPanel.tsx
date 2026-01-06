import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Moon, Sun, Film, Sparkles, User, Lock, Clock, Check, LogIn, Mail, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { InputOTP, InputOTPGroup, InputOTPSlot } from '@/components/ui/input-otp';
import { useTheme, Theme } from '@/contexts/ThemeContext';
import { useAuth } from '@/hooks/useAuth';
import { useNavigate } from 'react-router-dom';
import { toast } from '@/hooks/use-toast';
import { supabase } from '@/integrations/supabase/client';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

const themeOptions: { value: Theme; label: string; icon: React.ReactNode; description: string }[] = [
  { 
    value: 'dark', 
    label: 'Dark Navy', 
    icon: <Moon className="w-5 h-5" />,
    description: 'Deep ocean vibes with electric accents'
  },
  { 
    value: 'light', 
    label: 'Light Mode', 
    icon: <Sun className="w-5 h-5" />,
    description: 'Clean and bright professional look'
  },
  { 
    value: 'cinematic', 
    label: 'Cinematic', 
    icon: <Film className="w-5 h-5" />,
    description: 'Rich contrast for video professionals'
  },
  { 
    value: 'neon', 
    label: 'Neon Pulse', 
    icon: <Sparkles className="w-5 h-5" />,
    description: 'Vibrant cyberpunk aesthetics'
  },
];

// Future enhancement: expand to full result logs and downloadable reports
const mockRecentUploads = [
  { id: '1', name: 'interview_footage.mp4', confidence: 91, date: '2 hours ago' },
  { id: '2', name: 'news_segment.mp4', confidence: 8, date: '1 day ago' },
  { id: '3', name: 'social_media_clip.mp4', confidence: 87, date: '3 days ago' },
];

export const SettingsPanel = ({ isOpen, onClose }: SettingsPanelProps) => {
  const { theme, setTheme } = useTheme();
  const { user, signInWithOTP, verifyOTP, signOut } = useAuth();
  const navigate = useNavigate();
  
  // Login form state
  const [email, setEmail] = useState('');
  const [otp, setOtp] = useState('');
  const [otpSent, setOtpSent] = useState(false);
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  const handleDeleteAccount = async () => {
    setIsDeleting(true);
    try {
      // Get the current session to pass authorization
      const { data: { session }, error: sessionError } = await supabase.auth.getSession();
      
      if (sessionError || !session) {
        throw new Error('No active session. Please sign in again.');
      }

      const { data, error } = await supabase.functions.invoke('delete-account', {
        headers: {
          'Authorization': `Bearer ${session.access_token}`
        }
      });
      
      if (error) {
        console.error('Delete account error:', error);
        toast({
          variant: "destructive",
          title: "Failed to delete account",
          description: error.message || "An error occurred while deleting your account.",
        });
        setIsDeleting(false);
        return;
      }

      // Sign out the user before redirecting
      await signOut();

      toast({
        title: "Account deleted",
        description: "Your account and all associated data have been permanently removed.",
      });
      
      onClose();
      setTimeout(() => navigate('/'), 1000);
    } catch (error: any) {
      console.error('Delete account error:', error);
      toast({
        variant: "destructive",
        title: "Failed to delete account",
        description: error.message || "An error occurred while deleting your account.",
      });
      setIsDeleting(false);
    }
  };

  const handleSendOTP = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) {
      toast({
        variant: "destructive",
        title: "Missing email",
        description: "Please enter your email address.",
      });
      return;
    }

    setIsLoggingIn(true);
    const { error } = await signInWithOTP(email);
    setIsLoggingIn(false);

    if (!error) {
      setOtpSent(true);
    }
  };

  const handleVerifyOTP = async () => {
    if (otp.length !== 6) return;

    setIsLoggingIn(true);
    const { error } = await verifyOTP(email, otp);
    setIsLoggingIn(false);

    if (!error) {
      setEmail('');
      setOtp('');
      setOtpSent(false);
    }
  };

  const handleBackToEmail = () => {
    setOtpSent(false);
    setOtp('');
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40"
            onClick={onClose}
          />

          {/* Panel */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="fixed right-0 top-0 h-full w-full max-w-md bg-card border-l border-border shadow-2xl z-50 overflow-y-auto"
          >
            <div className="p-6 space-y-6">
              {/* Header */}
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                  Settings
                </h2>
                <Button variant="ghost" size="icon" onClick={onClose}>
                  <X className="w-5 h-5" />
                </Button>
              </div>

              {/* Theme Selector - Always visible */}
              <Card className="glass transition-smooth">
                <CardHeader>
                  <CardTitle className="text-lg">Appearance</CardTitle>
                  <CardDescription>Choose your preferred theme</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {themeOptions.map((option) => (
                    <motion.button
                      key={option.value}
                      onClick={() => setTheme(option.value)}
                      whileHover={{ scale: 1.01 }}
                      whileTap={{ scale: 0.99 }}
                      className={`w-full p-4 rounded-lg border transition-all text-left ${
                        theme === option.value
                          ? 'border-primary bg-primary/10'
                          : 'border-border/50 hover:border-primary/30 hover:bg-muted/30'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <div className={theme === option.value ? 'text-primary' : 'text-muted-foreground'}>
                          {option.icon}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="font-medium text-sm">{option.label}</div>
                          <div className="text-xs text-muted-foreground truncate">{option.description}</div>
                        </div>
                        {theme === option.value && (
                          <Check className="w-4 h-4 text-primary flex-shrink-0" />
                        )}
                      </div>
                    </motion.button>
                  ))}
                </CardContent>
              </Card>

              {/* Login Section - Show when not logged in */}
              {!user ? (
                <Card className="glass transition-smooth">
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center">
                      <LogIn className="w-5 h-5 mr-2" />
                      Sign In
                    </CardTitle>
                    <CardDescription>
                      {otpSent ? `Enter the code sent to ${email}` : 'Sign in with your email'}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {!otpSent ? (
                      <form onSubmit={handleSendOTP} className="space-y-4">
                        <div className="space-y-2">
                          <Label htmlFor="login-email">Email</Label>
                          <div className="relative">
                            <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                            <Input 
                              id="login-email" 
                              type="email" 
                              placeholder="your@email.com"
                              value={email}
                              onChange={(e) => setEmail(e.target.value)}
                              className="bg-background/50 pl-10" 
                            />
                          </div>
                        </div>
                        <Button type="submit" className="w-full" disabled={isLoggingIn}>
                          {isLoggingIn ? 'Sending code...' : 'Send verification code'}
                        </Button>
                      </form>
                    ) : (
                      <div className="space-y-4">
                        <div className="flex justify-center">
                          <InputOTP
                            value={otp}
                            onChange={setOtp}
                            maxLength={6}
                            onComplete={handleVerifyOTP}
                          >
                            <InputOTPGroup>
                              <InputOTPSlot index={0} />
                              <InputOTPSlot index={1} />
                              <InputOTPSlot index={2} />
                              <InputOTPSlot index={3} />
                              <InputOTPSlot index={4} />
                              <InputOTPSlot index={5} />
                            </InputOTPGroup>
                          </InputOTP>
                        </div>
                        <Button 
                          className="w-full" 
                          disabled={isLoggingIn || otp.length !== 6}
                          onClick={handleVerifyOTP}
                        >
                          {isLoggingIn ? 'Verifying...' : 'Verify & Sign In'}
                        </Button>
                        <div className="flex flex-col items-center gap-2 text-sm">
                          <button
                            type="button"
                            onClick={handleSendOTP}
                            disabled={isLoggingIn}
                            className="text-primary hover:underline disabled:opacity-50"
                          >
                            Resend code
                          </button>
                          <button
                            type="button"
                            onClick={handleBackToEmail}
                            className="text-muted-foreground hover:text-foreground"
                          >
                            Use a different email
                          </button>
                        </div>
                      </div>
                    )}
                    <Separator />
                    <div className="text-center">
                      <p className="text-sm text-muted-foreground mb-2">Need a full sign up flow?</p>
                      <Button 
                        variant="outline" 
                        className="w-full"
                        onClick={() => {
                          onClose();
                          navigate('/auth');
                        }}
                      >
                        Go to Auth Page
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <>
                  {/* Profile Section - Show when logged in */}
                  <Card className="glass transition-smooth">
                    <CardHeader>
                      <CardTitle className="text-lg flex items-center">
                        <User className="w-5 h-5 mr-2" />
                        Profile
                      </CardTitle>
                      <CardDescription>Manage your account details</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="name">Display Name</Label>
                        <Input id="name" placeholder="Your name" className="bg-background/50" />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="email">Email</Label>
                        <Input 
                          id="email" 
                          type="email" 
                          value={user?.email || ''} 
                          className="bg-background/50" 
                          disabled 
                        />
                      </div>
                      <Separator />
                      <Button 
                        variant="destructive" 
                        className="w-full"
                        onClick={signOut}
                      >
                        Sign Out
                      </Button>
                      <Separator />
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button 
                            variant="outline" 
                            className="w-full border-destructive/50 text-destructive hover:bg-destructive/10 hover:text-destructive"
                          >
                            <Trash2 className="w-4 h-4 mr-2" />
                            Delete Account
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
                            <AlertDialogDescription>
                              This action cannot be undone. This will permanently delete your account, 
                              all your uploaded videos, detection results, and remove all your data from our servers.
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>Cancel</AlertDialogCancel>
                            <AlertDialogAction
                              onClick={handleDeleteAccount}
                              disabled={isDeleting}
                              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                            >
                              {isDeleting ? 'Deleting...' : 'Delete Account'}
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </CardContent>
                  </Card>

                  {/* Recent Activity - Show when logged in */}
                  <Card className="glass transition-smooth">
                    <CardHeader>
                      <CardTitle className="text-lg flex items-center">
                        <Clock className="w-5 h-5 mr-2" />
                        Recent Uploads
                      </CardTitle>
                      <CardDescription>Your latest video analyses</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      {mockRecentUploads.map((upload) => (
                        <motion.div
                          key={upload.id}
                          whileHover={{ x: 4 }}
                          className="p-3 rounded-lg bg-muted/50 hover:bg-muted cursor-pointer transition-smooth"
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-medium truncate">{upload.name}</p>
                              <p className="text-xs text-muted-foreground">{upload.date}</p>
                            </div>
                            <div className={`ml-3 text-sm font-bold ${
                              upload.confidence > 50 ? 'text-success' : 'text-destructive'
                            }`}>
                              {upload.confidence}%
                            </div>
                          </div>
                        </motion.div>
                      ))}
                      <p className="text-xs text-muted-foreground text-center pt-2">
                        {/* Future enhancement: expand to full result logs and downloadable reports */}
                        Full history and detailed logs coming soon
                      </p>
                    </CardContent>
                  </Card>
                </>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
