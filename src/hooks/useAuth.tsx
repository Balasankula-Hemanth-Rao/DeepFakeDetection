import { useState, useEffect, createContext, useContext, ReactNode } from 'react';
import { User, Session } from '@supabase/supabase-js';
import { supabase } from '@/integrations/supabase/client';
import { toast } from '@/hooks/use-toast';

interface AuthContextType {
  user: User | null;
  session: Session | null;
  loading: boolean;
  signUp: (email: string, password: string, metadata?: any) => Promise<{ error: any }>;
  signInWithPassword: (email: string, password: string) => Promise<{ error: any }>;
  signInWithOTP: (email: string, metadata?: any) => Promise<{ error: any }>;
  verifyOTP: (email: string, token: string) => Promise<{ error: any }>;
  signInWithGoogle: () => Promise<{ error: any }>;
  signOut: () => Promise<void>;
  verifyOtp: (email: string, token: string, type: 'email' | 'sms') => Promise<{ error: any }>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider = ({ children }: AuthProviderProps) => {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Set up auth state listener FIRST
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (event, session) => {
        setSession(session);
        setUser(session?.user ?? null);
        setLoading(false);
        
        if (event === 'SIGNED_IN') {
          toast({
            title: "Welcome!",
            description: "You have successfully signed in.",
          });
        } else if (event === 'SIGNED_OUT') {
          toast({
            title: "Signed out",
            description: "You have been signed out successfully.",
          });
        }
      }
    );

    // THEN check for existing session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setUser(session?.user ?? null);
      setLoading(false);
    });

    return () => subscription.unsubscribe();
  }, []);

  const signUp = async (email: string, password: string, metadata?: any) => {
    try {
      const { error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          emailRedirectTo: `${window.location.origin}/dashboard`,
          data: metadata
        }
      });
      
      if (error) {
        toast({
          variant: "destructive",
          title: "Sign up failed",
          description: error.message,
        });
      } else {
        toast({
          title: "Account created!",
          description: "Please check your email to confirm your account.",
        });
      }
      
      return { error };
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "Sign up failed",
        description: error.message,
      });
      return { error };
    }
  };

  const signInWithPassword = async (email: string, password: string) => {
    try {
      const { error } = await supabase.auth.signInWithPassword({
        email,
        password
      });
      
      if (error) {
        toast({
          variant: "destructive",
          title: "Sign in failed",
          description: error.message,
        });
      }
      
      return { error };
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "Sign in failed",
        description: error.message,
      });
      return { error };
    }
  };

  const signInWithOTP = async (email: string, metadata?: any) => {
    try {
      const { error } = await supabase.auth.signInWithOtp({
        email,
        options: {
          emailRedirectTo: `${window.location.origin}/dashboard`,
          data: metadata
        }
      });
      
      if (error) {
        toast({
          variant: "destructive",
          title: "Failed to send code",
          description: error.message,
        });
      } else {
        toast({
          title: "Check your email",
          description: "We've sent you a one-time password (OTP). Please enter it to verify your account.",
        });
      }
      
      return { error };
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "Failed to send code",
        description: error.message,
      });
      return { error };
    }
  };

  const verifyOTP = async (email: string, token: string) => {
    try {
      const { error } = await supabase.auth.verifyOtp({
        email,
        token,
        type: 'email'
      });
      
      if (error) {
        toast({
          variant: "destructive",
          title: "Verification failed",
          description: error.message,
        });
      }
      
      return { error };
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "Verification failed",
        description: error.message,
      });
      return { error };
    }
  };

  const signInWithGoogle = async () => {
    try {
      const { error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: `${window.location.origin}/dashboard`,
        },
      });
      
      if (error) {
        toast({
          variant: "destructive",
          title: "Google sign in failed",
          description: error.message,
        });
      }
      
      return { error };
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "Google sign in failed",
        description: error.message,
      });
      return { error };
    }
  };

  const signOut = async () => {
    try {
      const { error } = await supabase.auth.signOut();
      if (error) {
        toast({
          variant: "destructive",
          title: "Sign out failed",
          description: error.message,
        });
      }
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "Sign out failed",
        description: error.message,
      });
    }
  };

  const verifyOtp = async (email: string, token: string) => {
    try {
      const { error, data } = await supabase.auth.verifyOtp({
        email,
        token,
        type: 'email' as const,
      });

      if (error) {
        toast({
          variant: "destructive",
          title: "OTP verification failed",
          description: error.message,
        });
      } else {
        toast({
          title: "Email verified",
          description: "Your email has been successfully verified!",
        });
      }

      return { error, data };
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "OTP verification failed",
        description: error.message,
      });
      return { error };
    }
  };

  const value = {
    user,
    session,
    loading,
    signUp,
    signInWithPassword,
    signInWithOTP,
    verifyOTP,
    signInWithGoogle,
    signOut,
    verifyOtp,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
