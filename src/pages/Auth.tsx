import { useState, useEffect, useMemo } from 'react';
import { Navigate, useSearchParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { InputOTP, InputOTPGroup, InputOTPSlot } from '@/components/ui/input-otp';
import { useAuth } from '@/hooks/useAuth';
import { LoadingState } from '@/components/ui/loading-state';
import { Shield, Zap, Brain, ArrowLeft, Mail, CheckCircle } from 'lucide-react';

const GoogleIcon = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24">
    <path
      fill="currentColor"
      d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
    />
    <path
      fill="currentColor"
      d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
    />
    <path
      fill="currentColor"
      d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
    />
    <path
      fill="currentColor"
      d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
    />
  </svg>
);

const Auth = () => {
  const { user, signInWithOTP, verifyOTP, signInWithGoogle, loading } = useAuth();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);
  const [email, setEmail] = useState('');
  const [otpSent, setOtpSent] = useState(false);
  const [otp, setOtp] = useState('');
  const [fullName, setFullName] = useState('');

  // Generate particles only once
  const particles = useMemo(() => 
    [...Array(15)].map((_, i) => ({
      id: i,
      initialX: Math.random() * 100,
      initialY: Math.random() * 100,
      targetX: Math.random() * 100,
      targetY: Math.random() * 100,
      duration: Math.random() * 20 + 15,
      size: Math.random() * 2 + 1,
    })), 
  []);

  if (user) {
    return <Navigate to="/dashboard" replace />;
  }

  const handleGoogleSignIn = async () => {
    setIsGoogleLoading(true);
    await signInWithGoogle();
  };

  const handleSendOTP = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!email) return;
    
    setIsSubmitting(true);
    const { error } = await signInWithOTP(email, { full_name: fullName });
    
    if (!error) {
      setOtpSent(true);
    }
    setIsSubmitting(false);
  };

  const handleVerifyOTP = async () => {
    if (otp.length !== 6) return;
    
    setIsSubmitting(true);
    await verifyOTP(email, otp);
    setIsSubmitting(false);
  };

  const handleResendOTP = async () => {
    setIsSubmitting(true);
    await signInWithOTP(email, { full_name: fullName });
    setIsSubmitting(false);
  };

  const handleBackToEmail = () => {
    setOtpSent(false);
    setOtp('');
  };

  if (loading) {
    return <LoadingState message="Checking authentication..." fullScreen />;
  }

  const features = [
    { icon: Brain, label: 'AI-Powered' },
    { icon: Zap, label: 'Instant Results' },
    { icon: Shield, label: '99.7% Accurate' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5 flex items-center justify-center p-4 relative overflow-hidden">
      {/* Animated background particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {particles.map((particle) => (
          <motion.div
            key={particle.id}
            className="absolute rounded-full bg-primary/20"
            style={{
              width: particle.size,
              height: particle.size,
              left: `${particle.initialX}%`,
              top: `${particle.initialY}%`,
            }}
            animate={{
              x: [0, (particle.targetX - particle.initialX) * 10],
              y: [0, (particle.targetY - particle.initialY) * 10],
            }}
            transition={{
              duration: particle.duration,
              repeat: Infinity,
              repeatType: 'reverse',
              ease: 'linear',
            }}
          />
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md relative z-10"
      >
        {/* Back button */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-6"
        >
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={() => navigate('/')}
            className="gap-2 text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Home
          </Button>
        </motion.div>

        {/* Logo and title */}
        <div className="mb-8 text-center">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
            className="inline-flex items-center justify-center w-16 h-16 bg-primary/10 backdrop-blur-sm rounded-2xl border border-primary/20 mb-4"
          >
            <Shield className="w-8 h-8 text-primary" />
          </motion.div>
          <h1 className="text-3xl font-bold text-gradient">
            Aura Veracity
          </h1>
          <p className="text-muted-foreground mt-2">
            Advanced AI-powered deepfake detection
          </p>
        </div>

        {/* Auth card */}
        <Card className="glass-strong border-border/50">
          <CardHeader className="space-y-1 pb-4">
            <CardTitle className="text-2xl font-bold text-center">
              {otpSent ? 'Enter verification code' : 'Welcome'}
            </CardTitle>
            <CardDescription className="text-center">
              {otpSent 
                ? `We sent a 6-digit code to ${email}` 
                : 'Sign in or create an account with your email'
              }
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {!otpSent ? (
              <>
                {/* Google Sign In Button */}
                <Button
                  type="button"
                  variant="outline"
                  className="w-full gap-3 h-11"
                  onClick={handleGoogleSignIn}
                  disabled={isGoogleLoading}
                >
                  <GoogleIcon />
                  {isGoogleLoading ? 'Connecting...' : 'Continue with Google'}
                </Button>
                
                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <Separator className="w-full" />
                  </div>
                  <div className="relative flex justify-center text-xs uppercase">
                    <span className="bg-card px-2 text-muted-foreground">or</span>
                  </div>
                </div>

                <form onSubmit={handleSendOTP} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="fullName">Full Name</Label>
                    <Input
                      id="fullName"
                      type="text"
                      placeholder="John Doe"
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                      autoComplete="name"
                      className="bg-background/50"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="email">Email</Label>
                    <Input
                      id="email"
                      type="email"
                      placeholder="you@example.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                      autoComplete="email"
                      className="bg-background/50"
                    />
                  </div>
                  <Button type="submit" className="w-full gap-2" disabled={isSubmitting || !email}>
                    <Mail className="w-4 h-4" />
                    {isSubmitting ? 'Sending code...' : 'Send verification code'}
                  </Button>
                </form>
              </>
            ) : (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                <div className="flex justify-center">
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: 'spring', stiffness: 200 }}
                    className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center"
                  >
                    <CheckCircle className="w-8 h-8 text-primary" />
                  </motion.div>
                </div>

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
                  disabled={isSubmitting || otp.length !== 6}
                  onClick={handleVerifyOTP}
                >
                  {isSubmitting ? 'Verifying...' : 'Verify & Sign In'}
                </Button>

                <div className="flex flex-col items-center gap-2 text-sm">
                  <button
                    type="button"
                    onClick={handleResendOTP}
                    disabled={isSubmitting}
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
              </motion.div>
            )}
          </CardContent>
        </Card>

        {/* Features */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="mt-8 grid grid-cols-3 gap-4"
        >
          {features.map((feature, index) => (
            <motion.div 
              key={feature.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 + index * 0.1 }}
              className="text-center"
            >
              <div className="inline-flex items-center justify-center w-10 h-10 bg-primary/10 rounded-lg mb-2">
                <feature.icon className="w-5 h-5 text-primary" />
              </div>
              <p className="text-xs text-muted-foreground">{feature.label}</p>
            </motion.div>
          ))}
        </motion.div>
      </motion.div>
    </div>
  );
};

export default Auth;
