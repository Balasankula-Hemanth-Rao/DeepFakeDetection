import { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Shield, Brain, Zap, LayoutDashboard } from 'lucide-react';
import heroImage from '@/assets/hero-neural-bg.jpg';
import { useAuth } from '@/hooks/useAuth';

const Hero = () => {
  const navigate = useNavigate();
  const { user, loading } = useAuth();

  // Memoize floating particles
  const particles = useMemo(() =>
    [...Array(8)].map((_, i) => ({
      id: i,
      left: Math.random() * 100,
      top: Math.random() * 100,
      duration: 3 + Math.random() * 2,
      delay: Math.random() * 2,
    })),
    []);

  const features = [
    {
      icon: Brain,
      title: 'Multimodal Analysis',
      description: 'Analyzes both video frames and audio signals for comprehensive detection'
    },
    {
      icon: Shield,
      title: 'Advanced Detection',
      description: 'Deep learning models trained to identify synthetic media artifacts'
    },
    {
      icon: Zap,
      title: 'Quick Processing',
      description: 'Upload your video and receive analysis results efficiently'
    }
  ];

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0">
        <img
          src={heroImage}
          alt=""
          className="w-full h-full object-cover opacity-20"
          loading="lazy"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-background/40 via-background/80 to-background" />
      </div>

      {/* Floating particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {particles.map((particle) => (
          <motion.div
            key={particle.id}
            className="absolute w-2 h-2 bg-primary rounded-full opacity-40"
            style={{ left: `${particle.left}%`, top: `${particle.top}%` }}
            animate={{ y: [-20, 20, -20], opacity: [0.2, 0.6, 0.2] }}
            transition={{
              duration: particle.duration,
              repeat: Infinity,
              delay: particle.delay,
              ease: 'easeInOut',
            }}
          />
        ))}
      </div>

      {/* Main content */}
      <div className="relative z-10 max-w-6xl mx-auto px-6 text-center pt-24 pb-16">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="mb-10"
        >
          {/* Badge */}
          <motion.div
            className="inline-flex items-center gap-2 glass px-4 py-2 rounded-full mb-8"
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
          >
            <Shield className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium text-muted-foreground">
              Advanced AI Detection Technology
            </span>
          </motion.div>

          {/* Headline */}
          <h1 className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-bold leading-tight mb-8">
            <span className="text-foreground">Separate</span>
            <br />
            <span className="text-gradient">Truth</span>
            {' '}
            <span className="text-foreground">from</span>
            <br />
            <span className="bg-gradient-secondary bg-clip-text text-transparent">
              Deception
            </span>
          </h1>

          {/* Subheadline */}
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="text-lg sm:text-xl md:text-2xl text-muted-foreground max-w-3xl mx-auto leading-relaxed"
          >
            Aura Veracity uses multimodal AI to analyze videos for potential deepfake manipulation.
            Upload a video and get detailed authenticity analysis.
          </motion.p>
        </motion.div>

        {/* CTA Buttons */}
        <motion.div
          className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.6 }}
        >
          {loading ? (
            <div className="w-48 h-14 bg-muted/50 rounded-lg animate-pulse" />
          ) : user ? (
            <Button
              variant="hero"
              size="lg"
              className="text-lg px-8 py-6 min-w-[180px]"
              onClick={() => navigate('/dashboard')}
            >
              <Brain className="w-5 h-5 mr-2" />
              Start Analysis
            </Button>
          ) : (
            <>
              <Button
                variant="hero"
                size="lg"
                className="text-lg px-8 py-6 min-w-[180px]"
                onClick={() => navigate('/auth?mode=signup')}
              >
                <Brain className="w-5 h-5 mr-2" />
                Get Started
              </Button>
              <Button
                variant="glass"
                size="lg"
                className="text-lg px-8 py-6 min-w-[180px]"
                onClick={() => navigate('/auth?mode=signin')}
              >
                Sign In
              </Button>
            </>
          )}
        </motion.div>

        {/* Feature cards */}
        <motion.div
          className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 max-w-4xl mx-auto"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8, duration: 0.8 }}
        >
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              className="glass-strong p-6 rounded-xl transition-all duration-300 hover:scale-[1.02] hover:shadow-glow-primary"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1 + index * 0.15, duration: 0.5 }}
            >
              <feature.icon className="w-8 h-8 text-primary mb-4 mx-auto" />
              <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </motion.div>
      </div>

      {/* Bottom gradient */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent pointer-events-none" />
    </section>
  );
};

export default Hero;
