import { motion } from 'framer-motion';
import { Shield, Zap, Eye, Brain, Lock, BarChart3 } from 'lucide-react';

const features = [
  {
    icon: Shield,
    title: 'AI-Powered Detection',
    description: 'Advanced machine learning models trained on diverse datasets to identify manipulated and synthetic media.',
  },
  {
    icon: Brain,
    title: 'Multimodal Analysis',
    description: 'Combines visual and audio analysis for comprehensive detection of AI-generated content.',
  },
  {
    icon: Zap,
    title: 'Fast Processing',
    description: 'Optimized processing pipeline delivers analysis results quickly and efficiently.',
  },
  {
    icon: Eye,
    title: 'Frame-level Analysis',
    description: 'Detailed frame-by-frame examination to identify potential manipulation artifacts.',
  },
  {
    icon: Lock,
    title: 'Privacy First',
    description: 'Your uploads are encrypted and handled securely. We prioritize your data privacy.',
  },
  {
    icon: BarChart3,
    title: 'Detailed Reports',
    description: 'Export comprehensive analysis reports suitable for documentation and review.',
  },
];

const Features = () => {
  return (
    <section id="features" className="py-24 px-4 sm:px-6 relative">
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-primary/5 to-transparent" />
      
      <div className="max-w-7xl mx-auto relative">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Powerful Features for <span className="text-gradient">Digital Truth</span>
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Built for India's digital ecosystem — protecting citizens, journalists, and organizations from synthetic media threats.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              className="glass-strong p-6 rounded-2xl border border-border/30 hover:border-primary/50 transition-all duration-300 group"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              whileHover={{ y: -5 }}
            >
              <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                <feature.icon className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">{feature.title}</h3>
              <p className="text-muted-foreground text-sm">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
