import { motion } from 'framer-motion';
import { Shield, Target, Lightbulb } from 'lucide-react';

const About = () => {
  return (
    <section id="about" className="py-24 px-4 sm:px-6 relative bg-muted/30">
      <div className="max-w-7xl mx-auto">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          {/* Left content */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-6">
              About <span className="text-gradient">Aura Veracity</span>
            </h2>
            
            <div className="space-y-4 text-muted-foreground">
              <p>
                Aura Veracity is a deepfake detection platform built to help individuals and 
                organizations verify the authenticity of video content in an age of increasingly 
                sophisticated AI-generated media.
              </p>
              <p>
                Our platform combines visual and audio analysis techniques to identify potential 
                signs of manipulation, providing users with detailed reports to make informed 
                decisions about content authenticity.
              </p>
              <p>
                We are committed to continuously improving our detection capabilities as 
                synthetic media technology evolves, ensuring our users have access to 
                effective verification tools.
              </p>
            </div>

            {/* Key points */}
            <div className="mt-8 space-y-4">
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <Shield className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <h4 className="font-semibold text-foreground">Privacy Focused</h4>
                  <p className="text-sm text-muted-foreground">
                    Your uploaded content is handled securely and with respect for your privacy.
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <Target className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <h4 className="font-semibold text-foreground">Multimodal Approach</h4>
                  <p className="text-sm text-muted-foreground">
                    Analyzes both visual frames and audio signals for comprehensive detection.
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <Lightbulb className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <h4 className="font-semibold text-foreground">Continuous Improvement</h4>
                  <p className="text-sm text-muted-foreground">
                    Regularly updating our models to keep pace with evolving deepfake techniques.
                  </p>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Right - Vision card */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            {/* Vision card */}
            <div className="glass-strong p-8 rounded-2xl border border-border/30">
              <h3 className="text-2xl font-bold text-foreground mb-4">Our Mission</h3>
              <p className="text-muted-foreground mb-6">
                To provide accessible and reliable deepfake detection tools that help users 
                navigate the challenges of synthetic media in today's digital landscape.
              </p>
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-primary" />
                  <span className="text-foreground">Accessible detection tools</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-primary" />
                  <span className="text-foreground">Transparent analysis results</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-primary" />
                  <span className="text-foreground">User-friendly experience</span>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default About;
