import { motion } from 'framer-motion';
import { Upload, Cpu, FileCheck, Download } from 'lucide-react';

const steps = [
  {
    icon: Upload,
    step: '01',
    title: 'Upload Your Video',
    description: 'Drag and drop or select a video file. We support common video formats.',
  },
  {
    icon: Cpu,
    step: '02',
    title: 'AI Analysis',
    description: 'Our multimodal AI analyzes visual frames and audio patterns to detect potential manipulation.',
  },
  {
    icon: FileCheck,
    step: '03',
    title: 'View Results',
    description: 'Review confidence scores, detected anomalies, and detailed analysis breakdown.',
  },
  {
    icon: Download,
    step: '04',
    title: 'Export Report',
    description: 'Download analysis reports for your records and further review.',
  },
];

const HowItWorks = () => {
  return (
    <section id="how-it-works" className="py-24 px-4 sm:px-6 relative bg-muted/30">
      <div className="max-w-7xl mx-auto">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            How <span className="text-gradient">Aura Veracity</span> Works
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Simple, fast, and accurate deepfake detection in four easy steps.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {steps.map((item, index) => (
            <motion.div
              key={item.step}
              className="relative"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.15 }}
            >
              {/* Connector line */}
              {index < steps.length - 1 && (
                <div className="hidden lg:block absolute top-12 left-full w-full h-0.5 bg-gradient-to-r from-primary/50 to-transparent z-0" />
              )}
              
              <div className="glass-strong p-6 rounded-2xl border border-border/30 relative z-10 h-full">
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
                    <item.icon className="w-6 h-6 text-primary" />
                  </div>
                  <span className="text-3xl font-bold text-primary/30">{item.step}</span>
                </div>
                <h3 className="text-lg font-semibold text-foreground mb-2">{item.title}</h3>
                <p className="text-muted-foreground text-sm">{item.description}</p>
              </div>
            </motion.div>
          ))}
        </div>

      </div>
    </section>
  );
};

export default HowItWorks;
