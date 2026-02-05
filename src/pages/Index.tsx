import { lazy, Suspense } from "react";
import Navbar from "@/components/landing/Navbar";
import Hero from "@/components/landing/Hero";
import Footer from "@/components/landing/Footer";

// Lazy load below-the-fold sections for better performance
const Features = lazy(() => import("@/components/landing/Features"));
const HowItWorks = lazy(() => import("@/components/landing/HowItWorks"));
const Pricing = lazy(() => import("@/components/landing/Pricing"));
const About = lazy(() => import("@/components/landing/About"));

const Index = () => {
  return (
    <div className="min-h-screen bg-background scroll-smooth">
      <Navbar />
      <section id="hero">
        <Hero />
      </section>
      <Suspense fallback={<div className="min-h-screen" />}>
        <section id="features">
          <Features />
        </section>
        <section id="how-it-works">
          <HowItWorks />
        </section>
        <section id="pricing">
          <Pricing />
        </section>
        <section id="about">
          <About />
        </section>
      </Suspense>
      <Footer />
    </div>
  );
};

export default Index;
