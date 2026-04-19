"use client";

import Link from "next/link";
import { 
  ArrowRight, Video, Target, TrendingUp, Sparkles, 
  Zap, MessageCircle, BarChart
} from "lucide-react";
import { useEffect, useState } from "react";

const Step = ({ number, title, desc, icon: Icon, delay }: { number: string, title: string, desc: string, icon: React.ElementType, delay: number }) => (
  <div className="flex flex-col items-center animate-fade-in-up" style={{ animationDelay: `${delay}ms` }}>
    <div className="relative mb-6">
      <div className="w-16 h-16 rounded-2xl bg-slate-800 border border-slate-700 flex items-center justify-center text-violet-400 group-hover:scale-110 transition-transform shadow-xl">
        <Icon size={28} />
      </div>
      <div className="absolute -top-3 -right-3 w-8 h-8 rounded-full bg-gradient-to-br from-violet-600 to-cyan-500 flex items-center justify-center text-white font-black text-sm border-4 border-slate-900 shadow-lg">
        {number}
      </div>
    </div>
    <h3 className="text-lg font-bold text-white mb-2">{title}</h3>
    <p className="text-slate-400 text-sm text-center leading-relaxed max-w-[200px]">{desc}</p>
  </div>
);

export default function Home() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <main className="min-h-screen relative overflow-hidden selection:bg-violet-500/30">
      
      {/* Navigation */}
      <nav className={`fixed top-0 w-full z-50 transition-all duration-300 ${scrolled ? 'bg-slate-950/80 backdrop-blur-lg border-b border-slate-800/50 py-4' : 'bg-transparent py-6'}`}>
        <div className="max-w-7xl mx-auto px-6 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-600 to-cyan-500 flex items-center justify-center text-white font-black italic shadow-lg shadow-violet-500/20">
              IQ
            </div>
            <span className="font-black text-xl tracking-tighter text-white">Interview<span className="text-violet-500">IQ</span></span>
          </div>
          <div className="hidden md:flex items-center gap-8 text-sm font-bold text-slate-400">
            <Link href="#how-it-works" className="hover:text-white transition-colors">Process</Link>
            <Link href="#" className="hover:text-white transition-colors">Models</Link>
            <Link href="#" className="hover:text-white transition-colors">Pricing</Link>
          </div>
          <Link href="/interview" className="btn-primary text-xs py-2.5 px-6">
            Get Started
          </Link>
        </div>
      </nav>
      
      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-6">
        {/* Simple background glow instead of particles */}
        <div className="absolute top-1/4 -left-20 w-96 h-96 bg-violet-600/10 rounded-full blur-[120px] pointer-events-none" />
        <div className="absolute bottom-1/4 -right-20 w-96 h-96 bg-cyan-600/10 rounded-full blur-[120px] pointer-events-none" />

        <div className="max-w-5xl mx-auto flex flex-col items-center text-center relative z-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-400 text-[10px] font-black uppercase tracking-widest mb-8 animate-fade-in-up">
            <Sparkles size={12} className="animate-pulse" /> Multimodal AI Engine 3.0
          </div>
          
          <h1 className="text-6xl md:text-8xl font-black mb-8 tracking-tighter text-white animate-fade-in-up" style={{ animationDelay: '100ms' }}>
            Master the art of <br/>
            <span className="text-gradient">Professional Influence.</span>
          </h1>
          
          <p className="text-xl text-slate-400 mb-12 max-w-2xl mx-auto leading-relaxed animate-fade-in-up" style={{ animationDelay: '200ms' }}>
            Go beyond practicing. Our AI analyzes your facial micro-expressions, speech patterns, and semantic logic to give you a clinical breakdown of your performance.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-6 justify-center animate-fade-in-up" style={{ animationDelay: '300ms' }}>
            <Link href="/interview" className="btn-primary text-lg px-10 group">
              Launch Interview Simulator 
              <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link href="/results/example" className="btn-secondary text-lg px-10">
              Live Demo Report
            </Link>
          </div>

          <div className="mt-20 w-full max-w-4xl relative animate-fade-in-up" style={{ animationDelay: '400ms' }}>
            <div className="glass-panel p-4 overflow-hidden shadow-2xl shadow-violet-500/10 border-gradient">
              <div className="aspect-video rounded-xl bg-slate-900 flex items-center justify-center relative group">
                <div className="absolute inset-0 bg-gradient-to-t from-slate-950 to-transparent opacity-60" />
                <div className="flex flex-col items-center relative z-10">
                  <div className="w-20 h-20 rounded-full bg-violet-600/20 border border-violet-500/30 flex items-center justify-center text-violet-400 group-hover:scale-110 transition-transform cursor-pointer">
                    <Video size={40} />
                  </div>
                  <p className="text-slate-400 text-sm font-bold mt-4 uppercase tracking-widest">Preview Interface</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Steps Section */}
      <section id="how-it-works" className="py-24 relative px-6 bg-slate-950/30">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-black text-white mb-4 tracking-tight">The 4-Step Process</h2>
            <p className="text-slate-400 font-medium">From raw input to scientific career insights.</p>
          </div>
          
          <div className="grid md:grid-cols-4 gap-12 relative">
            <div className="hidden md:block absolute top-8 left-[10%] right-[10%] h-px bg-gradient-to-r from-transparent via-slate-800 to-transparent" />
            
            <Step 
              number="1" 
              title="Record" 
              desc="Our simulator captures high-fidelity audio and video feed." 
              icon={Video} 
              delay={100} 
            />
            <Step 
              number="2" 
              title="Analyze" 
              desc="Neural models process pitch, cadence, and facial cues." 
              icon={Zap} 
              delay={200} 
            />
            <Step 
              number="3" 
              title="Index" 
              desc="Text is mapped against semantic benchmark libraries." 
              icon={Target} 
              delay={300} 
            />
            <Step 
              number="4" 
              title="Report" 
              desc="Get a clinical breakdown with actionable steps." 
              icon={BarChart} 
              delay={400} 
            />
          </div>
        </div>
      </section>

      {/* Feature Grid */}
      <section className="py-24 px-6 max-w-7xl mx-auto">
        <div className="grid md:grid-cols-3 gap-8">
          <div className="glass-panel p-8 glass-panel-hover flex flex-col group">
            <div className="w-14 h-14 rounded-2xl bg-violet-500/10 flex items-center justify-center text-violet-400 mb-6 border border-violet-500/20 group-hover:bg-violet-500 group-hover:text-white transition-all">
              <MessageCircle size={28} />
            </div>
            <h3 className="text-xl font-bold text-white mb-4">Semantic Intelligence</h3>
            <p className="text-slate-400 leading-relaxed text-sm">
              We don&apos;t just transcribe. We analyze the logic, coherence, and technical accuracy of your answers using specialized LLM scoring.
            </p>
          </div>
          
          <div className="glass-panel p-8 glass-panel-hover flex flex-col group">
            <div className="w-14 h-14 rounded-2xl bg-cyan-500/10 flex items-center justify-center text-cyan-400 mb-6 border border-cyan-500/20 group-hover:bg-cyan-500 group-hover:text-white transition-all">
              <TrendingUp size={28} />
            </div>
            <h3 className="text-xl font-bold text-white mb-4">Biometric Analysis</h3>
            <p className="text-slate-400 leading-relaxed text-sm">
              Computer vision models track eye contact, posture stability, and emotion frequency to determine your perceived confidence.
            </p>
          </div>

          <div className="glass-panel p-8 glass-panel-hover flex flex-col group">
            <div className="w-14 h-14 rounded-2xl bg-pink-500/10 flex items-center justify-center text-pink-400 mb-6 border border-pink-500/20 group-hover:bg-pink-500 group-hover:text-white transition-all">
              <Target size={28} />
            </div>
            <h3 className="text-xl font-bold text-white mb-4">Granular Metrics</h3>
            <p className="text-slate-400 leading-relaxed text-sm">
              From filler word frequency (um, like, so) to speaking rate (WPM), we provide the hard data needed to polish your delivery.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-slate-900 text-center">
        <p className="text-slate-600 text-sm font-bold uppercase tracking-widest">
          © 2026 INTERVIEW<span className="text-slate-500">IQ</span> ENGINE
        </p>
      </footer>
    </main>
  );
}
