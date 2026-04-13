import Link from "next/link";
import { ArrowRight, Video, Target, TrendingUp } from "lucide-react";

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-8 text-center relative overflow-hidden">
      
      {/* Hero Section */}
      <div className="max-w-4xl glass-panel p-12 relative z-10 glass-panel-hover">
        <h1 className="text-6xl font-extrabold mb-6 tracking-tight">
          Master Your Next <br/>
          <span className="text-gradient">Interview</span>
        </h1>
        <p className="text-xl text-slate-400 mb-10 max-w-2xl mx-auto leading-relaxed">
          AI-powered multimodal analysis of your audio, facial expressions, and semantics.
          Get actionable feedback tailored to propel your career.
        </p>
        
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link href="/interview" className="btn-primary text-lg">
            Start Live Simulation <ArrowRight size={20} />
          </Link>
          <Link href="/results/example" className="btn-secondary text-lg">
            View Example Report
          </Link>
        </div>
      </div>

      {/* Feature grid */}
      <div className="grid md:grid-cols-3 gap-6 mt-16 max-w-5xl relative z-10">
        <div className="glass-panel p-6 glass-panel-hover flex flex-col items-center">
          <div className="w-12 h-12 rounded-full bg-violet-600/20 flex items-center justify-center mb-4 text-violet-400">
            <Video size={24} />
          </div>
          <h3 className="font-semibold text-lg mb-2">Multimodal Analysis</h3>
          <p className="text-slate-400 text-sm text-center">Simultaneously captures facial cues, pitch modulation, and semantic keyword matching.</p>
        </div>
        <div className="glass-panel p-6 glass-panel-hover flex flex-col items-center">
          <div className="w-12 h-12 rounded-full bg-cyan-600/20 flex items-center justify-center mb-4 text-cyan-400">
            <Target size={24} />
          </div>
          <h3 className="font-semibold text-lg mb-2">Granular Scoring</h3>
          <p className="text-slate-400 text-sm text-center">Configurable weighting splits metrics into Confidence, Tech, and Communication axes.</p>
        </div>
        <div className="glass-panel p-6 glass-panel-hover flex flex-col items-center">
          <div className="w-12 h-12 rounded-full bg-pink-600/20 flex items-center justify-center mb-4 text-pink-400">
            <TrendingUp size={24} />
          </div>
          <h3 className="font-semibold text-lg mb-2">Actionable Feedback</h3>
          <p className="text-slate-400 text-sm text-center">Receive micro-targeted guidance such as eye-contact duration and filler-word reduction.</p>
        </div>
      </div>
      
    </main>
  );
}
