"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Camera, Loader2, StopCircle, PlayCircle, ChevronRight } from "lucide-react";
import classNames from "classnames";

const QUESTIONS = [
  "Tell me about a time you had to overcome a significant technical challenge.",
  "How do you handle disagreements with team members or stakeholders?",
  "What is your approach to learning a completely new technology?"
];

export default function InterviewPage() {
  const [recording, setRecording] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  
  const [questionIndex, setQuestionIndex] = useState(0);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const router = useRouter();

  const startRecording = async () => {
    try {
      const ms = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      setStream(ms);
      if (videoRef.current) {
        videoRef.current.srcObject = ms;
      }
      
      const mediaRecorder = new MediaRecorder(ms);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = handleUpload;
      mediaRecorder.start();
      setRecording(true);
    } catch (err) {
      console.error("Failed to access camera", err);
      alert("Microphone and Camera permissions are required.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      setRecording(false);
      setProcessing(true);
      
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    }
  };

  const nextQuestion = () => {
    if (questionIndex < QUESTIONS.length - 1) {
      setQuestionIndex(prev => prev + 1);
    } else {
      // If no more questions, auto-stop recording
      stopRecording();
    }
  };

  const handleUpload = async () => {
    const blob = new Blob(chunksRef.current, { type: "video/webm" });
    const formData = new FormData();
    formData.append("video_file", blob, "interview.webm");
    
    try {
      const res = await fetch("http://127.0.0.1:8000/api/analyze", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      
      if (res.ok && data.session_id) {
        router.push(`/results/${data.session_id}`);
      } else {
        alert("Error processing video: " + data.detail);
        setProcessing(false);
      }
    } catch (err) {
      console.error("Upload failed", err);
      alert("Failed to connect to AI engine.");
      setProcessing(false);
    }
  };

  return (
    <div className="min-h-screen p-4 md:p-8 flex flex-col items-center justify-center relative">
      <div className="max-w-5xl w-full flex flex-col md:flex-row gap-6">
        
        {/* Main Camera View */}
        <div className="flex-1 glass-panel p-6 flex flex-col items-center relative overflow-hidden">
          
          {/* Question Overlay Header */}
          <div className="w-full text-center mb-6">
            <span className="text-violet-400 font-semibold mb-2 block uppercase tracking-wider text-sm">
              Question {questionIndex + 1} of {QUESTIONS.length}
            </span>
            <h2 className="text-2xl font-bold min-h-[4rem] flex items-center justify-center">
              "{QUESTIONS[questionIndex]}"
            </h2>
          </div>
          
          <div className="w-full aspect-video bg-black/50 rounded-xl overflow-hidden mb-6 border border-slate-700 relative shadow-inner flex items-center justify-center">
            <video 
              ref={videoRef} 
              autoPlay 
              muted 
              playsInline 
              className="w-full h-full object-cover"
            />
            
            {!recording && !stream && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 bg-slate-900/50 backdrop-blur-sm">
                <Camera size={48} className="mb-4 opacity-70"/>
                <p>Press Start to begin your interview</p>
              </div>
            )}
            
            {recording && (
              <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1 bg-red-500/20 text-red-400 rounded-full border border-red-500/30 animate-pulse">
                <div className="w-2 h-2 rounded-full bg-red-500" />
                <span className="text-sm font-medium">Recording</span>
              </div>
            )}
          </div>

          <div className="flex gap-4 w-full justify-center">
            {!recording && !processing ? (
              <button onClick={startRecording} className="btn-primary w-full md:w-auto">
                <PlayCircle size={20} />
                Start Interview
              </button>
            ) : processing ? (
              <button disabled className="btn-secondary !opacity-70 cursor-not-allowed w-full md:w-auto">
                <Loader2 size={20} className="animate-spin" />
                Analyzing Performance...
              </button>
            ) : (
              <div className="flex gap-4 w-full justify-center">
                <button 
                  onClick={nextQuestion} 
                  className={classNames(
                    "btn-primary flex-1 max-w-[250px]",
                    questionIndex === QUESTIONS.length - 1 ? "!bg-emerald-600 hover:!bg-emerald-500" : ""
                  )}
                >
                  {questionIndex === QUESTIONS.length - 1 ? "Finish Interview" : "Next Question"}
                  <ChevronRight size={20} />
                </button>
                <button 
                  onClick={stopRecording} 
                  className="btn-secondary !text-red-400 !border-red-500/50 hover:!bg-red-500/10"
                  title="Force Stop & End"
                >
                  <StopCircle size={20} /> End Early
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Sidebar Status / Map */}
        <div className="w-full md:w-80 flex flex-col gap-4">
          <div className="glass-panel p-6 flex-1">
            <h3 className="font-bold text-lg mb-4 text-slate-200 border-b border-slate-700/50 pb-2">Interview Plan</h3>
            <ul className="space-y-4">
              {QUESTIONS.map((q, idx) => (
                <li key={idx} className="flex gap-3 items-start">
                  <div className={classNames(
                    "w-6 h-6 rounded-full flex items-center justify-center shrink-0 mt-0.5 text-xs font-bold",
                    idx < questionIndex ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/50" : 
                    idx === questionIndex ? "bg-violet-500 text-white shadow-[0_0_10px_rgba(139,92,246,0.6)]" : 
                    "bg-slate-800 text-slate-500 border border-slate-700"
                  )}>
                    {idx + 1}
                  </div>
                  <span className={classNames(
                    "text-sm leading-snug",
                    idx < questionIndex ? "text-slate-400 line-through decoration-slate-600" :
                    idx === questionIndex ? "text-slate-100 font-medium" : "text-slate-500"
                  )}>
                    {q}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
