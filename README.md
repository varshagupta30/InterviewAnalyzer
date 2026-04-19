# InterviewIQ — Multimodal AI Interview Performance Analyzer

![InterviewIQ Banner](https://images.unsplash.com/photo-1551836022-d5d88e9218df?auto=format&fit=crop&q=80&w=2070)

InterviewIQ is a cutting-edge, multimodal AI platform designed to provide professional-grade feedback on interview performance. By leveraging computer vision, audio signal processing, and advanced NLP, the system provides a granular breakdown of how a candidate is perceived across multiple dimensions.

## 🚀 Core Features

- **Multimodal Intelligence**: Simultaneous analysis of video (facial expressions), audio (prosody and pace), and text (semantic relevance).
- **Performance Intelligence Dashboard**: Rich data visualization featuring:
  - **Animated Competency Rings**: Overall, Confidence, Communication, and Technical Depth scores.
  - **Biometric Sentiment Analysis**: Emotion distribution tracking (Positive, Neutral, Stress).
  - **Competency Radar**: Visual mapping of core strengths.
  - **Indexed Timeline Transcript**: Full session transcription with temporal markers.
- **Actionable AI Feedback**: Categorized Strengths and Opportunities for improvement.
- **Granular Audio Metrics**: WPM (Words Per Minute) tracking, filler word detection (um, like, so), and pause analysis.

## 🛠️ Tech Stack

### Frontend
- **Framework**: [Next.js 16](https://nextjs.org/) (Turbopack)
- **Styling**: Tailwind CSS 4.0 (Modern Glassmorphism)
- **Icons**: Lucide React
- **Charts**: Recharts (Radar, Area, Responsive Containers)

### Backend
- **Framework**: FastAPI (Python 3.10+)
- **Models**:
  - **OpenAI Whisper**: High-fidelity speech-to-text.
  - **BART-Large-MNLI**: Zero-shot classification for semantic relevance.
  - **Computer Vision**: Facial landmark and expression analysis.
- **Services**: Modular architecture for Audio, Vision, NLP, and Scoring aggregation.

## 🏁 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 20+
- NVIDIA GPU (Recommended for Whisper/BART acceleration)

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the API server:
   ```bash
   python main.py
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

## 📊 How It Works

1. **Capture**: The system records audio and video via the browser.
2. **Transcription**: Whisper processes the audio stream into time-stamped text.
3. **Analysis**:
   - **Audio Service** calculates WPM and detects filler word frequency.
   - **Vision Service** analyzes facial landmarks for eye contact and smiles.
   - **NLP Service** compares responses against technical benchmarks.
4. **Scoring**: The **Scoring Engine** aggregates metrics into four normalized axes.
5. **Reporting**: Next.js renders a premium, interactive performance report.

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---
Built by Team AI-Tards
