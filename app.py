"""Flask entry point for interview analyzer web application."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json
import random
import uuid

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from analyzer.video_analyzer import VideoAnalyzer


BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
QUESTIONS_FILE = BASE_DIR / "questions.txt"


app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-secret-change-me"

# Ensure local storage folders exist.
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize analyzer once for request reuse.
analyzer = VideoAnalyzer()


def load_questions() -> List[str]:
    """Load editable interview questions from questions.txt."""
    if not QUESTIONS_FILE.exists():
        return ["[EASY] Tell me about yourself."]

    questions: List[str] = []
    for line in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        questions.append(line)

    return questions or ["[EASY] Tell me about yourself."]


@app.get("/")
def index():
    """Show landing page with a random question from editable bank."""
    questions = load_questions()
    selected_question = random.choice(questions)

    # Keep question in session so record page can reuse it.
    session["current_question"] = selected_question
    return render_template("index.html", question=selected_question)


@app.get("/record")
def record_page():
    """Render 60-second recording page with current interview question."""
    question = session.get("current_question", "[EASY] Tell me about yourself.")
    return render_template("record.html", question=question, duration_seconds=60)


@app.post("/upload")
def upload_video():
    """Receive recorded webcam video, analyze it, and persist session result."""
    if "video" not in request.files:
        return jsonify({"error": "Missing 'video' file in request."}), 400

    video_file = request.files["video"]
    question = request.form.get("question", "[EASY] Tell me about yourself.")

    # Use UUID for each session for retrieval and isolation.
    session_id = str(uuid.uuid4())
    suffix = ".webm"
    filename = f"{session_id}{suffix}"
    video_path = UPLOADS_DIR / filename

    # Save uploaded recording file.
    video_file.save(video_path)

    try:
        analysis = analyzer.analyze_video(str(video_path), max_seconds=60)
    except Exception as exc:
        return jsonify({"error": f"Analysis failed: {exc}"}), 500

    payload: Dict[str, object] = {
        "session_id": session_id,
        "question": question,
        "analysis": analysis,
    }

    result_file = RESULTS_DIR / f"{session_id}.json"
    result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return jsonify({"session_id": session_id, "result_url": url_for("results_page", id=session_id)})


@app.get("/results/<id>")
def results_page(id: str):
    """Show scores and feedback for a completed interview session."""
    result_file = RESULTS_DIR / f"{id}.json"
    if not result_file.exists():
        return redirect(url_for("index"))

    payload = json.loads(result_file.read_text(encoding="utf-8"))
    return render_template(
        "results.html",
        session_id=id,
        question=payload.get("question", "N/A"),
        scores=payload["analysis"]["scores"],
        feedback=payload["analysis"]["feedback"],
    )


if __name__ == "__main__":
    app.run(debug=True)
