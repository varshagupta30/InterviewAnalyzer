const preview = document.getElementById("preview");
const countdownEl = document.getElementById("countdown");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

const durationSeconds = window.RECORD_CONFIG?.durationSeconds || 60;
const questionText = window.RECORD_CONFIG?.question || "";

let stream = null;
let mediaRecorder = null;
let chunks = [];
let countdownTimer = null;
let recordTimer = null;

async function initCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    preview.srcObject = stream;
    statusEl.textContent = "Camera ready. Click start when you are ready.";
  } catch (error) {
    statusEl.textContent = `Camera error: ${error.message}`;
    startBtn.disabled = true;
  }
}

function setControls(isRecording) {
  startBtn.disabled = isRecording;
  stopBtn.disabled = !isRecording;
}

function runCountdown(seconds) {
  return new Promise((resolve) => {
    let remaining = seconds;
    countdownEl.textContent = `Starting in ${remaining}...`;

    countdownTimer = setInterval(() => {
      remaining -= 1;
      if (remaining <= 0) {
        clearInterval(countdownTimer);
        countdownEl.textContent = "Recording...";
        resolve();
        return;
      }
      countdownEl.textContent = `Starting in ${remaining}...`;
    }, 1000);
  });
}

function startRecording() {
  chunks = [];
  mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm" });

  mediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      chunks.push(event.data);
    }
  };

  mediaRecorder.onstop = async () => {
    const blob = new Blob(chunks, { type: "video/webm" });
    await uploadRecording(blob);
  };

  mediaRecorder.start(1000);
  setControls(true);

  let remaining = durationSeconds;
  countdownEl.textContent = `Recording... ${remaining}s left`;

  recordTimer = setInterval(() => {
    remaining -= 1;
    countdownEl.textContent = `Recording... ${remaining}s left`;

    if (remaining <= 0) {
      stopRecording();
    }
  }, 1000);

  statusEl.textContent = "Recording in progress. Please answer the question naturally.";
}

function stopRecording() {
  if (recordTimer) {
    clearInterval(recordTimer);
  }

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }

  setControls(false);
  statusEl.textContent = "Uploading and analyzing your response...";
  countdownEl.textContent = "Processing...";
}

async function uploadRecording(blob) {
  const formData = new FormData();
  formData.append("video", blob, "response.webm");
  formData.append("question", questionText);

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "Upload failed.");
    }

    window.location.href = payload.result_url;
  } catch (error) {
    statusEl.textContent = `Upload/analysis failed: ${error.message}`;
    setControls(false);
    startBtn.disabled = false;
    countdownEl.textContent = "Ready to retry";
  }
}

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  await runCountdown(3);
  startRecording();
});

stopBtn.addEventListener("click", () => {
  stopRecording();
});

initCamera();
