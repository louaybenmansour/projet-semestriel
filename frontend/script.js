const API_URL = "http://127.0.0.1:8000/predict";

const form = document.getElementById("predict-form");
const predictBtn = document.getElementById("predict-btn");
const resultBox = document.getElementById("result");
const loadingBox = document.getElementById("loading");
const errorBox = document.getElementById("error");
const resultScore = document.getElementById("result-score");
const resultLabel = document.getElementById("result-label");
const resultInterpretation = document.getElementById("result-interpretation");
const resultModelType = document.getElementById("result-model-type");
const resultConfidence = document.getElementById("result-confidence");
const sliderMap = [
  { id: "studyHours", rangeId: "studyHoursRange", badgeId: "studyHoursBadge", suffix: " h", decimals: 1 },
  { id: "sleepHours", rangeId: "sleepHoursRange", badgeId: "sleepHoursBadge", suffix: " h", decimals: 1 },
  { id: "attendance", rangeId: "attendanceRange", badgeId: "attendanceBadge", suffix: "%", decimals: 0 },
  { id: "stressLevel", rangeId: "stressLevelRange", badgeId: "stressLevelBadge", suffix: "", decimals: 0 },
];

function parseInput(id, min, max) {
  const value = Number(document.getElementById(id).value);
  if (Number.isNaN(value)) {
    throw new Error(`${id} must be a valid number.`);
  }
  if (value < min || value > max) {
    throw new Error(`${id} must be between ${min} and ${max}.`);
  }
  return value;
}

function setLoading(isLoading) {
  predictBtn.disabled = isLoading;
  loadingBox.classList.toggle("hidden", !isLoading);
}

function resetMessages() {
  errorBox.classList.add("hidden");
  resultBox.classList.add("hidden");
  errorBox.textContent = "";
}

async function predict(payload) {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    const detail = data?.detail || "Prediction request failed.";
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function renderResult(data) {
  resultScore.textContent = `${data.prediction}/100`;
  resultLabel.textContent = `Label: ${data.label}`;
  resultInterpretation.textContent = `Interpretation: ${data.interpretation}`;
  resultModelType.textContent = `Model: ${data.model_name || "Unknown"} (${data.model_type || "ML"})`;
  const confidencePct = Number(data.confidence || 0) * 100;
  resultConfidence.textContent = `Confidence: ${confidencePct.toFixed(0)}%`;
  resultBox.classList.remove("hidden");
}

function updateBadge(meta, value) {
  const badge = document.getElementById(meta.badgeId);
  if (!badge) return;
  const num = Number(value);
  if (Number.isNaN(num)) return;
  badge.textContent = `${num.toFixed(meta.decimals)}${meta.suffix}`;
}

function syncControls(meta) {
  const numberInput = document.getElementById(meta.id);
  const rangeInput = document.getElementById(meta.rangeId);
  if (!numberInput || !rangeInput) return;

  const handleValue = (value, source) => {
    const next = Number(value);
    if (Number.isNaN(next)) return;
    if (source !== "number") numberInput.value = String(next);
    if (source !== "range") rangeInput.value = String(next);
    updateBadge(meta, next);
  };

  rangeInput.addEventListener("input", (e) => handleValue(e.target.value, "range"));
  numberInput.addEventListener("input", (e) => handleValue(e.target.value, "number"));
  updateBadge(meta, numberInput.value);
}

sliderMap.forEach(syncControls);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resetMessages();

  try {
    const payload = {
      StudyHours: parseInput("studyHours", 0, 24),
      SleepHours: parseInput("sleepHours", 0, 16),
      Attendance: parseInput("attendance", 0, 100),
      StressLevel: parseInput("stressLevel", 0, 100),
    };

    setLoading(true);
    const data = await predict(payload);
    renderResult(data);
  } catch (error) {
    errorBox.textContent = error.message || "Unexpected error.";
    errorBox.classList.remove("hidden");
  } finally {
    setLoading(false);
  }
});
