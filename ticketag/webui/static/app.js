// TickeTag web UI: submits a ticket to /api/classify and renders the result.
// Vanilla JS, no build step, no dependencies -- fetch is all this needs.

const form = document.getElementById("classify-form");
const textField = document.getElementById("ticket-text");
const submitButton = document.getElementById("submit-button");
const errorBanner = document.getElementById("error-banner");
const result = document.getElementById("result");
const resultBackend = document.getElementById("result-backend");
const resultCategory = document.getElementById("result-category");
const resultConfidence = document.getElementById("result-confidence");
const confidenceFill = document.getElementById("confidence-fill");
const resultJustification = document.getElementById("result-justification");

const BACKEND_LABELS = {
  rag: "RAG",
  zeroshot: "Zero-shot",
};

/** Hide the error banner and clear its message. */
function clearError() {
  errorBanner.hidden = true;
  errorBanner.textContent = "";
}

/** Show a human-readable error message inline, above the result area. */
function showError(message) {
  errorBanner.textContent = message;
  errorBanner.hidden = false;
  result.hidden = true;
}

/** Render a successful classification response into the result card. */
function showResult(payload) {
  resultBackend.textContent = BACKEND_LABELS[payload.backend] || payload.backend;
  resultCategory.textContent = payload.category;
  const percent = Math.round(payload.confidence * 100);
  resultConfidence.textContent = `${percent}%`;
  confidenceFill.style.width = `${percent}%`;
  resultJustification.textContent = payload.justification;
  result.hidden = false;
}

/** Read the selected backend from the radio group. Zero-shot is disabled in the UI. */
function selectedBackend() {
  const checked = form.querySelector('input[name="backend"]:checked:not(:disabled)');
  return checked ? checked.value : "rag";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();

  const text = textField.value.trim();
  if (!text) {
    showError("Please enter some ticket text before classifying.");
    return;
  }

  submitButton.disabled = true;
  submitButton.textContent = "Classifying...";
  try {
    const response = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, backend: selectedBackend() }),
    });
    const payload = await response.json();
    if (!response.ok) {
      const detail = typeof payload.detail === "string" ? payload.detail : "Classification failed.";
      showError(detail);
      return;
    }
    showResult(payload);
  } catch (error) {
    showError(`Could not reach the server: ${error.message}`);
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Classify ticket";
  }
});
