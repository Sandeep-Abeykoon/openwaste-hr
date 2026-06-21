const predictionForm = document.getElementById("predictionForm");
const imagePathInput = document.getElementById("imagePathInput");
const sampleIdInput = document.getElementById("sampleIdInput");
const apiBaseInput = document.getElementById("apiBaseInput");
const predictBtn = document.getElementById("predictBtn");
const statusMessage = document.getElementById("statusMessage");

const resultSection = document.getElementById("resultSection");
const finalLabel = document.getElementById("finalLabel");
const decisionType = document.getElementById("decisionType");
const decisionReason = document.getElementById("decisionReason");
const predLabel = document.getElementById("predLabel");
const confidenceValue = document.getElementById("confidenceValue");
const coarseValue = document.getElementById("coarseValue");
const probabilityTable = document.getElementById("probabilityTable");
const rawJson = document.getElementById("rawJson");

function setStatus(message, type = "") {
  statusMessage.textContent = message;
  statusMessage.className = `status-message ${type}`.trim();
}

function formatNumber(value) {
  if (typeof value !== "number") {
    return value;
  }

  return value.toFixed(6);
}

function createRequestId() {
  return `frontend_demo_${Date.now()}`;
}

function renderProbabilityTable(probabilities) {
  const rows = Object.entries(probabilities || {})
    .map(([label, probability]) => {
      return `
        <tr>
          <td>${label}</td>
          <td>${formatNumber(probability)}</td>
        </tr>
      `;
    })
    .join("");

  probabilityTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Fine Label</th>
          <th>Probability</th>
        </tr>
      </thead>
      <tbody>
        ${rows}
      </tbody>
    </table>
  `;
}

function renderSuccessResponse(data) {
  const prediction = data.prediction || {};
  const decision = data.decision || {};

  finalLabel.textContent = decision.final_label || "-";
  decisionType.textContent = `Decision type: ${decision.decision_type || "-"}`;
  decisionReason.textContent = `Reason: ${decision.reason || "-"}`;

  predLabel.textContent = prediction.pred_label || "-";
  confidenceValue.textContent = `Confidence: ${formatNumber(
    prediction.max_softmax_confidence
  )}`;
  coarseValue.textContent = `Coarse: ${prediction.top_coarse_label || "-"} (${formatNumber(
    prediction.top_coarse_confidence
  )})`;

  renderProbabilityTable(data.class_probabilities || {});
  rawJson.textContent = JSON.stringify(data, null, 2);

  resultSection.classList.remove("hidden");
}

async function runPrediction(event) {
  event.preventDefault();

  const apiBase = apiBaseInput.value.trim().replace(/\/$/, "");
  const imagePath = imagePathInput.value.trim();
  const sampleId = sampleIdInput.value.trim();

  if (!apiBase || !imagePath) {
    setStatus("Backend URL and image path are required.", "error");
    return;
  }

  const payload = {
    image_path: imagePath,
    sample_id: sampleId || null,
    request_id: createRequestId(),
  };

  predictBtn.disabled = true;
  setStatus("Running inference...", "");

  try {
    const response = await fetch(`${apiBase}/api/inference/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      rawJson.textContent = JSON.stringify(data, null, 2);
      resultSection.classList.remove("hidden");
      setStatus("Backend returned an error. Check the raw response.", "error");
      return;
    }

    renderSuccessResponse(data);
    setStatus("Prediction completed successfully.", "success");
  } catch (error) {
    setStatus(`Request failed: ${error.message}`, "error");
  } finally {
    predictBtn.disabled = false;
  }
}

predictionForm.addEventListener("submit", runPrediction);