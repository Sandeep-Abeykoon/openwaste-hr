import { useMemo, useRef, useState } from "react";

const API_URL = "http://127.0.0.1:8000/api/predict";

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "0.00%";
  }

  return `${(value * 100).toFixed(2)}%`;
}

function formatNumber(value, digits = 4) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "0";
  }

  return value.toFixed(digits);
}

function classNameForDecision(accepted) {
  return accepted ? "accepted" : "rejected";
}

function getDecisionTitle(accepted) {
  return accepted ? "Accepted as Known Waste" : "Manual Review Required";
}

function getDecisionSubtitle(accepted) {
  return accepted
    ? "The item passed the Fusion Gate v2 safety threshold."
    : "The item did not safely match the supported known classes.";
}

function ProbabilityBars({ probabilities }) {
  const entries = Object.entries(probabilities || {}).sort((a, b) => b[1] - a[1]);

  return (
    <div className="probability-list">
      {entries.map(([label, value]) => (
        <div className="probability-row" key={label}>
          <div className="probability-label">
            <span>{label}</span>
            <strong>{formatPercent(value)}</strong>
          </div>
          <div className="probability-track">
            <div
              className="probability-fill"
              style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function ScoreMeter({ score, threshold }) {
  const scorePercent = Math.max(0, Math.min(100, score * 100));
  const thresholdPercent = Math.max(0, Math.min(100, threshold * 100));

  return (
    <div className="score-meter">
      <div className="score-meter-top">
        <span>Fusion knownness score</span>
        <strong>{formatNumber(score, 4)}</strong>
      </div>

      <div className="meter-track">
        <div className="meter-fill" style={{ width: `${scorePercent}%` }} />
        <div className="meter-threshold" style={{ left: `${thresholdPercent}%` }} />
      </div>

      <div className="score-meter-bottom">
        <span>0.0000</span>
        <span>Threshold: {formatNumber(threshold, 4)}</span>
        <span>1.0000</span>
      </div>
    </div>
  );
}

function TechnicalDetails({ result }) {
  const [open, setOpen] = useState(false);

  if (!result) {
    return null;
  }

  const prediction = result.prediction;
  const mahalanobis = result.mahalanobis;
  const fusionGate = result.fusion_gate;

  return (
    <div className="technical-panel">
      <button className="technical-toggle" onClick={() => setOpen(!open)}>
        {open ? "Hide technical details" : "Show technical details"}
      </button>

      {open && (
        <div className="technical-grid">
          <div className="technical-card">
            <span>Internal top-1 prediction</span>
            <strong>{prediction.internal_top1_prediction}</strong>
          </div>

          <div className="technical-card">
            <span>Raw confidence</span>
            <strong>{formatPercent(prediction.raw_confidence)}</strong>
          </div>

          <div className="technical-card">
            <span>Temperature confidence</span>
            <strong>{formatPercent(prediction.temperature_scaled_confidence)}</strong>
          </div>

          <div className="technical-card">
            <span>Energy</span>
            <strong>{formatNumber(prediction.energy, 4)}</strong>
          </div>

          <div className="technical-card">
            <span>Softmax margin</span>
            <strong>{formatNumber(prediction.softmax_margin, 4)}</strong>
          </div>

          <div className="technical-card">
            <span>Softmax entropy</span>
            <strong>{formatNumber(prediction.softmax_entropy, 4)}</strong>
          </div>

          <div className="technical-card">
            <span>Mahalanobis nearest class</span>
            <strong>{mahalanobis.mahalanobis_nearest_class}</strong>
          </div>

          <div className="technical-card">
            <span>Mahalanobis knownness</span>
            <strong>{formatNumber(mahalanobis.mahalanobis_knownness, 4)}</strong>
          </div>

          <div className="technical-card">
            <span>Decision type</span>
            <strong>{fusionGate.decision_type}</strong>
          </div>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [apiResponse, setApiResponse] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [isPredicting, setIsPredicting] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef(null);

  const result = apiResponse?.result || null;
  const finalDecision = result?.final_decision || null;
  const prediction = result?.prediction || null;
  const fusionGate = result?.fusion_gate || null;

  const accepted = Boolean(finalDecision?.accepted_as_known);

  const supportedClasses = useMemo(
    () => ["cardboard", "glass", "metal", "paper", "plastic"],
    []
  );

  function handleFile(file) {
    setErrorMessage("");
    setApiResponse(null);

    if (!file) {
      return;
    }

    if (!file.type.startsWith("image/")) {
      setErrorMessage("Please select a valid image file.");
      return;
    }

    setSelectedFile(file);

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  }

  function handleFileInputChange(event) {
    const file = event.target.files?.[0];
    handleFile(file);
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);

    const file = event.dataTransfer.files?.[0];
    handleFile(file);
  }

  async function handlePredict() {
    if (!selectedFile) {
      setErrorMessage("Please upload an image first.");
      return;
    }

    setIsPredicting(true);
    setErrorMessage("");
    setApiResponse(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setApiResponse(data);
    } catch (error) {
      setErrorMessage(
        error.message ||
          "Could not connect to the prediction API. Make sure the backend is running."
      );
    } finally {
      setIsPredicting(false);
    }
  }

  function resetInterface() {
    setSelectedFile(null);
    setPreviewUrl("");
    setApiResponse(null);
    setErrorMessage("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  return (
    <main className="app-shell">
      <section className="hero-section">
        <div className="hero-copy">
          <div className="eyebrow">OpenWaste-HR</div>
          <h1>Smart Waste Prediction Interface</h1>
          <p>
            Upload a waste image and let the final Fusion Gate v2 policy decide whether
            it can be safely classified or should be sent for manual review.
          </p>

          <div className="hero-badges">
            <span>MobileNetV3</span>
            <span>Fusion Gate v2</span>
            <span>Mahalanobis Safety Gate</span>
          </div>
        </div>

        <div className="policy-card">
          <span>Final policy</span>
          <strong>Fusion Gate v2 + Mahalanobis</strong>
          <p>Rejects uncertain or unknown items before showing a final label.</p>
        </div>
      </section>

      <section className="main-grid">
        <div className="upload-card">
          <div className="card-header">
            <span>Step 1</span>
            <h2>Upload Image</h2>
          </div>

          <div
            className={`dropzone ${isDragging ? "dragging" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileInputChange}
              hidden
            />

            {previewUrl ? (
              <img className="preview-image" src={previewUrl} alt="Uploaded waste preview" />
            ) : (
              <div className="dropzone-content">
                <div className="upload-icon">↑</div>
                <h3>Drop an image here</h3>
                <p>or click to browse from your computer</p>
              </div>
            )}
          </div>

          {selectedFile && (
            <div className="file-meta">
              <span>Selected file</span>
              <strong>{selectedFile.name}</strong>
            </div>
          )}

          {errorMessage && <div className="error-box">{errorMessage}</div>}

          <div className="button-row">
            <button
              className="primary-button"
              onClick={handlePredict}
              disabled={!selectedFile || isPredicting}
            >
              {isPredicting ? "Predicting..." : "Run Prediction"}
            </button>

            <button className="secondary-button" onClick={resetInterface}>
              Reset
            </button>
          </div>

          <div className="supported-card">
            <span>Supported known classes</span>
            <div className="class-pills">
              {supportedClasses.map((className) => (
                <span key={className}>{className}</span>
              ))}
            </div>
          </div>
        </div>

        <div className="result-card">
          <div className="card-header">
            <span>Step 2</span>
            <h2>Prediction Result</h2>
          </div>

          {!result && (
            <div className="empty-state">
              <div className="empty-icon">◎</div>
              <h3>No prediction yet</h3>
              <p>Upload an image and click Run Prediction to see the result.</p>
            </div>
          )}

          {result && (
            <>
              <div className={`decision-banner ${classNameForDecision(accepted)}`}>
                <div>
                  <span>{getDecisionTitle(accepted)}</span>
                  <h3>{finalDecision.user_visible_label}</h3>
                  <p>{getDecisionSubtitle(accepted)}</p>
                </div>

                <div className="decision-chip">
                  {accepted ? "Known" : "Review"}
                </div>
              </div>

              <div className="message-card">
                <span>User-facing message</span>
                <p>{finalDecision.user_message}</p>
              </div>

              <ScoreMeter
                score={fusionGate.knownness_score}
                threshold={fusionGate.threshold}
              />

              <div className="metric-grid">
                <div className="metric-card">
                  <span>Temperature confidence</span>
                  <strong>{formatPercent(prediction.temperature_scaled_confidence)}</strong>
                </div>

                <div className="metric-card">
                  <span>Raw confidence</span>
                  <strong>{formatPercent(prediction.raw_confidence)}</strong>
                </div>

                <div className="metric-card">
                  <span>Coarse category</span>
                  <strong>{finalDecision.coarse_label}</strong>
                </div>

                <div className="metric-card">
                  <span>Policy version</span>
                  <strong>{result.policy_version}</strong>
                </div>
              </div>

              <div className="probability-card">
                <div className="section-heading">
                  <span>Class probability distribution</span>
                  <p>
                    These are internal known-class probabilities. For rejected items,
                    the final user label remains manual review.
                  </p>
                </div>

                <ProbabilityBars probabilities={prediction.class_probabilities} />
              </div>

              <TechnicalDetails result={result} />
            </>
          )}
        </div>
      </section>
    </main>
  );
}
