import { startTransition, useRef, useState } from "react";
import { DEMO_SCENARIOS } from "./demoScenarios.js";

const DEFAULT_API_BASE = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const SUPPORTED_IMAGE_EXTENSIONS = [
  ".jpg",
  ".jpeg",
  ".png",
  ".webp",
  ".bmp",
  ".gif",
];

function cloneScenario(payload) {
  return JSON.parse(JSON.stringify(payload));
}

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }

  return `${(value * 100).toFixed(2)}%`;
}

function formatNumber(value, digits = 4) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }

  return value.toFixed(digits);
}

function formatLabel(value) {
  if (!value) {
    return "--";
  }

  return String(value).replace(/_/g, " ");
}

function truncateFileName(fileName, maxLength = 42) {
  const normalizedName = String(fileName || "").trim();

  if (normalizedName.length <= maxLength) {
    return normalizedName;
  }

  const safeExtensionIndex = normalizedName.lastIndexOf(".");

  if (
    safeExtensionIndex <= 0 ||
    safeExtensionIndex < normalizedName.length - 8
  ) {
    return `${normalizedName.slice(0, maxLength - 3)}...`;
  }

  const safeExtension = normalizedName.slice(safeExtensionIndex);
  const safeBaseName = normalizedName.slice(0, safeExtensionIndex);
  const safeBaseLength = Math.max(8, maxLength - safeExtension.length - 3);

  return `${safeBaseName.slice(0, safeBaseLength)}...${safeExtension}`;
}

function extractErrorMessage(payload, statusCode) {
  if (payload?.detail && typeof payload.detail === "string") {
    return payload.detail;
  }

  if (payload?.detail?.message) {
    return payload.detail.message;
  }

  if (payload?.message) {
    return payload.message;
  }

  return `Request failed with status ${statusCode}.`;
}

function revokeObjectUrl(url) {
  if (url) {
    URL.revokeObjectURL(url);
  }
}

function hasSupportedImageExtension(fileName) {
  const normalizedName = String(fileName || "").trim().toLowerCase();

  return SUPPORTED_IMAGE_EXTENSIONS.some((extension) =>
    normalizedName.endsWith(extension)
  );
}

function getProbabilityEntries(probabilities) {
  return Object.entries(probabilities || {}).sort(
    (left, right) => right[1] - left[1]
  );
}

function ProbabilityBars({ probabilities }) {
  const entries = getProbabilityEntries(probabilities);

  return (
    <div className="probability-list">
      {entries.map(([label, value]) => (
        <div className="probability-row" key={label}>
          <div className="probability-meta">
            <span>{formatLabel(label)}</span>
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

function KeyCard({ label, value, tone = "default" }) {
  return (
    <div className={`key-card tone-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function clampUnitInterval(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return 0;
  }

  return Math.max(0, Math.min(1, value));
}

function GateMeter({ knownness, threshold, accepted, marginLabel }) {
  const knownnessPosition = clampUnitInterval(knownness) * 100;
  const thresholdPosition = clampUnitInterval(threshold) * 100;

  return (
    <div className={`gate-meter tone-${accepted ? "accept" : "review"}`}>
      <div className="gate-meter-header">
        <div className="gate-meter-title">
          <span className="section-tag">Fusion Gate</span>
          <strong>
            {accepted
              ? "Knownness clears the threshold"
              : "Knownness stays below the threshold"}
          </strong>
        </div>
        <span
          className={`gate-margin-pill tone-${accepted ? "accept" : "review"}`}
        >
          {marginLabel}
        </span>
      </div>

      <div className="gate-meter-track" aria-hidden="true">
        <div
          className="gate-meter-fill"
          style={{ width: `${knownnessPosition}%` }}
        />
        <div
          className="gate-meter-threshold"
          style={{ left: `${thresholdPosition}%` }}
        />
        <div
          className="gate-meter-point"
          style={{ left: `${knownnessPosition}%` }}
        />
      </div>

      <div className="gate-meter-scale">
        <span>0.0</span>
        <span>1.0</span>
      </div>

      <div className="gate-meter-legend">
        <span className="gate-meter-chip">
          <span className="gate-meter-dot knownness" />
          Knownness {formatNumber(knownness)}
        </span>
        <span className="gate-meter-chip">
          <span className="gate-meter-dot threshold" />
          Threshold {formatNumber(threshold)}
        </span>
      </div>
    </div>
  );
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [resultImageUrl, setResultImageUrl] = useState("");
  const [resultPayload, setResultPayload] = useState(
    cloneScenario(DEMO_SCENARIOS.known.payload)
  );
  const [resultSource, setResultSource] = useState("demo");
  const [statusMessage, setStatusMessage] = useState({
    tone: "info",
    text: DEMO_SCENARIOS.known.statusText,
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef(null);
  const loadingStageTimersRef = useRef([]);

  const result = resultPayload?.result || null;
  const prediction = result?.prediction || null;
  const fusionGate = result?.fusion_gate || null;
  const finalDecision = result?.final_decision || null;
  const accepted = Boolean(finalDecision?.accepted_as_known);
  const displayedResultImageUrl =
    resultSource === "live" && resultImageUrl ? resultImageUrl : "";
  const activeSampleImageUrl = displayedResultImageUrl || previewUrl;
  const outcomeBadgeLabel = accepted
    ? "Accepted known item"
    : "Manual review required";
  const knownnessScore =
    typeof fusionGate?.knownness_score === "number"
      ? fusionGate.knownness_score
      : null;
  const thresholdValue =
    typeof fusionGate?.threshold === "number" ? fusionGate.threshold : null;
  const gateMargin =
    knownnessScore !== null && thresholdValue !== null
      ? knownnessScore - thresholdValue
      : null;
  const gateMarginLabel =
    gateMargin === null
      ? "margin unavailable"
      : `${formatNumber(Math.abs(gateMargin))} ${
          gateMargin >= 0 ? "above threshold" : "below threshold"
        }`;
  const decisionReasonChips = accepted
    ? ["Known item", "Above threshold", "Safe to show"]
    : ["Manual review", "Below threshold", "Guess hidden"];
  const hasResolvedLiveResult =
    resultSource === "live" && Boolean(resultImageUrl);
  const shouldShowUploadSummary =
    Boolean(selectedFile) ||
    isSubmitting ||
    isCheckingHealth ||
    statusMessage.tone !== "info";

  const sampleTitle =
    resultPayload?.uploaded_filename || selectedFile?.name || "No sample selected";
  const resultSampleCaption = displayedResultImageUrl
    ? "Evaluated sample"
    : previewUrl
      ? "Selected sample"
      : "Demo sample";
  const interpretationSummary = accepted
    ? `The classifier guess is considered safe, so the ${formatLabel(
        finalDecision?.user_visible_label
      )} label is shown to the user.`
    : "The classifier still makes an internal guess, but the Fusion Gate blocks it and sends the sample to manual review.";

  function clearLoadingStageTimers() {
    loadingStageTimersRef.current.forEach((timerId) => clearTimeout(timerId));
    loadingStageTimersRef.current = [];
  }

  function scheduleInferenceStages() {
    clearLoadingStageTimers();

    loadingStageTimersRef.current = [
      setTimeout(() => {
        setStatusMessage({
          tone: "info",
          text: "Running classifier on the uploaded sample...",
        });
      }, 350),
      setTimeout(() => {
        setStatusMessage({
          tone: "info",
          text: "Computing gate score and comparing it with the threshold...",
        });
      }, 1250),
      setTimeout(() => {
        setStatusMessage({
          tone: "info",
          text: "Preparing final output and user-facing decision...",
        });
      }, 2200),
    ];
  }

  function clearLiveSample() {
    clearLoadingStageTimers();
    revokeObjectUrl(previewUrl);

    setSelectedFile(null);
    setPreviewUrl("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }

    setStatusMessage({
      tone: "info",
      text: "Live sample cleared. Upload a new image or use a demo scenario.",
    });
  }

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  function handleFile(file) {
    if (!file) {
      return;
    }

    clearLoadingStageTimers();
    const hasImageMime = String(file.type || "").startsWith("image/");
    const hasAllowedExtension = hasSupportedImageExtension(file.name);

    if (!hasImageMime && !hasAllowedExtension) {
      setStatusMessage({
        tone: "error",
        text: "Please choose a valid image file for live inference.",
      });
      return;
    }

    revokeObjectUrl(previewUrl);

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setStatusMessage({
      tone: "info",
      text: "Image selected. Run live inference to update the result.",
    });
  }

  function loadScenario(name) {
    const scenario = DEMO_SCENARIOS[name];

    clearLoadingStageTimers();
    revokeObjectUrl(previewUrl);
    revokeObjectUrl(resultImageUrl);

    startTransition(() => {
      setSelectedFile(null);
      setPreviewUrl("");
      setResultImageUrl("");
      setResultPayload(cloneScenario(scenario.payload));
      setResultSource("demo");
      setStatusMessage({
        tone: "info",
        text: scenario.statusText,
      });
    });

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  async function handleHealthCheck() {
    clearLoadingStageTimers();
    const normalizedBase = DEFAULT_API_BASE.trim().replace(/\/+$/, "");

    if (!normalizedBase) {
      setStatusMessage({
        tone: "error",
        text: "Backend URL is not configured.",
      });
      return;
    }

    setIsCheckingHealth(true);
    setStatusMessage({
      tone: "info",
      text: "Checking backend health...",
    });

    try {
      const response = await fetch(`${normalizedBase}/api/health`);
      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, response.status));
      }

      setStatusMessage({
        tone: "success",
        text: `Backend ready: ${payload.policy_version || "final policy"} on ${
          payload.device || "unknown device"
        }.`,
      });
    } catch (error) {
      setStatusMessage({
        tone: "error",
        text:
          error.message ||
          "Health check failed. Make sure the backend is running.",
      });
    } finally {
      setIsCheckingHealth(false);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    clearLoadingStageTimers();

    const normalizedBase = DEFAULT_API_BASE.trim().replace(/\/+$/, "");

    if (!normalizedBase) {
      setStatusMessage({
        tone: "error",
        text: "Backend URL is not configured.",
      });
      return;
    }

    if (!selectedFile) {
      setStatusMessage({
        tone: "error",
        text: "Choose an image first, or use one of the demo scenarios.",
      });
      return;
    }

    setIsSubmitting(true);
    setStatusMessage({
      tone: "info",
      text: "Uploading sample to the prototype pipeline...",
    });
    scheduleInferenceStages();

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${normalizedBase}/api/predict`, {
        method: "POST",
        body: formData,
      });

      const payload = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, response.status));
      }

      const nextResultImageUrl = URL.createObjectURL(selectedFile);
      revokeObjectUrl(resultImageUrl);

      startTransition(() => {
        setResultImageUrl(nextResultImageUrl);
        setResultPayload(payload);
        setResultSource("live");
      });

      setStatusMessage({
        tone: "success",
        text: "Live prediction complete. Review the final decision and gate result.",
      });
    } catch (error) {
      setStatusMessage({
        tone: "error",
        text:
          error.message ||
          "Live prediction failed. The built-in demo scenarios are still available.",
      });
    } finally {
      clearLoadingStageTimers();
      setIsSubmitting(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="top-header panel simple-header">
        <div className="header-main">
          <div className="section-tag">OpenWaste-HR Prototype</div>
          <h1>Safety-aware waste classification</h1>
          <p>
            Upload one sample and show how the classifier guess is filtered by
            the Fusion Gate before a label is shown to the user.
          </p>
        </div>
      </header>

      <section className="workspace-grid simple-workspace">
        <aside className="panel input-panel simple-input-panel">
          <form
            className={`input-form ${hasResolvedLiveResult ? "compact-after-result" : ""}`}
            onSubmit={handleSubmit}
          >
            <div
              className={`upload-box ${isDragging ? "dragging" : ""} ${
                hasResolvedLiveResult ? "compact" : ""
              }`}
              role="button"
              tabIndex={0}
              aria-label="Upload an image for live inference"
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setIsDragging(false);
                handleFile(event.dataTransfer.files?.[0]);
              }}
              onClick={openFilePicker}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  openFilePicker();
                }
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                hidden
                onChange={(event) => handleFile(event.target.files?.[0])}
              />

              {previewUrl ? (
                <>
                  <img
                    className="preview-image"
                    src={previewUrl}
                    alt="Selected sample preview"
                  />
                  <div className="image-caption">Selected sample</div>
                </>
              ) : (
                <div className="upload-copy">
                  <div className="upload-icon">+</div>
                  <h3>Upload image</h3>
                  <p>Click or drag a sample here.</p>
                </div>
              )}
            </div>

            {shouldShowUploadSummary ? (
              <div
                className={`upload-summary tone-${statusMessage.tone}`}
                aria-live="polite"
              >
                <div className="upload-summary-head">
                  <span>{selectedFile ? "Current sample" : "System status"}</span>
                  {selectedFile ? (
                    <strong title={selectedFile.name}>
                      {truncateFileName(selectedFile.name, 38)}
                    </strong>
                  ) : null}
                </div>
                <p>{statusMessage.text}</p>
              </div>
            ) : null}

            <button
              className="primary-button"
              type="submit"
              disabled={isSubmitting}
            >
              {isSubmitting ? "Running..." : "Run live inference"}
            </button>

            <div className="utility-row">
              <button
                className="secondary-button support-button"
                type="button"
                onClick={handleHealthCheck}
                disabled={isCheckingHealth}
              >
                {isCheckingHealth ? "Checking..." : "Check backend"}
              </button>
              <button
                className="ghost-button"
                type="button"
                onClick={clearLiveSample}
              >
                Clear sample
              </button>
            </div>
          </form>

          <div className="scenario-block">
            <div className="section-tag">Demo Cases</div>
            <div className="scenario-buttons compact-row">
              <button
                className="scenario-button accept"
                type="button"
                onClick={() => loadScenario("known")}
              >
                Accepted case
              </button>
              <button
                className="scenario-button review"
                type="button"
                onClick={() => loadScenario("review")}
              >
                Review case
              </button>
            </div>
          </div>
        </aside>

        <section className="center-column">
          <div className="panel result-panel simple-result-panel">
            <div className={`decision-card tone-${accepted ? "accept" : "review"}`}>
              <div className="decision-header">
                <div>
                  <div className="decision-kicker">
                    <div className="section-tag">Final Decision</div>
                    <div
                      className={`outcome-pill tone-${accepted ? "accept" : "review"}`}
                    >
                      {outcomeBadgeLabel}
                    </div>
                  </div>
                  <h2>
                    {accepted
                      ? formatLabel(finalDecision?.user_visible_label)
                      : "Manual review required"}
                  </h2>
                </div>

                <div className={`source-pill tone-${resultSource}`}>
                  {resultSource === "live" ? "Live" : "Demo"}
                </div>
              </div>

              <p>{finalDecision?.user_message || "No user-facing message returned."}</p>

              <div className="result-main-grid">
                <div className="sample-card result-sample-card">
                  <div className="sample-frame">
                    {activeSampleImageUrl ? (
                      <>
                        <img
                          className="result-image"
                          src={activeSampleImageUrl}
                          alt="Evaluated waste sample preview"
                        />
                        <div className="image-caption">{resultSampleCaption}</div>
                      </>
                    ) : (
                      <div
                        className={`sample-placeholder ${accepted ? "accept" : "review"}`}
                      >
                        <div className="sample-type">
                          {resultSource === "live" ? "Live sample" : "Demo replay"}
                        </div>
                        <strong title={sampleTitle}>{sampleTitle}</strong>
                        <span>{formatLabel(prediction?.internal_top1_prediction)}</span>
                      </div>
                    )}
                  </div>
                </div>

                <div className="result-main-stack">
                  <GateMeter
                    knownness={fusionGate?.knownness_score}
                    threshold={fusionGate?.threshold}
                    accepted={accepted}
                    marginLabel={gateMarginLabel}
                  />

                  <div className="reason-chip-list" aria-label="Decision reasons">
                    {decisionReasonChips.map((reason) => (
                      <span
                        className={`reason-chip tone-${accepted ? "accept" : "review"}`}
                        key={reason}
                      >
                        {reason}
                      </span>
                    ))}
                  </div>

                  <div className="score-block">
                    <div className="section-tag">Important Scores</div>
                    <div className="key-grid simple-key-grid">
                      <KeyCard
                        label="Shown to user"
                        value={formatLabel(finalDecision?.user_visible_label)}
                        tone={accepted ? "accept" : "review"}
                      />
                      <KeyCard
                        label="Internal guess"
                        value={formatLabel(prediction?.internal_top1_prediction)}
                      />
                      <KeyCard
                        label="Knownness"
                        value={formatNumber(fusionGate?.knownness_score)}
                        tone={accepted ? "accept" : "review"}
                      />
                      <KeyCard
                        label="Threshold"
                        value={formatNumber(fusionGate?.threshold)}
                      />
                      <KeyCard
                        label="Temp confidence"
                        value={formatPercent(
                          prediction?.temperature_scaled_confidence
                        )}
                        tone="live"
                      />
                      <KeyCard
                        label="Raw confidence"
                        value={formatPercent(prediction?.raw_confidence)}
                      />
                    </div>
                  </div>

                  <div className="decision-explainer">
                    <div className="section-tag">Interpretation</div>
                    <p>
                      {interpretationSummary} The internal top-1 class is{" "}
                      <strong>{formatLabel(prediction?.internal_top1_prediction)}</strong>.
                    </p>
                  </div>

                  <div className="probability-section inline-probability">
                    <div className="probability-heading">
                      <div className="section-tag">Internal Prediction Scores</div>
                      <span>Classifier percentages across known classes</span>
                    </div>
                    <ProbabilityBars probabilities={prediction?.class_probabilities} />
                  </div>
                </div>
              </div>
            </div>
          </div>

          <details className="panel raw-json simple-details">
            <summary>More technical details</summary>
            <div className="details-body">
              <pre>{JSON.stringify(resultPayload, null, 2)}</pre>
            </div>
          </details>
        </section>
      </section>
    </main>
  );
}
