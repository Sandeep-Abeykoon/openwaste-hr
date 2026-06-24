import { startTransition, useRef, useState } from "react";
import {
  DEMO_SCENARIOS,
  KNOWN_CLASSES,
  RESEARCH_STATS,
} from "./demoScenarios.js";

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
  return Object.entries(probabilities || {}).sort((left, right) => right[1] - left[1]);
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

function CompactStat({ label, value }) {
  return (
    <div className="compact-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function FlowStep({ index, title, value, note, tone = "default" }) {
  return (
    <div className={`flow-step tone-${tone}`}>
      <div className="flow-index">{index}</div>
      <div className="flow-copy">
        <span>{title}</span>
        <strong>{value}</strong>
        <small>{note}</small>
      </div>
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

function SignalRow({ label, value }) {
  return (
    <div className="signal-row">
      <span>{label}</span>
      <strong>{value}</strong>
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

  const result = resultPayload?.result || null;
  const prediction = result?.prediction || null;
  const fusionGate = result?.fusion_gate || null;
  const mahalanobis = result?.mahalanobis || null;
  const finalDecision = result?.final_decision || null;
  const accepted = Boolean(finalDecision?.accepted_as_known);
  const displayedResultImageUrl =
    resultSource === "live" && resultImageUrl ? resultImageUrl : "";

  const sampleTitle =
    resultPayload?.uploaded_filename || selectedFile?.name || "No sample selected";

  const flowItems = [
    {
      index: "01",
      title: "Sample",
      value: sampleTitle,
      note: resultSource === "live" ? "live upload" : "research replay",
      tone: resultSource === "live" ? "live" : "demo",
    },
    {
      index: "02",
      title: "Classifier Guess",
      value: formatLabel(prediction?.internal_top1_prediction),
      note: `temp conf ${formatPercent(prediction?.temperature_scaled_confidence)}`,
    },
    {
      index: "03",
      title: "Safety Gate",
      value: formatNumber(fusionGate?.knownness_score),
      note: `threshold ${formatNumber(fusionGate?.threshold)}`,
      tone: accepted ? "accept" : "review",
    },
    {
      index: "04",
      title: "User Output",
      value: accepted ? formatLabel(finalDecision?.user_visible_label) : "manual review",
      note: accepted ? "safe to show" : "prediction hidden",
      tone: accepted ? "accept" : "review",
    },
  ];

  function clearLiveSample() {
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
      text: `Selected ${file.name}. Run live inference to send it through the prototype pipeline.`,
    });
  }

  function loadScenario(name) {
    const scenario = DEMO_SCENARIOS[name];

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
      text: "Checking backend health for the live prototype flow...",
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
          "Health check failed. Make sure `ml.api.predict_api:app` is running.",
      });
    } finally {
      setIsCheckingHealth(false);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();

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
      text: "Uploading image and running the prototype...",
    });

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
        text: `Live prediction complete for ${selectedFile.name}.`,
      });
    } catch (error) {
      setStatusMessage({
        tone: "error",
        text:
          error.message ||
          "Live prediction failed. The built-in demo scenarios are still available for presentation.",
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="top-header panel">
        <div className="header-main">
          <div className="section-tag">OpenWaste-HR Research Prototype</div>
          <h1>Safety-aware waste classification demo</h1>
          <p>
            Upload a sample or replay a research case to show how the classifier output
            is filtered by the Fusion Gate before it reaches the user.
          </p>
        </div>

        <div className="header-stats">
          {RESEARCH_STATS.map((item) => (
            <CompactStat key={item.label} label={item.label} value={item.value} />
          ))}
        </div>
      </header>

      <section className="workspace-grid">
        <aside className="panel input-panel">
          <form className="input-form" onSubmit={handleSubmit}>
            <div
              className={`upload-box ${isDragging ? "dragging" : ""}`}
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
                <img className="preview-image" src={previewUrl} alt="Selected sample preview" />
              ) : (
                <div className="upload-copy">
                  <div className="upload-icon">+</div>
                  <h3>Upload image</h3>
                  <p>Click or drag a sample here for a live run.</p>
                </div>
              )}
            </div>

            {selectedFile ? (
              <div className="selected-file">
                <span>Selected file</span>
                <strong>{selectedFile.name}</strong>
              </div>
            ) : null}

            <div className={`status-banner tone-${statusMessage.tone}`} aria-live="polite">
              {statusMessage.text}
            </div>

            <div className="button-row">
              <button className="primary-button" type="submit" disabled={isSubmitting}>
                {isSubmitting ? "Running..." : "Run live inference"}
              </button>
              <button
                className="secondary-button"
                type="button"
                onClick={handleHealthCheck}
                disabled={isCheckingHealth}
              >
                {isCheckingHealth ? "Checking..." : "Check backend"}
              </button>
            </div>

            <button className="ghost-button" type="button" onClick={clearLiveSample}>
              Clear sample
            </button>
          </form>

          <div className="scenario-block">
            <div className="section-tag">Demo Cases</div>
            <div className="scenario-buttons">
              <button className="scenario-button accept" type="button" onClick={() => loadScenario("known")}>
                Accepted recyclable case
              </button>
              <button className="scenario-button review" type="button" onClick={() => loadScenario("review")}>
                Manual-review case
              </button>
            </div>
          </div>

          <div className="taxonomy-block">
            <div className="section-tag">Known Taxonomy</div>
            <div className="taxonomy-pills">
              {KNOWN_CLASSES.map((className) => (
                <span key={className}>{className}</span>
              ))}
            </div>
          </div>
        </aside>

        <section className="center-column">
          <div className="panel result-panel">
            <div className="result-top">
              <div className={`sample-card ${displayedResultImageUrl ? "has-image" : ""}`}>
                {displayedResultImageUrl ? (
                  <img
                    className="result-image"
                    src={displayedResultImageUrl}
                    alt="Evaluated waste sample preview"
                  />
                ) : (
                  <div className={`sample-placeholder ${accepted ? "accept" : "review"}`}>
                    <div className="sample-type">{resultSource === "live" ? "Live sample" : "Demo replay"}</div>
                    <strong>{sampleTitle}</strong>
                    <span>{formatLabel(prediction?.internal_top1_prediction)}</span>
                  </div>
                )}
              </div>

              <div className={`decision-card tone-${accepted ? "accept" : "review"}`}>
                <div className="decision-header">
                  <div>
                    <div className="section-tag">Final Decision</div>
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

                <div className="key-grid">
                  <KeyCard
                    label="Visible label"
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
                    label="Decision type"
                    value={formatLabel(finalDecision?.decision_type)}
                  />
                </div>
              </div>
            </div>

            <div className="probability-section">
              <div className="probability-heading">
                <div className="section-tag">Probability Distribution</div>
                <span>Internal known-class scores</span>
              </div>
              <ProbabilityBars probabilities={prediction?.class_probabilities} />
            </div>
          </div>

          <details className="panel raw-json">
            <summary>Technical details and raw JSON</summary>
            <pre>{JSON.stringify(resultPayload, null, 2)}</pre>
          </details>
        </section>

        <aside className="panel research-panel">
          <div className="section-tag">Research Flow</div>
          <div className="flow-list">
            {flowItems.map((item) => (
              <FlowStep
                key={item.index}
                index={item.index}
                title={item.title}
                value={item.value}
                note={item.note}
                tone={item.tone}
              />
            ))}
          </div>

          <div className="section-tag signals-tag">Gate Signals</div>
          <div className="signal-list">
            <SignalRow label="Fusion threshold" value={formatNumber(fusionGate?.threshold)} />
            <SignalRow
              label="Temperature confidence"
              value={formatPercent(prediction?.temperature_scaled_confidence)}
            />
            <SignalRow label="Raw confidence" value={formatPercent(prediction?.raw_confidence)} />
            <SignalRow label="Softmax margin" value={formatNumber(prediction?.softmax_margin)} />
            <SignalRow label="Energy" value={formatNumber(prediction?.energy)} />
            <SignalRow
              label="Mahalanobis"
              value={formatNumber(mahalanobis?.mahalanobis_knownness)}
            />
            <SignalRow
              label="Nearest class"
              value={formatLabel(mahalanobis?.mahalanobis_nearest_class)}
            />
          </div>

          <div className="narration-card">
            <div className="section-tag">Narration</div>
            <p>
              The classifier internally predicts{" "}
              <strong>{formatLabel(prediction?.internal_top1_prediction)}</strong>, then
              the Fusion Gate compares <strong>{formatNumber(fusionGate?.knownness_score)}</strong>{" "}
              against <strong>{formatNumber(fusionGate?.threshold)}</strong> before deciding
              whether the label is safe to show.
            </p>
          </div>
        </aside>
      </section>
    </main>
  );
}
