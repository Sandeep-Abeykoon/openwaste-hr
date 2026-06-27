import { startTransition, useEffect, useRef, useState } from "react";
import { DEMO_SCENARIOS, KNOWN_CLASSES } from "./demoScenarios.js";

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

const MANUAL_REVIEW_OPTIONS = [
  ...KNOWN_CLASSES,
  "biological",
  "textile",
  "mixed_material",
  "other_unknown",
];

const EMPTY_REVIEW_SUMMARY = {
  pending_count: 0,
  reviewed_count: 0,
  intelligence_count: 0,
  total_count: 0,
};

function cloneScenario(payload) {
  return JSON.parse(JSON.stringify(payload));
}

function normalizeApiBase() {
  return DEFAULT_API_BASE.trim().replace(/\/+$/, "");
}

function buildApiUrl(path) {
  const base = normalizeApiBase();
  return base ? `${base}${path}` : path;
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

function createReviewFormState(item) {
  const review = item?.review || {};

  return {
    selectedLabel: review.selected_label || "",
    customLabel: review.custom_label || "",
    reviewNotes: review.review_notes || "",
    promoteToIntelligence: Boolean(review.promote_to_intelligence),
  };
}

function getResolvedReviewLabel(item) {
  const review = item?.review || {};
  return review.review_label || review.custom_label || review.selected_label || "";
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

function TabButton({ active, label, onClick, badge }) {
  return (
    <button
      className={`tab-button ${active ? "active" : ""}`}
      type="button"
      onClick={onClick}
    >
      <span>{label}</span>
      {badge ? <strong>{badge}</strong> : null}
    </button>
  );
}

function ReviewQueueItem({ item, active, onSelect, onDelete, isDeleting }) {
  const reviewed = item.status === "reviewed";
  const reviewLabel = getResolvedReviewLabel(item);
  const displayName = item.uploaded_filename || item.review_id;

  return (
    <div
      className={`review-queue-item ${active ? "active" : ""} ${
        reviewed ? "reviewed" : "pending"
      }`}
    >
      <button
        className="review-queue-delete"
        type="button"
        aria-label={`Delete ${displayName} from the review queue`}
        title="Delete from queue"
        onClick={() => onDelete(item.review_id)}
        disabled={isDeleting}
      >
        {isDeleting ? "..." : "x"}
      </button>

      <button
        className="review-queue-select"
        type="button"
        onClick={onSelect}
      >
        <div className="review-queue-top">
          <strong title={displayName}>
            {truncateFileName(displayName, 28)}
          </strong>
          <span className={`mini-pill ${reviewed ? "tone-accept" : "tone-review"}`}>
            {reviewed ? "Reviewed" : "Pending"}
          </span>
        </div>
        <div className="review-queue-meta">
          <span>Internal guess</span>
          <strong>{formatLabel(item.summary?.internal_top1_prediction)}</strong>
        </div>
        <div className="review-queue-meta">
          <span>Knownness</span>
          <strong>{formatNumber(item.summary?.knownness_score)}</strong>
        </div>
        {reviewed ? (
          <div className="review-queue-footer">
            Final review: <strong>{formatLabel(reviewLabel)}</strong>
          </div>
        ) : null}
      </button>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("inference");
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
  const [manualReviewItems, setManualReviewItems] = useState([]);
  const [manualReviewSummary, setManualReviewSummary] = useState(
    EMPTY_REVIEW_SUMMARY
  );
  const [selectedReviewId, setSelectedReviewId] = useState("");
  const [reviewForm, setReviewForm] = useState(createReviewFormState(null));
  const [reviewStatusMessage, setReviewStatusMessage] = useState({
    tone: "info",
    text: "Rejected live samples will appear here for human review.",
  });
  const [isLoadingReviewQueue, setIsLoadingReviewQueue] = useState(false);
  const [isSubmittingReview, setIsSubmittingReview] = useState(false);
  const [deletingReviewId, setDeletingReviewId] = useState("");

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
  const pendingReviewCount = manualReviewSummary.pending_count || 0;
  const selectedReviewItem =
    manualReviewItems.find((item) => item.review_id === selectedReviewId) ||
    manualReviewItems[0] ||
    null;
  const selectedReviewResult = selectedReviewItem?.result || null;
  const selectedReviewPrediction = selectedReviewResult?.prediction || null;
  const selectedReviewFusionGate = selectedReviewResult?.fusion_gate || null;
  const selectedReviewFinalDecision =
    selectedReviewResult?.final_decision || null;
  const selectedReviewLabel = getResolvedReviewLabel(selectedReviewItem);
  const selectedReviewKnownness =
    selectedReviewFusionGate?.knownness_score ?? null;
  const selectedReviewThreshold = selectedReviewFusionGate?.threshold ?? null;
  const selectedReviewAccepted = Boolean(
    selectedReviewFinalDecision?.accepted_as_known
  );
  const selectedReviewMargin =
    typeof selectedReviewKnownness === "number" &&
    typeof selectedReviewThreshold === "number"
      ? selectedReviewKnownness - selectedReviewThreshold
      : null;
  const selectedReviewMarginLabel =
    selectedReviewMargin === null
      ? "margin unavailable"
      : `${formatNumber(Math.abs(selectedReviewMargin))} ${
          selectedReviewMargin >= 0 ? "above threshold" : "below threshold"
        }`;

  function clearLoadingStageTimers() {
    loadingStageTimersRef.current.forEach((timerId) => clearTimeout(timerId));
    loadingStageTimersRef.current = [];
  }

  function syncManualReviewSelection(
    nextItems,
    { preferredReviewId = "", replaceForm = false } = {}
  ) {
    const hasPreferredReviewId =
      preferredReviewId !== "" &&
      nextItems.some((item) => item.review_id === preferredReviewId);
    const hasCurrentReviewId =
      selectedReviewId !== "" &&
      nextItems.some((item) => item.review_id === selectedReviewId);
    const nextSelectedReviewId = hasPreferredReviewId
      ? preferredReviewId
      : hasCurrentReviewId
        ? selectedReviewId
        : nextItems[0]?.review_id || "";
    const nextSelectedItem =
      nextItems.find((item) => item.review_id === nextSelectedReviewId) || null;

    setSelectedReviewId(nextSelectedReviewId);

    if (
      replaceForm ||
      nextSelectedReviewId !== selectedReviewId ||
      selectedReviewId === ""
    ) {
      setReviewForm(createReviewFormState(nextSelectedItem));
    }
  }

  async function refreshManualReviewQueue({
    preferredReviewId = "",
    replaceForm = false,
    silent = false,
  } = {}) {
    const normalizedBase = normalizeApiBase();

    if (!normalizedBase) {
      setReviewStatusMessage({
        tone: "error",
        text: "Backend URL is not configured for manual review.",
      });
      return;
    }

    setIsLoadingReviewQueue(true);

    if (!silent) {
      setReviewStatusMessage({
        tone: "info",
        text: "Loading manual review queue...",
      });
    }

    try {
      const response = await fetch(`${normalizedBase}/api/manual-review`);
      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, response.status));
      }

      const nextItems = Array.isArray(payload.items) ? payload.items : [];
      const nextSummary = payload.summary || EMPTY_REVIEW_SUMMARY;

      setManualReviewItems(nextItems);
      setManualReviewSummary(nextSummary);
      syncManualReviewSelection(nextItems, {
        preferredReviewId,
        replaceForm,
      });

      if (!silent) {
        setReviewStatusMessage({
          tone: "success",
          text:
            nextItems.length > 0
              ? `Loaded ${nextSummary.total_count} review item${
                  nextSummary.total_count === 1 ? "" : "s"
                }.`
              : "No manual review items yet. Rejected live samples will appear here.",
        });
      }
    } catch (error) {
      setReviewStatusMessage({
        tone: "error",
        text:
          error.message ||
          "Could not load the manual review queue. Check that the backend is running.",
      });
    } finally {
      setIsLoadingReviewQueue(false);
    }
  }

  useEffect(() => {
    if (activeTab === "review") {
      void refreshManualReviewQueue({
        replaceForm: manualReviewItems.length === 0,
        silent: manualReviewItems.length > 0,
      });
    }
  }, [activeTab]);

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
      setActiveTab("inference");
    });

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  function selectReviewItem(item) {
    setSelectedReviewId(item.review_id);
    setReviewForm(createReviewFormState(item));
  }

  function updateReviewForm(field, value) {
    setReviewForm((current) => ({
      ...current,
      [field]: value,
    }));
  }

  async function handleHealthCheck() {
    clearLoadingStageTimers();
    const normalizedBase = normalizeApiBase();

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

    const normalizedBase = normalizeApiBase();

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

      if (payload?.manual_review_entry) {
        const queueStatus = payload.manual_review_entry_status;
        const alreadyQueued = queueStatus === "existing";

        setStatusMessage({
          tone: "success",
          text: alreadyQueued
            ? "Live prediction complete. This blocked filename was already in the manual review queue, so it was not added again."
            : "Live prediction complete. The sample was blocked and added to the manual review queue.",
        });
        setReviewStatusMessage({
          tone: "success",
          text: alreadyQueued
            ? "That filename was already queued, so the existing review item was kept."
            : "A new blocked sample has been added to manual review.",
        });
        await refreshManualReviewQueue({
          preferredReviewId: payload.manual_review_entry.review_id,
          replaceForm: false,
          silent: true,
        });
      } else {
        setStatusMessage({
          tone: "success",
          text: "Live prediction complete. Review the final decision and gate result.",
        });
      }
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

  async function handleDeleteReview(reviewId) {
    const normalizedBase = normalizeApiBase();

    if (!normalizedBase) {
      setReviewStatusMessage({
        tone: "error",
        text: "Backend URL is not configured for manual review.",
      });
      return;
    }

    setDeletingReviewId(reviewId);
    setReviewStatusMessage({
      tone: "info",
      text: "Deleting review item from the queue...",
    });

    try {
      const response = await fetch(
        `${normalizedBase}/api/manual-review/${reviewId}`,
        {
          method: "DELETE",
        }
      );
      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, response.status));
      }

      const nextItems = manualReviewItems.filter(
        (item) => item.review_id !== reviewId
      );

      setManualReviewItems(nextItems);
      setManualReviewSummary(payload.summary || EMPTY_REVIEW_SUMMARY);
      syncManualReviewSelection(nextItems, {
        replaceForm: true,
      });
      setReviewStatusMessage({
        tone: "success",
        text: payload.deleted_image_from_system
          ? "Review item removed and its stored image was deleted from the system."
          : "Review item removed from the queue.",
      });
    } catch (error) {
      setReviewStatusMessage({
        tone: "error",
        text:
          error.message ||
          "Could not delete the review item. Please try again.",
      });
    } finally {
      setDeletingReviewId("");
    }
  }

  async function handleSubmitReview(event) {
    event.preventDefault();

    const normalizedBase = normalizeApiBase();

    if (!normalizedBase) {
      setReviewStatusMessage({
        tone: "error",
        text: "Backend URL is not configured for manual review.",
      });
      return;
    }

    if (!selectedReviewItem) {
      setReviewStatusMessage({
        tone: "error",
        text: "Choose a manual review item first.",
      });
      return;
    }

    if (
      reviewForm.selectedLabel.trim() === "" &&
      reviewForm.customLabel.trim() === ""
    ) {
      setReviewStatusMessage({
        tone: "error",
        text: "Choose a review label or type a custom one before saving.",
      });
      return;
    }

    setIsSubmittingReview(true);
    setReviewStatusMessage({
      tone: "info",
      text: "Saving human review decision...",
    });

    try {
      const response = await fetch(
        `${normalizedBase}/api/manual-review/${selectedReviewItem.review_id}/decision`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            selected_label: reviewForm.selectedLabel,
            custom_label: reviewForm.customLabel,
            review_notes: reviewForm.reviewNotes,
            promote_to_intelligence: reviewForm.promoteToIntelligence,
          }),
        }
      );

      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, response.status));
      }

      const updatedItem = payload.item;
      const nextItems = manualReviewItems.map((item) =>
        item.review_id === updatedItem.review_id ? updatedItem : item
      );

      setManualReviewItems(nextItems);
      setManualReviewSummary(payload.summary || EMPTY_REVIEW_SUMMARY);
      setSelectedReviewId(updatedItem.review_id);
      setReviewForm(createReviewFormState(updatedItem));
      setReviewStatusMessage({
        tone: "success",
        text: reviewForm.promoteToIntelligence
          ? "Review saved and marked for future intelligence improvements."
          : "Review saved. The item stays in the reviewed queue.",
      });
    } catch (error) {
      setReviewStatusMessage({
        tone: "error",
        text:
          error.message ||
          "Could not save the review decision. Please try again.",
      });
    } finally {
      setIsSubmittingReview(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="top-header panel simple-header">
        <div className="header-main">
          <div className="section-tag">OpenWaste-HR Prototype</div>
          <h1>Safety-aware waste classification</h1>
          <p>
            Upload one sample, see how the Fusion Gate blocks risky outputs,
            and review rejected images in a separate intelligence workflow.
          </p>
        </div>
      </header>

      <div className="panel tab-strip">
        <TabButton
          active={activeTab === "inference"}
          label="Live Inference"
          onClick={() => setActiveTab("inference")}
        />
        <TabButton
          active={activeTab === "review"}
          label="Manual Review"
          badge={pendingReviewCount > 0 ? String(pendingReviewCount) : ""}
          onClick={() => setActiveTab("review")}
        />
      </div>

      {activeTab === "inference" ? (
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

            <div className="review-handoff-card">
              <div className="section-tag">Manual Review Queue</div>
              <p>
                Rejected live samples are stored for human review and can be
                flagged for future intelligence improvements.
              </p>
              <div className="key-grid simple-key-grid review-summary-grid">
                <KeyCard
                  label="Pending"
                  value={String(manualReviewSummary.pending_count || 0)}
                  tone="review"
                />
                <KeyCard
                  label="Reviewed"
                  value={String(manualReviewSummary.reviewed_count || 0)}
                  tone="accept"
                />
              </div>
              <button
                className="secondary-button"
                type="button"
                onClick={() => setActiveTab("review")}
              >
                Open manual review
              </button>
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

                <p>
                  {finalDecision?.user_message || "No user-facing message returned."}
                </p>

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
      ) : (
        <section className="workspace-grid simple-workspace review-workspace">
          <aside className="panel input-panel simple-input-panel review-sidebar">
            <div className="review-queue-header">
              <div className="section-tag">Review Queue</div>
              <p>
                Work through rejected samples one by one, confirm a label, and
                decide whether the example should improve future intelligence.
              </p>
            </div>

            <div className="key-grid simple-key-grid review-summary-grid">
              <KeyCard
                label="Pending"
                value={String(manualReviewSummary.pending_count || 0)}
                tone="review"
              />
              <KeyCard
                label="Reviewed"
                value={String(manualReviewSummary.reviewed_count || 0)}
                tone="accept"
              />
              <KeyCard
                label="For intelligence"
                value={String(manualReviewSummary.intelligence_count || 0)}
                tone="live"
              />
              <KeyCard
                label="Total"
                value={String(manualReviewSummary.total_count || 0)}
              />
            </div>

            <div
              className={`upload-summary tone-${reviewStatusMessage.tone}`}
              aria-live="polite"
            >
              <div className="upload-summary-head">
                <span>Review status</span>
              </div>
              <p>{reviewStatusMessage.text}</p>
            </div>

            <div className="utility-row review-utility-row">
              <button
                className="secondary-button support-button"
                type="button"
                onClick={() => {
                  void refreshManualReviewQueue({ replaceForm: false });
                }}
                disabled={isLoadingReviewQueue}
              >
                {isLoadingReviewQueue ? "Refreshing..." : "Refresh queue"}
              </button>
              <button
                className="ghost-button"
                type="button"
                onClick={() => setActiveTab("inference")}
              >
                Back to inference
              </button>
            </div>

            <div className="review-queue-list">
              {manualReviewItems.length > 0 ? (
                manualReviewItems.map((item) => (
                  <ReviewQueueItem
                    key={item.review_id}
                    item={item}
                    active={item.review_id === selectedReviewItem?.review_id}
                    onSelect={() => selectReviewItem(item)}
                    onDelete={handleDeleteReview}
                    isDeleting={deletingReviewId === item.review_id}
                  />
                ))
              ) : (
                <div className="empty-review-state">
                  <strong>No review items yet</strong>
                  <p>
                    Run live inference on an uncertain image and blocked samples
                    will show up here automatically.
                  </p>
                </div>
              )}
            </div>
          </aside>

          <section className="center-column">
            {selectedReviewItem ? (
              <>
                <div className="panel result-panel simple-result-panel">
                  <div className="decision-card tone-review">
                    <div className="decision-header">
                      <div>
                        <div className="decision-kicker">
                          <div className="section-tag">Human Review</div>
                          <div
                            className={`outcome-pill tone-${
                              selectedReviewItem.status === "reviewed"
                                ? "accept"
                                : "review"
                            }`}
                          >
                            {selectedReviewItem.status === "reviewed"
                              ? "Reviewed"
                              : "Needs review"}
                          </div>
                        </div>
                        <h2>
                          {selectedReviewItem.status === "reviewed"
                            ? formatLabel(selectedReviewLabel)
                            : "Pending human decision"}
                        </h2>
                      </div>

                      <div className="source-pill tone-live">Queued</div>
                    </div>

                    <p>
                      {selectedReviewFinalDecision?.user_message ||
                        "This sample was routed to manual review by the Fusion Gate."}
                    </p>

                    <div className="result-main-grid">
                      <div className="sample-card result-sample-card">
                        <div className="sample-frame">
                          <img
                            className="result-image"
                            src={buildApiUrl(selectedReviewItem.image_url)}
                            alt="Manual review sample"
                          />
                          <div className="image-caption">Queued sample</div>
                        </div>
                      </div>

                      <div className="result-main-stack">
                        <GateMeter
                          knownness={selectedReviewKnownness}
                          threshold={selectedReviewThreshold}
                          accepted={selectedReviewAccepted}
                          marginLabel={selectedReviewMarginLabel}
                        />

                        <div className="score-block">
                          <div className="section-tag">Review Context</div>
                          <div className="key-grid simple-key-grid">
                            <KeyCard
                              label="Uploaded file"
                              value={truncateFileName(
                                selectedReviewItem.uploaded_filename,
                                30
                              )}
                            />
                            <KeyCard
                              label="Internal guess"
                              value={formatLabel(
                                selectedReviewPrediction?.internal_top1_prediction
                              )}
                            />
                            <KeyCard
                              label="Knownness"
                              value={formatNumber(selectedReviewKnownness)}
                              tone="review"
                            />
                            <KeyCard
                              label="Threshold"
                              value={formatNumber(selectedReviewThreshold)}
                            />
                            <KeyCard
                              label="Current review"
                              value={
                                selectedReviewItem.status === "reviewed"
                                  ? formatLabel(selectedReviewLabel)
                                  : "Not reviewed yet"
                              }
                              tone={
                                selectedReviewItem.status === "reviewed"
                                  ? "accept"
                                  : "review"
                              }
                            />
                            <KeyCard
                              label="Add to intelligence"
                              value={
                                selectedReviewItem.review?.promote_to_intelligence
                                  ? "Yes"
                                  : "Not yet"
                              }
                              tone={
                                selectedReviewItem.review?.promote_to_intelligence
                                  ? "live"
                                  : "default"
                              }
                            />
                          </div>
                        </div>

                        <form className="review-form" onSubmit={handleSubmitReview}>
                          <div className="review-form-header">
                            <div className="section-tag">Review Controls</div>
                            <p>
                              Pick a supported label, type a better label, and
                              decide whether this sample should help future model
                              improvement.
                            </p>
                          </div>

                          <div className="review-form-grid">
                            <label className="field">
                              <span>Suggested label</span>
                              <select
                                value={reviewForm.selectedLabel}
                                onChange={(event) =>
                                  updateReviewForm(
                                    "selectedLabel",
                                    event.target.value
                                  )
                                }
                              >
                                <option value="">Choose a label</option>
                                {MANUAL_REVIEW_OPTIONS.map((option) => (
                                  <option key={option} value={option}>
                                    {formatLabel(option)}
                                  </option>
                                ))}
                              </select>
                            </label>

                            <label className="field">
                              <span>Custom label</span>
                              <input
                                type="text"
                                value={reviewForm.customLabel}
                                placeholder="Type a more specific label if needed"
                                onChange={(event) =>
                                  updateReviewForm(
                                    "customLabel",
                                    event.target.value
                                  )
                                }
                              />
                            </label>
                          </div>

                          <label className="field">
                            <span>Reviewer notes</span>
                            <textarea
                              rows="4"
                              value={reviewForm.reviewNotes}
                              placeholder="Record why you picked this label or what makes the sample ambiguous."
                              onChange={(event) =>
                                updateReviewForm(
                                  "reviewNotes",
                                  event.target.value
                                )
                              }
                            />
                          </label>

                          <label className="review-checkbox">
                            <input
                              type="checkbox"
                              checked={reviewForm.promoteToIntelligence}
                              onChange={(event) =>
                                updateReviewForm(
                                  "promoteToIntelligence",
                                  event.target.checked
                                )
                              }
                            />
                            <span>
                              Add this reviewed sample to the intelligence
                              backlog for future retraining or dataset review.
                            </span>
                          </label>

                          <div className="utility-row review-utility-row">
                            <button
                              className="primary-button"
                              type="submit"
                              disabled={isSubmittingReview}
                            >
                              {isSubmittingReview
                                ? "Saving..."
                                : selectedReviewItem.status === "reviewed"
                                  ? "Update review"
                                  : "Save review"}
                            </button>
                            <button
                              className="ghost-button"
                              type="button"
                              onClick={() =>
                                setReviewForm(createReviewFormState(selectedReviewItem))
                              }
                            >
                              Reset form
                            </button>
                          </div>
                        </form>

                        <div className="probability-section inline-probability">
                          <div className="probability-heading">
                            <div className="section-tag">Internal Prediction Scores</div>
                            <span>
                              Hidden from the end user, but available to the reviewer
                            </span>
                          </div>
                          <ProbabilityBars
                            probabilities={
                              selectedReviewPrediction?.class_probabilities
                            }
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <details className="panel raw-json simple-details">
                  <summary>Queued item details</summary>
                  <div className="details-body">
                    <pre>{JSON.stringify(selectedReviewItem, null, 2)}</pre>
                  </div>
                </details>
              </>
            ) : (
              <div className="panel result-panel simple-result-panel empty-review-panel">
                <div className="decision-card tone-review">
                  <div className="section-tag">Manual Review</div>
                  <h2>No queued sample selected</h2>
                  <p>
                    Once the backend blocks a live sample, it will appear here
                    with an image preview, reviewer controls, and an option to
                    add the result to future intelligence work.
                  </p>
                </div>
              </div>
            )}
          </section>
        </section>
      )}
    </main>
  );
}
