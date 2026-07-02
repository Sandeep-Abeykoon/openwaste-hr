import { startTransition, useEffect, useRef, useState } from "react";
import {
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

const LIVE_IMAGE_PICKER_OPTIONS = {
  id: "openwaste-live-inference-image",
  multiple: false,
  excludeAcceptAllOption: false,
  types: [
    {
      description: "Image files",
      accept: {
        "image/*": SUPPORTED_IMAGE_EXTENSIONS,
      },
    },
  ],
};

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

const TECHNICAL_TERM_DETAILS = {
  AUROC:
    "Area under the ROC curve. It measures how well the system separates known samples from unknown samples across decision thresholds.",
  "Known Coverage":
    "Share of true known recyclable samples that the safety gate still allows through instead of sending to manual review.",
  "Unknown Rejection":
    "Share of unsupported or risky samples correctly blocked and sent to manual review.",
  "False Acceptance":
    "Share of unknown samples that were incorrectly accepted as if they were known recyclable items.",
  "Accepted-Known Accuracy":
    "Accuracy measured only on the known samples that the safety gate accepted.",
  "Fusion Gate ECE":
    "Expected calibration error for the final gate score. Lower values mean the score better matches real-world outcomes.",
  "Brier Score":
    "Probability-quality score for the final gate output. Lower values indicate better calibrated, more reliable probabilities.",
  "pAUC FAR <= 0.05":
    "Partial AUROC focused only on the low false-acceptance region up to a false acceptance rate of 5 percent.",
  "pAUC FAR <= 0.10":
    "Partial AUROC focused only on the low false-acceptance region up to a false acceptance rate of 10 percent.",
  Knownness:
    "Fusion Gate knownness score from 0 to 1. Higher values mean the sample looks more like the supported known classes.",
  Threshold:
    "Decision cutoff used by the gate. Scores below this are routed to manual review, while scores at or above it can be accepted.",
  "Fusion threshold":
    "Configured cutoff used by the final Fusion Gate policy to accept or reject a sample.",
  "Temp confidence":
    "Top-class confidence after temperature scaling calibration. It is easier to interpret than the raw confidence and is used as one of the gate inputs.",
  "Raw confidence":
    "Original top-class classifier confidence before temperature scaling calibration is applied.",
  Temperature:
    "Calibration parameter used to soften or sharpen confidence scores before they are displayed or fed into the gate.",
  "Max logit":
    "Highest raw pre-softmax classifier score before probabilities are computed.",
  Energy:
    "Logit-derived uncertainty signal used as one of the Fusion Gate inputs for open-set style decisions.",
  "Softmax margin":
    "Difference between the top predicted probability and the second-best probability. Larger margins usually mean clearer separation.",
  "Softmax entropy":
    "Uncertainty of the full class-probability distribution. Lower entropy usually means the model is more concentrated on one class.",
  "Mahalanobis knownness":
    "Feature-space knownness signal derived from Mahalanobis distance. It helps the system judge whether an embedding looks like known training data.",
  "Min distance":
    "Smallest Mahalanobis distance from the sample embedding to a known class distribution. Smaller values mean the sample is closer to known data.",
  "Nearest class":
    "Known class whose feature distribution is closest to this sample in embedding space.",
  "Decision type":
    "Internal decision code describing whether the system accepted a known label or routed the sample to manual review.",
  "Embedding layer":
    "Neural-network layer from which the feature vector was extracted for Mahalanobis scoring.",
  "Embedding dimension":
    "Number of numeric values in the extracted feature vector used for feature-space comparisons.",
};

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

function formatLabelList(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return "--";
  }

  return values.map((value) => formatLabel(value)).join(", ");
}

function formatTimestamp(value) {
  if (!value) {
    return "--";
  }

  const parsedDate = new Date(value);

  if (Number.isNaN(parsedDate.getTime())) {
    return String(value);
  }

  return parsedDate.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

function formatReviewNotes(value) {
  const normalizedValue = String(value || "").trim();
  return normalizedValue || "No reviewer notes yet.";
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

function getTechnicalTermDetail(label) {
  return TECHNICAL_TERM_DETAILS[label] || "";
}

function LabelWithInfo({ label, detail = "" }) {
  const resolvedDetail = detail || getTechnicalTermDetail(label);

  return (
    <span className="label-with-info">
      <span className="label-with-info-text">{label}</span>
      {resolvedDetail ? (
        <abbr
          className="info-icon"
          title={resolvedDetail}
          aria-label={`${label}: ${resolvedDetail}`}
        >
          i
        </abbr>
      ) : null}
    </span>
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
      <span>
        <LabelWithInfo label={label} />
      </span>
      <strong>{value}</strong>
    </div>
  );
}

function ResearchMetricCard({ label, value, description }) {
  return (
    <div className="research-metric-card">
      <span>
        <LabelWithInfo label={label} detail={description} />
      </span>
      <strong>{value}</strong>
      <p>{description}</p>
    </div>
  );
}

function ContextSection({ title, rows }) {
  return (
    <div className="score-block">
      <div className="section-tag">{title}</div>
      <div className="context-card">
        {rows.map((row) => (
          <div className="context-row" key={row.label}>
            <span>
              <LabelWithInfo label={row.label} detail={row.detail} />
            </span>
            <strong>{row.value}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function SystemInfoSection({ result }) {
  const fusionGate = result?.fusion_gate || null;

  return (
    <ContextSection
      title="System Info"
      rows={[
        {
          label: "Policy version",
          value: result?.policy_version || "--",
        },
        {
          label: "Device",
          value: result?.device || "--",
        },
        {
          label: "Known classes",
          value: formatLabelList(result?.known_classes),
        },
        {
          label: "Fusion threshold",
          value: formatNumber(fusionGate?.threshold),
        },
        {
          label: "Temperature",
          value: formatNumber(result?.temperature),
        },
      ]}
    />
  );
}

function AuditTrailSection({ item }) {
  return (
    <ContextSection
      title="Audit Trail"
      rows={[
        {
          label: "Queued at",
          value: formatTimestamp(item?.created_at),
        },
        {
          label: "Reviewed at",
          value: formatTimestamp(item?.review?.reviewed_at),
        },
        {
          label: "Review notes",
          value: formatReviewNotes(item?.review?.review_notes),
        },
      ]}
    />
  );
}

function AdvancedEvidenceSection({
  result,
  summaryLabel = "Advanced Evidence",
}) {
  const prediction = result?.prediction || null;
  const mahalanobis = result?.mahalanobis || null;
  const fusionGate = result?.fusion_gate || null;
  const finalDecision = result?.final_decision || null;
  const signalRows = [
    {
      label: "Max logit",
      value: formatNumber(prediction?.max_logit),
    },
    {
      label: "Energy",
      value: formatNumber(prediction?.energy),
    },
    {
      label: "Softmax margin",
      value: formatNumber(prediction?.softmax_margin),
    },
    {
      label: "Softmax entropy",
      value: formatNumber(prediction?.softmax_entropy),
    },
    {
      label: "Mahalanobis knownness",
      value: formatNumber(mahalanobis?.mahalanobis_knownness),
    },
    {
      label: "Min distance",
      value: formatNumber(mahalanobis?.mahalanobis_min_distance),
    },
  ];
  const traceRows = [
    {
      label: "Nearest class",
      value: formatLabel(mahalanobis?.mahalanobis_nearest_class),
    },
    {
      label: "Decision type",
      value: formatLabel(
        finalDecision?.decision_type || fusionGate?.decision_type
      ),
    },
    {
      label: "Embedding layer",
      value: result?.embedding_layer || "--",
    },
    {
      label: "Embedding dimension",
      value:
        typeof result?.embedding_dimension === "number"
          ? String(result.embedding_dimension)
          : "--",
    },
  ];

  return (
    <details className="panel raw-json simple-details technical-details">
      <summary>{summaryLabel}</summary>
      <div className="details-body technical-details-body">
        <div className="technical-grid">
          <div className="score-block">
                <div className="section-tag">Score Signals</div>
                <div className="signal-list">
                  {signalRows.map((row) => (
                    <div className="signal-row" key={row.label}>
                      <span>
                        <LabelWithInfo label={row.label} detail={row.detail} />
                      </span>
                      <strong>{row.value}</strong>
                    </div>
                  ))}
                </div>
          </div>

          <div className="score-block">
                <div className="section-tag">Model Trace</div>
                <div className="signal-list">
                  {traceRows.map((row) => (
                    <div className="signal-row" key={row.label}>
                      <span>
                        <LabelWithInfo label={row.label} detail={row.detail} />
                      </span>
                      <strong>{row.value}</strong>
                    </div>
                  ))}
                </div>
          </div>
        </div>
      </div>
    </details>
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
          <LabelWithInfo label="Knownness" /> {formatNumber(knownness)}
        </span>
        <span className="gate-meter-chip">
          <span className="gate-meter-dot threshold" />
          <LabelWithInfo label="Threshold" /> {formatNumber(threshold)}
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
  const thumbnailLabel = reviewed ? "Reviewed sample" : "Queued sample";

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
        <div className="review-queue-thumb-frame">
          <img
            className="review-queue-thumb"
            src={buildApiUrl(item.image_url)}
            alt={displayName}
          />
          <div className="review-queue-thumb-label">{thumbnailLabel}</div>
        </div>

        <div className="review-queue-top">
          <strong title={displayName}>
            {truncateFileName(displayName, 28)}
          </strong>
          <span className={`mini-pill ${reviewed ? "tone-accept" : "tone-review"}`}>
            {reviewed ? "Reviewed" : "Pending"}
          </span>
        </div>
        <div className="review-queue-meta">
          <span>
            <LabelWithInfo label="Internal guess" />
          </span>
          <strong>{formatLabel(item.summary?.internal_top1_prediction)}</strong>
        </div>
        <div className="review-queue-meta">
          <span>
            <LabelWithInfo label="Knownness" />
          </span>
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

function ReviewQueueSection({
  title,
  description,
  items,
  selectedReviewId,
  deletingReviewId,
  onSelect,
  onDelete,
  emptyMessage,
  showHeader = true,
}) {
  return (
    <section className="review-gallery-section">
      {showHeader ? (
        <div className="review-gallery-header">
          <div className="section-tag">{title}</div>
          <p>{description}</p>
        </div>
      ) : null}

      <div className="review-queue-list">
        {items.length > 0 ? (
          items.map((item) => (
            <ReviewQueueItem
              key={item.review_id}
              item={item}
              active={item.review_id === selectedReviewId}
              onSelect={() => onSelect(item)}
              onDelete={onDelete}
              isDeleting={deletingReviewId === item.review_id}
            />
          ))
        ) : (
          <div className="empty-review-state compact">
            <strong>{emptyMessage}</strong>
          </div>
        )}
      </div>
    </section>
  );
}

function IntelligenceGalleryItem({
  item,
  active,
  onSelect,
  onDelete,
  isDeleting,
}) {
  const displayName = item.uploaded_filename || item.review_id;
  const reviewLabel = getResolvedReviewLabel(item);

  return (
    <div className={`intelligence-card ${active ? "active" : ""}`}>
      <button
        className="review-queue-delete"
        type="button"
        aria-label={`Delete ${displayName} from the intelligence list`}
        title="Delete from intelligence list"
        onClick={() => onDelete(item.review_id)}
        disabled={isDeleting}
      >
        {isDeleting ? "..." : "x"}
      </button>

      <button
        className="intelligence-card-button"
        type="button"
        onClick={onSelect}
      >
        <div className="intelligence-card-frame">
          <img
            className="intelligence-card-image"
            src={buildApiUrl(item.image_url)}
            alt={displayName}
          />
          <div className="review-queue-thumb-label">Intelligence sample</div>
        </div>

        <div className="intelligence-card-body">
          <strong title={displayName}>
            {truncateFileName(displayName, 24)}
          </strong>
          <div className="intelligence-card-meta">
            <span>
              <LabelWithInfo label="Current review" />
            </span>
            <strong>{formatLabel(reviewLabel)}</strong>
          </div>
          <div className="intelligence-card-meta">
            <span>
              <LabelWithInfo label="Guess" />
            </span>
            <strong>
              {formatLabel(item.summary?.internal_top1_prediction)}
            </strong>
          </div>
        </div>
      </button>
    </div>
  );
}

function IntelligenceGallerySection({
  title,
  description,
  items,
  selectedReviewId,
  deletingReviewId,
  onSelect,
  onDelete,
  emptyMessage,
  showHeader = true,
}) {
  return (
    <section className="review-gallery-section intelligence-gallery-section">
      {showHeader ? (
        <div className="review-gallery-header">
          <div className="section-tag">{title}</div>
          <p>{description}</p>
        </div>
      ) : null}

      <div className="intelligence-grid">
        {items.length > 0 ? (
          items.map((item) => (
            <IntelligenceGalleryItem
              key={item.review_id}
              item={item}
              active={item.review_id === selectedReviewId}
              onSelect={() => onSelect(item)}
              onDelete={onDelete}
              isDeleting={deletingReviewId === item.review_id}
            />
          ))
        ) : (
          <div className="empty-review-state compact">
            <strong>{emptyMessage}</strong>
          </div>
        )}
      </div>
    </section>
  );
}

function getItemsForTab(tabName, items) {
  if (tabName === "review") {
    return items.filter((item) => !item.review?.promote_to_intelligence);
  }

  if (tabName === "intelligence") {
    return items.filter((item) => Boolean(item.review?.promote_to_intelligence));
  }

  return items;
}

function ReviewDetailPanel({
  item,
  reviewForm,
  onUpdateReviewForm,
  onSubmitReview,
  onResetReviewForm,
  isSubmittingReview,
  sourceLabel,
  emptyTitle,
  emptyDescription,
}) {
  if (!item) {
    return (
      <div className="panel result-panel simple-result-panel empty-review-panel">
        <div className="decision-card tone-review">
          <div className="section-tag">{sourceLabel}</div>
          <h2>{emptyTitle}</h2>
          <p>{emptyDescription}</p>
        </div>
      </div>
    );
  }

  const result = item.result || null;
  const prediction = result?.prediction || null;
  const fusionGate = result?.fusion_gate || null;
  const finalDecision = result?.final_decision || null;
  const reviewLabel = getResolvedReviewLabel(item);
  const knownness = fusionGate?.knownness_score ?? null;
  const threshold = fusionGate?.threshold ?? null;
  const accepted = Boolean(finalDecision?.accepted_as_known);
  const reviewMargin =
    typeof knownness === "number" && typeof threshold === "number"
      ? knownness - threshold
      : null;
  const reviewMarginLabel =
    reviewMargin === null
      ? "margin unavailable"
      : `${formatNumber(Math.abs(reviewMargin))} ${
          reviewMargin >= 0 ? "above threshold" : "below threshold"
        }`;

  return (
    <>
      <div className="panel result-panel simple-result-panel">
        <div className="decision-card tone-review">
          <div className="decision-header">
            <div>
              <div className="decision-kicker">
                <div className="section-tag">Human Review</div>
                <div
                  className={`outcome-pill tone-${
                    item.status === "reviewed" ? "accept" : "review"
                  }`}
                >
                  {item.status === "reviewed" ? "Reviewed" : "Needs review"}
                </div>
              </div>
              <h2>
                {item.status === "reviewed"
                  ? formatLabel(reviewLabel)
                  : "Pending human decision"}
              </h2>
            </div>

            <div className="source-pill tone-live">{sourceLabel}</div>
          </div>

          <p>
            {finalDecision?.user_message ||
              "This sample was routed to manual review by the Fusion Gate."}
          </p>

          <div className="result-main-grid">
            <div className="sample-card result-sample-card">
              <div className="sample-frame">
                <img
                  className="result-image"
                  src={buildApiUrl(item.image_url)}
                  alt="Manual review sample"
                />
                <div className="image-caption">
                  {sourceLabel === "Intelligence" ? "Intelligence sample" : "Queued sample"}
                </div>
              </div>
            </div>

            <div className="result-main-stack">
              <GateMeter
                knownness={knownness}
                threshold={threshold}
                accepted={accepted}
                marginLabel={reviewMarginLabel}
              />

              <div className="score-block">
                <div className="section-tag">Review Context</div>
                <div className="key-grid simple-key-grid">
                  <KeyCard
                    label="Uploaded file"
                    value={truncateFileName(item.uploaded_filename, 30)}
                  />
                  <KeyCard
                    label="Internal guess"
                    value={formatLabel(prediction?.internal_top1_prediction)}
                  />
                  <KeyCard
                    label="Knownness"
                    value={formatNumber(knownness)}
                    tone="review"
                  />
                  <KeyCard
                    label="Threshold"
                    value={formatNumber(threshold)}
                  />
                  <KeyCard
                    label="Current review"
                    value={
                      item.status === "reviewed"
                        ? formatLabel(reviewLabel)
                        : "Not reviewed yet"
                    }
                    tone={item.status === "reviewed" ? "accept" : "review"}
                  />
                  <KeyCard
                    label="Add to intelligence"
                    value={
                      item.review?.promote_to_intelligence ? "Yes" : "Not yet"
                    }
                    tone={item.review?.promote_to_intelligence ? "live" : "default"}
                  />
                </div>
              </div>

              <AuditTrailSection item={item} />

              <SystemInfoSection result={result} />

              <form className="review-form" onSubmit={onSubmitReview}>
                <div className="review-form-header">
                  <div className="section-tag">Review Controls</div>
                  <p>
                    Pick a supported label, type a better label, and decide
                    whether this sample should help future model improvement.
                  </p>
                </div>

                <div className="review-form-grid">
                  <label className="field">
                    <span>Suggested label</span>
                    <select
                      value={reviewForm.selectedLabel}
                      onChange={(event) =>
                        onUpdateReviewForm("selectedLabel", event.target.value)
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
                        onUpdateReviewForm("customLabel", event.target.value)
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
                      onUpdateReviewForm("reviewNotes", event.target.value)
                    }
                  />
                </label>

                <label className="review-checkbox">
                  <input
                    type="checkbox"
                    checked={reviewForm.promoteToIntelligence}
                    onChange={(event) =>
                      onUpdateReviewForm(
                        "promoteToIntelligence",
                        event.target.checked
                      )
                    }
                  />
                  <span>
                    Add this reviewed sample to the intelligence backlog for
                    future retraining or dataset review.
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
                      : item.status === "reviewed"
                        ? "Update review"
                        : "Save review"}
                  </button>
                  <button
                    className="ghost-button"
                    type="button"
                    onClick={() => onResetReviewForm(item)}
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
                <ProbabilityBars probabilities={prediction?.class_probabilities} />
              </div>

              <AdvancedEvidenceSection
                result={result}
                summaryLabel="Advanced evidence"
              />
            </div>
          </div>
        </div>
      </div>

      <details className="panel raw-json simple-details">
        <summary>Queued item details</summary>
        <div className="details-body">
          <pre>{JSON.stringify(item, null, 2)}</pre>
        </div>
      </details>
    </>
  );
}

function ResearchSummaryTab({
  pendingReviewCount,
  intelligenceCount,
}) {
  return (
    <section className="research-summary-layout">
      <div className="panel research-panel">
        <div className="research-intro">
          <div className="section-tag">Research Summary</div>
          <h2>Fusion Gate v2 + Mahalanobis</h2>
          <p>
            This demo shows the final OpenWaste-HR safety policy: a known-class
            classifier, temperature-scaled confidence, Mahalanobis
            feature-space evidence, and a Fusion Gate that decides whether the
            system should answer or defer to human review.
          </p>
        </div>

        <div className="research-metric-grid">
          {RESEARCH_STATS.map((stat) => (
            <ResearchMetricCard
              key={stat.label}
              label={stat.label}
              value={stat.value}
              description={stat.description}
            />
          ))}
        </div>
      </div>

      <div className="research-summary-grid">
        <div className="panel research-panel">
          <div className="section-tag">Demo Flow</div>
          <div className="research-flow-list">
            <div className="research-flow-card">
              <strong>1. Classifier makes an internal guess</strong>
              <p>
                The model predicts among {formatLabelList(KNOWN_CLASSES)} and
                produces confidence and score-based signals.
              </p>
            </div>
            <div className="research-flow-card">
              <strong>2. Safety gate checks if the sample looks known</strong>
              <p>
                Fusion Gate v2 is a separate fusion model: a
                StandardScaler plus Logistic Regression pipeline. It takes 7
                inputs from the classifier and Mahalanobis stage: raw
                confidence, temperature-scaled confidence, max logit, energy,
                softmax margin, softmax entropy, and Mahalanobis knownness. The
                model outputs a knownness score, and that score is then
                compared with the configured threshold before deciding whether
                the guess is safe to show.
              </p>
            </div>
            <div className="research-flow-card">
              <strong>3. Risky samples move to human review</strong>
              <p>
                Low-knownness items enter the manual review queue, where they
                can later be promoted into the intelligence backlog.
              </p>
            </div>
          </div>
        </div>

        <div className="panel research-panel">
          <div className="section-tag">What This UI Covers</div>
          <div className="context-card">
            <div className="context-row">
              <span>Live inference</span>
              <strong>
                Final label, gate decision, main scores, and technical evidence
              </strong>
            </div>
            <div className="context-row">
              <span>Manual review</span>
              <strong>
                {String(pendingReviewCount)} sample
                {pendingReviewCount === 1 ? "" : "s"} currently waiting for a
                human decision
              </strong>
            </div>
            <div className="context-row">
              <span>Intelligence backlog</span>
              <strong>
                {String(intelligenceCount)} promoted sample
                {intelligenceCount === 1 ? "" : "s"} ready for future dataset
                improvement
              </strong>
            </div>
            <div className="context-row">
              <span>Audience takeaway</span>
              <strong>
                The system is designed to avoid confidently mislabeling unknown
                waste, even when the classifier has an internal guess.
              </strong>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("inference");
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [resultImageUrl, setResultImageUrl] = useState("");
  const [resultPayload, setResultPayload] = useState(null);
  const [resultSource, setResultSource] = useState("live");
  const [statusMessage, setStatusMessage] = useState({
    tone: "info",
    text: "Upload an image and run live inference to see the result.",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isAddingCurrentToReview, setIsAddingCurrentToReview] = useState(false);
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
  const hasResult = Boolean(result);
  const currentManualReviewEntry = resultPayload?.manual_review_entry || null;
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
  const uploadSummaryFileName =
    statusMessage.fileName || selectedFile?.name || "";
  const uploadSummaryLabel = statusMessage.fileName
    ? "Selected file"
    : selectedFile
      ? "Current sample"
      : "System status";
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
  const isIntelligenceTab = activeTab === "intelligence";
  const intelligenceReviewItems = getItemsForTab("intelligence", manualReviewItems);
  const nonIntelligenceReviewItems = getItemsForTab("review", manualReviewItems);
  const activeReviewItems = getItemsForTab(activeTab, manualReviewItems);
  const selectedQueueItem =
    activeReviewItems.find((item) => item.review_id === selectedReviewId) ||
    (isIntelligenceTab ? null : activeReviewItems[0] || null);
  const reviewTabTitle =
    isIntelligenceTab ? "Intelligence Candidates" : "Review Queue";
  const reviewTabDescription =
    isIntelligenceTab
      ? "Reviewed items marked for future improvement. Use the image gallery on the right to open details only when needed."
      : "Pending items and reviewed samples that are still outside the intelligence list.";
  const visibleSidebarItems =
    isIntelligenceTab
      ? intelligenceReviewItems
      : nonIntelligenceReviewItems;
  const reviewTabEmptyMessage =
    isIntelligenceTab
      ? "No items have been added to intelligence yet."
      : "No remaining items outside the intelligence list.";
  const reviewDetailSourceLabel =
    isIntelligenceTab ? "Intelligence" : "Queued";
  const reviewDetailEmptyTitle =
    isIntelligenceTab
      ? "No intelligence candidate selected"
      : "No queued sample selected";
  const reviewDetailEmptyDescription =
    isIntelligenceTab
      ? "Select an intelligence photo to view its image, review notes, and promotion state."
      : "Once the backend blocks a live sample, it will appear here with an image preview, reviewer controls, and an option to add the result to future intelligence work.";
  const canQueueCurrentLiveResult =
    hasResult &&
    resultSource === "live" &&
    !currentManualReviewEntry &&
    Boolean(resultPayload?.uploaded_filename) &&
    Boolean(resultPayload?.stored_image_path);

  function clearLoadingStageTimers() {
    loadingStageTimersRef.current.forEach((timerId) => clearTimeout(timerId));
    loadingStageTimersRef.current = [];
  }

  function syncManualReviewSelection(
    nextItems,
    { preferredReviewId = "", replaceForm = false, tabName = activeTab } = {}
  ) {
    const visibleItems = getItemsForTab(tabName, nextItems);
    const shouldAutoSelect = tabName !== "intelligence";
    const hasPreferredReviewId =
      preferredReviewId !== "" &&
      visibleItems.some((item) => item.review_id === preferredReviewId);
    const hasCurrentReviewId =
      selectedReviewId !== "" &&
      visibleItems.some((item) => item.review_id === selectedReviewId);
    const nextSelectedReviewId = hasPreferredReviewId
      ? preferredReviewId
      : hasCurrentReviewId
        ? selectedReviewId
        : shouldAutoSelect
          ? visibleItems[0]?.review_id || ""
          : "";
    const nextSelectedItem =
      visibleItems.find((item) => item.review_id === nextSelectedReviewId) || null;

    setSelectedReviewId(nextSelectedReviewId);

    if (
      replaceForm ||
      nextSelectedReviewId !== selectedReviewId ||
      selectedReviewId === ""
    ) {
      setReviewForm(createReviewFormState(nextSelectedItem));
    }
  }

  function switchTab(nextTab) {
    setActiveTab(nextTab);

    if (nextTab === "review" || nextTab === "intelligence") {
      if (nextTab === "intelligence") {
        setSelectedReviewId("");
        setReviewForm(createReviewFormState(null));
      }

      syncManualReviewSelection(manualReviewItems, {
        replaceForm: true,
        tabName: nextTab,
      });
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
    if (activeTab === "review" || activeTab === "intelligence") {
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
    revokeObjectUrl(resultImageUrl);

    setSelectedFile(null);
    setPreviewUrl("");
    setResultImageUrl("");
    setResultPayload(null);
    setResultSource("live");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }

    setStatusMessage({
      tone: "info",
      text: "Live sample cleared. Upload a new image to continue.",
    });
  }

  async function openFilePicker() {
    if (
      typeof window !== "undefined" &&
      typeof window.showOpenFilePicker === "function"
    ) {
      try {
        const [fileHandle] = await window.showOpenFilePicker(
          LIVE_IMAGE_PICKER_OPTIONS
        );
        const nextFile = await fileHandle?.getFile();

        if (nextFile) {
          if (fileInputRef.current) {
            fileInputRef.current.value = "";
          }
          handleFile(nextFile);
        }
        return;
      } catch (error) {
        if (error?.name === "AbortError") {
          return;
        }
      }
    }

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
        text: selectedFile
          ? "Please choose a valid image file for live inference. The previous valid sample is still loaded."
          : "Please choose a valid image file for live inference.",
        fileName: file.name || "Unnamed file",
      });
      return;
    }

    revokeObjectUrl(previewUrl);
    revokeObjectUrl(resultImageUrl);

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResultImageUrl("");
    setResultPayload(null);
    setResultSource("live");
    setStatusMessage({
      tone: "info",
      text: "Image selected. Run live inference to update the result.",
    });
  }

  function selectReviewItem(item) {
    setSelectedReviewId(item.review_id);
    setReviewForm(createReviewFormState(item));
  }

  function clearSelectedReviewItem() {
    setSelectedReviewId("");
    setReviewForm(createReviewFormState(null));
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
        } with threshold ${formatNumber(payload.threshold)}.`,
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
        text: "Choose an image first.",
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
          "Live prediction failed. Please try again after checking the backend.",
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

  async function handleQueueCurrentResultForReview() {
    const normalizedBase = normalizeApiBase();

    if (!normalizedBase) {
      setStatusMessage({
        tone: "error",
        text: "Backend URL is not configured for manual review.",
      });
      return;
    }

    if (!canQueueCurrentLiveResult) {
      setStatusMessage({
        tone: "error",
        text: "Run a live prediction first before sending it to manual review.",
      });
      return;
    }

    setIsAddingCurrentToReview(true);
    setStatusMessage({
      tone: "info",
      text: "Adding this live result to manual review...",
    });

    try {
      const response = await fetch(`${normalizedBase}/api/manual-review/queue`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          uploaded_filename: resultPayload.uploaded_filename,
          stored_image_path: resultPayload.stored_image_path,
          result: resultPayload.result,
        }),
      });

      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(extractErrorMessage(payload, response.status));
      }

      startTransition(() => {
        setResultPayload((current) =>
          current
            ? {
                ...current,
                manual_review_entry: payload.item,
                manual_review_entry_status: payload.queue_status,
              }
            : current
        );
      });

      setManualReviewSummary(payload.summary || EMPTY_REVIEW_SUMMARY);
      setReviewStatusMessage({
        tone: "success",
        text:
          payload.queue_status === "existing"
            ? "That filename was already in the manual review queue, so the existing item was kept."
            : "The live result was added to manual review for later human checking.",
      });
      setStatusMessage({
        tone: "success",
        text:
          payload.queue_status === "existing"
            ? "This live result was already present in manual review."
            : "Live result added to manual review. You can review it from the Manual Review tab.",
      });

      await refreshManualReviewQueue({
        preferredReviewId: payload.item?.review_id || "",
        replaceForm: false,
        silent: true,
      });
    } catch (error) {
      setStatusMessage({
        tone: "error",
        text:
          error.message ||
          "Could not add the live result to manual review. Please try again.",
      });
    } finally {
      setIsAddingCurrentToReview(false);
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

    if (!selectedQueueItem) {
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
        `${normalizedBase}/api/manual-review/${selectedQueueItem.review_id}/decision`,
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
      syncManualReviewSelection(nextItems, {
        preferredReviewId: updatedItem.review_id,
        replaceForm: true,
        tabName: activeTab,
      });
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
          onClick={() => switchTab("inference")}
        />
        <TabButton
          active={activeTab === "review"}
          label="Manual Review"
          badge={pendingReviewCount > 0 ? String(pendingReviewCount) : ""}
          onClick={() => switchTab("review")}
        />
        <TabButton
          active={activeTab === "intelligence"}
          label="Intelligence"
          badge={
            manualReviewSummary.intelligence_count > 0
              ? String(manualReviewSummary.intelligence_count)
              : ""
          }
          onClick={() => switchTab("intelligence")}
        />
        <TabButton
          active={activeTab === "research"}
          label="Research Summary"
          onClick={() => switchTab("research")}
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
                    <span>{uploadSummaryLabel}</span>
                    {uploadSummaryFileName ? (
                      <strong title={uploadSummaryFileName}>
                        {truncateFileName(uploadSummaryFileName, 38)}
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

          </aside>

          <section className="center-column">
            {hasResult ? (
              <>
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

                    <div className="live-result-actions">
                      {canQueueCurrentLiveResult ? (
                        <button
                          className="secondary-button"
                          type="button"
                          onClick={handleQueueCurrentResultForReview}
                          disabled={isAddingCurrentToReview}
                        >
                          {isAddingCurrentToReview
                            ? "Adding..."
                            : "Add to manual review"}
                        </button>
                      ) : currentManualReviewEntry ? (
                        <span className="live-result-status-chip">
                          Already in manual review
                        </span>
                      ) : null}
                      <p className="live-result-actions-note">
                        Use this when the model prediction looks wrong, even if
                        the sample was accepted.
                      </p>
                    </div>

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

                        <SystemInfoSection result={result} />

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

                        <AdvancedEvidenceSection
                          result={result}
                          summaryLabel="Advanced evidence"
                        />
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
              </>
            ) : (
              <div className="panel result-panel simple-result-panel empty-review-panel">
                <div className="decision-card">
                  <div className="section-tag">Live Result</div>
                  <h2>No prediction yet</h2>
                  <p>
                    Upload an image and run live inference. The prediction, gate
                    score, and technical details will appear here after the
                    backend responds.
                  </p>
                </div>
              </div>
            )}
          </section>
        </section>
      ) : activeTab === "research" ? (
        <ResearchSummaryTab
          pendingReviewCount={pendingReviewCount}
          intelligenceCount={manualReviewSummary.intelligence_count || 0}
        />
      ) : (
        <section className="workspace-grid simple-workspace review-workspace">
          <aside className="panel input-panel simple-input-panel review-sidebar">
            <div className="review-queue-header">
              <div className="section-tag">{reviewTabTitle}</div>
              <p>{reviewTabDescription}</p>
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
                onClick={() => switchTab("inference")}
              >
                Back to inference
              </button>
            </div>

            {isIntelligenceTab ? (
              <a
                className="secondary-button button-link"
                href={buildApiUrl("/api/manual-review/intelligence/export")}
                target="_blank"
                rel="noreferrer"
              >
                Download intelligence JSON
              </a>
            ) : null}

            {manualReviewItems.length > 0 ? (
              isIntelligenceTab ? (
                <div className="empty-review-state compact intelligence-sidebar-note">
                  <strong>Gallery on the right</strong>
                  <p>
                    Open any promoted image from the main gallery when you want
                    to inspect its reviewed details.
                  </p>
                </div>
              ) : (
                <ReviewQueueSection
                  title={reviewTabTitle}
                  description={reviewTabDescription}
                  items={visibleSidebarItems}
                  selectedReviewId={selectedQueueItem?.review_id}
                  deletingReviewId={deletingReviewId}
                  onSelect={selectReviewItem}
                  onDelete={handleDeleteReview}
                  emptyMessage={reviewTabEmptyMessage}
                  showHeader={false}
                />
              )
            ) : (
              <div className="empty-review-state">
                <strong>No review items yet</strong>
                <p>
                  Run live inference on an uncertain image and blocked samples
                  will show up here automatically.
                </p>
              </div>
            )}
          </aside>

          <section className="center-column">
            {isIntelligenceTab ? (
              selectedQueueItem ? (
                <>
                  <div className="panel intelligence-detail-toolbar">
                    <button
                      className="secondary-button support-button"
                      type="button"
                      onClick={clearSelectedReviewItem}
                    >
                      Back to gallery
                    </button>
                    <p>
                      Viewing one promoted sample. Return to the gallery to open
                      another image.
                    </p>
                  </div>

                  <ReviewDetailPanel
                    item={selectedQueueItem}
                    reviewForm={reviewForm}
                    onUpdateReviewForm={updateReviewForm}
                    onSubmitReview={handleSubmitReview}
                    onResetReviewForm={(item) =>
                      setReviewForm(createReviewFormState(item))
                    }
                    isSubmittingReview={isSubmittingReview}
                    sourceLabel={reviewDetailSourceLabel}
                    emptyTitle={reviewDetailEmptyTitle}
                    emptyDescription={reviewDetailEmptyDescription}
                  />
                </>
              ) : intelligenceReviewItems.length > 0 ? (
                <div className="panel result-panel simple-result-panel intelligence-gallery-panel">
                  <IntelligenceGallerySection
                    title="Image Gallery"
                    description="These are already reviewed intelligence candidates. Click any image only when you want to inspect the full saved details."
                    items={intelligenceReviewItems}
                    selectedReviewId={selectedReviewId}
                    deletingReviewId={deletingReviewId}
                    onSelect={selectReviewItem}
                    onDelete={handleDeleteReview}
                    emptyMessage={reviewTabEmptyMessage}
                  />
                </div>
              ) : (
                <div className="panel result-panel simple-result-panel empty-review-panel">
                  <div className="decision-card tone-review">
                    <div className="section-tag">Intelligence</div>
                    <h2>No promoted samples yet</h2>
                    <p>
                      Reviewed items marked for future intelligence work will
                      appear here as an image gallery.
                    </p>
                  </div>
                </div>
              )
            ) : (
              <ReviewDetailPanel
                item={selectedQueueItem}
                reviewForm={reviewForm}
                onUpdateReviewForm={updateReviewForm}
                onSubmitReview={handleSubmitReview}
                onResetReviewForm={(item) =>
                  setReviewForm(createReviewFormState(item))
                }
                isSubmittingReview={isSubmittingReview}
                sourceLabel={reviewDetailSourceLabel}
                emptyTitle={reviewDetailEmptyTitle}
                emptyDescription={reviewDetailEmptyDescription}
              />
            )}
          </section>
        </section>
      )}
    </main>
  );
}
