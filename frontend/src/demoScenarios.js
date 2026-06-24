export const RESEARCH_STATS = [
  { label: "AUROC", value: "0.9269" },
  { label: "Unknown Rejection", value: "93.37%" },
  { label: "False Acceptance", value: "6.63%" },
];

export const KNOWN_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"];

export const DEMO_SCENARIOS = {
  known: {
    payload: {
      uploaded_filename: "plastic-demo.jpg",
      stored_image_path: "ml/outputs/inference_examples/plastic-demo.jpg",
      result: {
        policy_version: "fusion_gate_v2_mahalanobis_v1",
        image_path: "ml/data/raw/trashbox/plastic/plastic 1777.jpg",
        device: "cpu",
        embedding_layer: "classifier.3",
        embedding_dimension: 1280,
        known_classes: [...KNOWN_CLASSES],
        prediction: {
          internal_top1_prediction: "plastic",
          pred_index: 4,
          raw_confidence: 1.0,
          temperature_scaled_confidence: 0.9998892545700073,
          max_logit: 16.152263641357422,
          energy: -16.152263641357422,
          softmax_margin: 0.9999999738280625,
          softmax_entropy: 0.0000005142,
          class_probabilities: {
            cardboard: 0.0000000022,
            glass: 0.0,
            metal: 0.0000000007,
            paper: 0.0000000262,
            plastic: 1.0,
          },
        },
        mahalanobis: {
          mahalanobis_min_distance: 2937.63623046875,
          mahalanobis_knownness: -2937.63623046875,
          mahalanobis_nearest_class: "plastic",
        },
        fusion_gate: {
          feature_names: [
            "confidence",
            "temperature_scaled_confidence",
            "max_logit",
            "energy",
            "softmax_margin",
            "softmax_entropy",
            "mahalanobis_knownness",
          ],
          knownness_score: 0.9968887080867044,
          threshold: 0.6314586412215439,
          accepted_as_known: true,
          decision_type: "known_fine_label",
        },
        final_decision: {
          accepted_as_known: true,
          decision_type: "known_fine_label",
          user_visible_label: "plastic",
          coarse_label: "recyclable",
          show_internal_prediction_to_user: true,
          internal_top1_prediction_logged: true,
          user_message:
            "This item is likely plastic. It belongs to the recyclable category.",
        },
      },
    },
    statusText:
      "Accepted demo loaded. This shows the safe recyclable path for a known plastic item.",
  },
  review: {
    payload: {
      uploaded_filename: "unknown-demo.jpg",
      stored_image_path: "ml/outputs/inference_examples/unknown-demo.jpg",
      result: {
        policy_version: "fusion_gate_v2_mahalanobis_v1",
        image_path: "ml/data/raw/garbage_v2/clothes/clothes_319.jpg",
        device: "cpu",
        embedding_layer: "classifier.3",
        embedding_dimension: 1280,
        known_classes: [...KNOWN_CLASSES],
        prediction: {
          internal_top1_prediction: "paper",
          pred_index: 3,
          raw_confidence: 0.6594269871711731,
          temperature_scaled_confidence: 0.5049406886100769,
          max_logit: 3.7160091400146484,
          energy: -4.132392883300781,
          softmax_margin: 0.4784678667783737,
          softmax_entropy: 0.8791800632848068,
          class_probabilities: {
            cardboard: 0.18095912039279938,
            glass: 0.00004264667222741991,
            metal: 0.00026703518233262,
            paper: 0.6594269871711731,
            plastic: 0.15930409729480743,
          },
        },
        mahalanobis: {
          mahalanobis_min_distance: 1771.123291015625,
          mahalanobis_knownness: -1771.123291015625,
          mahalanobis_nearest_class: "plastic",
        },
        fusion_gate: {
          feature_names: [
            "confidence",
            "temperature_scaled_confidence",
            "max_logit",
            "energy",
            "softmax_margin",
            "softmax_entropy",
            "mahalanobis_knownness",
          ],
          knownness_score: 0.07132559245422432,
          threshold: 0.6314586412215439,
          accepted_as_known: false,
          decision_type: "unknown_manual_review",
        },
        final_decision: {
          accepted_as_known: false,
          decision_type: "unknown_manual_review",
          user_visible_label: "manual_review_required",
          coarse_label: "manual_review_required",
          show_internal_prediction_to_user: false,
          internal_top1_prediction_logged: true,
          user_message:
            "The system is not confident that this item belongs to the supported recyclable classes. Please send it for manual review.",
        },
      },
    },
    statusText:
      "Manual-review demo loaded. This shows the safety gate blocking a risky unknown item.",
  },
};
