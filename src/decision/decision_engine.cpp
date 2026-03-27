/**
 * @file decision_engine.cpp
 * @brief Default threshold-based decision engine.
 *
 * Pure business logic — no model knowledge, no image handling.
 */

#include "edgeai/decision/decision_engine.h"

namespace edgeai {

DecisionEngine::DecisionEngine(float reject_threshold, float review_threshold)
    : reject_threshold_(reject_threshold)
    , review_threshold_(review_threshold) {}

Verdict DecisionEngine::decide(const std::vector<Detection>& detections) const {
    if (detections.empty()) return Verdict::Pass;

    bool has_confident_defect = false;
    bool has_borderline = false;

    for (const auto& det : detections) {
        if (det.confidence >= reject_threshold_) {
            has_confident_defect = true;
            break;
        }
        if (det.confidence >= review_threshold_) {
            has_borderline = true;
        }
    }

    if (has_confident_defect) return Verdict::Reject;
    if (has_borderline) return Verdict::Review;
    return Verdict::Pass;
}

}  // namespace edgeai
