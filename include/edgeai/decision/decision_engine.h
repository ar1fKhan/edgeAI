#pragma once

/**
 * @file decision_engine.h
 * @brief Default decision engine — threshold-based verdict logic.
 *
 * Business rules:
 *   confidence >= reject_threshold  → REJECT
 *   confidence >= review_threshold  → REVIEW
 *   otherwise                       → PASS
 *
 * To customize for a different domain, either:
 *   1. Adjust thresholds via config
 *   2. Subclass IDecisionEngine with custom rules
 */

#include "edgeai/decision/idecision_engine.h"

namespace edgeai {

class DecisionEngine : public IDecisionEngine {
public:
    /**
     * @param reject_threshold  Confidence above this → REJECT
     * @param review_threshold  Confidence above this but below reject → REVIEW
     */
    DecisionEngine(float reject_threshold, float review_threshold);

    /// Determine verdict based on detection confidences
    [[nodiscard]] Verdict decide(
        const std::vector<Detection>& detections
    ) const override;

private:
    float reject_threshold_;
    float review_threshold_;
};

}  // namespace edgeai
