#pragma once

/**
 * @file idecision_engine.h
 * @brief Abstract decision engine interface — business logic boundary.
 *
 * The DecisionEngine answers: "Given these detections, what do we do?"
 *
 * This is where domain-specific business rules live:
 *   - Confidence thresholds for reject vs review
 *   - Multi-defect escalation rules
 *   - Per-defect-type severity weighting
 *   - Customer-specific acceptance criteria
 *
 * Different domains have different decision logic:
 *   - Paint cans: any dent → reject
 *   - Potatoes: small bruise → pass, large rot → reject
 *   - PCBs: solder bridge → reject, minor scratch → review
 *
 * Implement this interface to customize per-domain or per-customer.
 */

#include <vector>
#include "edgeai/common/types.h"

namespace edgeai {

class IDecisionEngine {
public:
    virtual ~IDecisionEngine() = default;

    /// Determine inspection verdict from detection results
    /// @param detections Decoded detections from postprocessor
    /// @return Verdict: Pass, Reject, or Review
    [[nodiscard]] virtual Verdict decide(
        const std::vector<Detection>& detections
    ) const = 0;
};

}  // namespace edgeai
