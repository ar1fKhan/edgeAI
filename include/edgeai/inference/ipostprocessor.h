#pragma once

/**
 * @file ipostprocessor.h
 * @brief Abstract postprocessor interface — enables swapping output decoding strategies.
 *
 * Implement this interface for any model output decoding pipeline:
 *   - YOLO anchor-based decoding (Postprocessor)
 *   - SSD decoding
 *   - Classification argmax
 *   - Custom accelerator output formats
 *
 * Responsibility boundary:
 *   IPostprocessor: raw float tensor → std::vector<Detection>
 *   IDecisionEngine: std::vector<Detection> → Verdict
 */

#include <vector>
#include "edgeai/common/types.h"

namespace edgeai {

class IPostprocessor {
public:
    virtual ~IPostprocessor() = default;

    /// Parse raw model output tensor into detections
    [[nodiscard]] virtual std::vector<Detection> process(
        const std::vector<float>& output,
        int output_rows,
        int output_cols,
        float scale_x, float scale_y,
        int pad_x, int pad_y,
        int orig_w, int orig_h
    ) const = 0;

    /// Apply Non-Maximum Suppression
    [[nodiscard]] virtual std::vector<Detection> apply_nms(
        std::vector<Detection>& detections
    ) const = 0;
};

}  // namespace edgeai
