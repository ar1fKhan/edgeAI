#pragma once

/**
 * @file postprocessing.h
 * @brief YOLO model output decoding — implements IPostprocessor.
 *
 * NMS, confidence filtering, coordinate mapping from raw tensor
 * to structured Detection objects.
 *
 * Responsibility boundary:
 *   IPostprocessor: raw float tensor → std::vector<Detection>
 *   IDecisionEngine: std::vector<Detection> → Verdict
 *
 * Swap: implement IPostprocessor for SSD, classification, etc.
 */

#include <vector>
#include "edgeai/inference/ipostprocessor.h"

namespace edgeai {

class Postprocessor : public IPostprocessor {
public:
    Postprocessor(float conf_threshold, float nms_threshold, int num_classes);

    /// Parse raw YOLO output tensor into detections (implements IPostprocessor)
    [[nodiscard]] std::vector<Detection> process(
        const std::vector<float>& output,
        int output_rows,
        int output_cols,
        float scale_x, float scale_y,
        int pad_x, int pad_y,
        int orig_w, int orig_h
    ) const override;

    /// Apply Non-Maximum Suppression (implements IPostprocessor)
    [[nodiscard]] std::vector<Detection> apply_nms(
        std::vector<Detection>& detections
    ) const override;

private:
    float conf_threshold_;
    float nms_threshold_;
    int   num_classes_;

    DefectType class_id_to_type(int class_id) const;
};

}  // namespace edgeai
