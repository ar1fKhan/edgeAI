/**
 * @file postprocessing.cpp
 * @brief Model output decoding: YOLO output parsing + NMS.
 *
 * No business logic here — only model-specific output decoding.
 * Verdict determination is handled by DecisionEngine.
 */

#include "edgeai/inference/postprocessing.h"
#include "edgeai/common/logger.h"

#include <algorithm>
#include <cmath>

namespace edgeai {

Postprocessor::Postprocessor(float conf_threshold, float nms_threshold, int num_classes)
    : conf_threshold_(conf_threshold)
    , nms_threshold_(nms_threshold)
    , num_classes_(num_classes) {}

std::vector<Detection> Postprocessor::process(
    const std::vector<float>& output,
    int output_rows,
    int output_cols,
    float scale_x, float scale_y,
    int pad_x, int pad_y,
    int orig_w, int orig_h) const
{
    std::vector<Detection> detections;

    for (int i = 0; i < output_rows; ++i) {
        const float* row = output.data() + i * output_cols;

        // YOLOv8 format: [cx, cy, w, h, class1_conf, class2_conf, ...]
        float cx = row[0];
        float cy = row[1];
        float w  = row[2];
        float h  = row[3];

        // YOLOv8 class scores are already probabilities in [0, 1] — no sigmoid needed
        int best_class = 0;
        float best_conf = 0.0f;
        for (int c = 0; c < num_classes_; ++c) {
            float conf = row[4 + c];
            if (conf > best_conf) {
                best_conf = conf;
                best_class = c;
            }
        }

        if (best_conf < conf_threshold_) continue;

        // Map back to original image coordinates
        // Remove padding and unscale
        float x1 = (cx - w / 2.0f - pad_x) / scale_x;
        float y1 = (cy - h / 2.0f - pad_y) / scale_y;
        float bw = w / scale_x;
        float bh = h / scale_y;

        // Clip to image bounds
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_w)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_h)));
        bw = std::min(bw, static_cast<float>(orig_w) - x1);
        bh = std::min(bh, static_cast<float>(orig_h) - y1);

        // Normalize to [0, 1]
        Detection det;
        det.type       = class_id_to_type(best_class);
        det.confidence = best_conf;
        det.class_id   = best_class;
        det.bbox.x      = x1 / orig_w;
        det.bbox.y      = y1 / orig_h;
        det.bbox.width  = bw / orig_w;
        det.bbox.height = bh / orig_h;

        detections.push_back(det);
    }

    return detections;
}

std::vector<Detection> Postprocessor::apply_nms(std::vector<Detection>& detections) const {
    if (detections.empty()) return {};

    // Sort by confidence (descending)
    std::sort(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<Detection> result;

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;

            // Only suppress same-class detections
            if (detections[i].class_id == detections[j].class_id) {
                float iou = detections[i].bbox.iou(detections[j].bbox);
                if (iou > nms_threshold_) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return result;
}

DefectType Postprocessor::class_id_to_type(int class_id) const {
    switch (class_id) {
        case 0: return DefectType::Dent;
        case 1: return DefectType::WrongLabel;
        case 2: return DefectType::MissingLabel;
        case 3: return DefectType::SealDefect;
        case 4: return DefectType::ColorMismatch;
        default: return DefectType::Unknown;
    }
}

}  // namespace edgeai
