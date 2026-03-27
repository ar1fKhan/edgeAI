#pragma once

/**
 * @file ipreprocessor.h
 * @brief Abstract preprocessor interface — enables swapping preprocessing strategies.
 *
 * Implement this interface for any image preprocessing pipeline:
 *   - YOLO letterbox + normalize (Preprocessor)
 *   - CenterCrop for classification models
 *   - Custom augmentation pipelines
 *   - Hardware-accelerated preprocessing (NPU / VPU)
 */

#include <opencv2/core.hpp>
#include <vector>

namespace edgeai {

/// Result of preprocessing — carries tensor + geometry needed for postprocessing
struct PreprocessResult {
    std::vector<float> tensor;   // CHW float tensor
    float              scale_x;  // horizontal scale factor
    float              scale_y;  // vertical scale factor
    int                pad_x;    // horizontal padding
    int                pad_y;    // vertical padding
    int                orig_w;
    int                orig_h;
};

class IPreprocessor {
public:
    virtual ~IPreprocessor() = default;

    /// Preprocess an image for model inference.
    /// Returns a PreprocessResult containing the float tensor and
    /// geometry metadata needed by the postprocessor.
    [[nodiscard]] virtual PreprocessResult process(const cv::Mat& image) const = 0;
};

}  // namespace edgeai
