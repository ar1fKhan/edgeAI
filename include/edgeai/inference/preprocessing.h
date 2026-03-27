#pragma once

/**
 * @file preprocessing.h
 * @brief YOLO image preprocessing — implements IPreprocessor.
 *
 * Letterbox resize, BGR→RGB, normalize [0,1], HWC→CHW.
 *
 * Swap: implement IPreprocessor with a different strategy
 *       (e.g., CenterCropPreprocessor, NPUPreprocessor).
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include "edgeai/inference/ipreprocessor.h"

namespace edgeai {

class Preprocessor : public IPreprocessor {
public:
    explicit Preprocessor(int target_w, int target_h);

    /// YOLO letterbox preprocessing (implements IPreprocessor)
    [[nodiscard]] PreprocessResult process(const cv::Mat& image) const override;

    /// Simple resize + normalize (no letterbox) — concrete-only convenience
    [[nodiscard]] std::vector<float> simple_preprocess(const cv::Mat& image) const;

private:
    int target_w_;
    int target_h_;

    /// Letterbox resize: maintain aspect ratio with padding
    cv::Mat letterbox(const cv::Mat& image, float& scale, int& pad_x, int& pad_y) const;

    /// Convert HWC BGR cv::Mat to CHW RGB float vector
    static std::vector<float> mat_to_chw(const cv::Mat& image);
};

}  // namespace edgeai
