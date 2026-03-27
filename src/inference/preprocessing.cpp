/**
 * @file preprocessing.cpp
 * @brief Image preprocessing for YOLO inference.
 */

#include "edgeai/inference/preprocessing.h"
#include "edgeai/common/logger.h"

namespace edgeai {

Preprocessor::Preprocessor(int target_w, int target_h)
    : target_w_(target_w), target_h_(target_h) {}

PreprocessResult Preprocessor::process(const cv::Mat& image) const {
    PreprocessResult result;
    result.orig_w = image.cols;
    result.orig_h = image.rows;

    // Step 1: Letterbox resize
    float scale;
    cv::Mat resized = letterbox(image, scale, result.pad_x, result.pad_y);

    result.scale_x = scale;
    result.scale_y = scale;

    // Step 2: BGR → RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Step 3: Normalize to [0, 1] float32
    cv::Mat normalized;
    rgb.convertTo(normalized, CV_32FC3, 1.0 / 255.0);

    // Step 4: HWC → CHW
    result.tensor = mat_to_chw(normalized);

    return result;
}

std::vector<float> Preprocessor::simple_preprocess(const cv::Mat& image) const {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(target_w_, target_h_));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    cv::Mat normalized;
    rgb.convertTo(normalized, CV_32FC3, 1.0 / 255.0);

    return mat_to_chw(normalized);
}

cv::Mat Preprocessor::letterbox(const cv::Mat& image, float& scale,
                                 int& pad_x, int& pad_y) const {
    int img_w = image.cols;
    int img_h = image.rows;

    // Compute scale to fit within target while maintaining aspect ratio
    float scale_w = static_cast<float>(target_w_) / img_w;
    float scale_h = static_cast<float>(target_h_) / img_h;
    scale = std::min(scale_w, scale_h);

    int new_w = static_cast<int>(img_w * scale);
    int new_h = static_cast<int>(img_h * scale);

    // Compute padding
    pad_x = (target_w_ - new_w) / 2;
    pad_y = (target_h_ - new_h) / 2;

    // Resize
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // Create padded image (gray padding = 114)
    cv::Mat padded(target_h_, target_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_w, new_h)));

    return padded;
}

std::vector<float> Preprocessor::mat_to_chw(const cv::Mat& image) {
    int h = image.rows;
    int w = image.cols;
    int c = image.channels();

    std::vector<float> tensor(c * h * w);

    // Split channels and interleave as CHW
    std::vector<cv::Mat> channels(c);
    cv::split(image, channels);

    for (int ch = 0; ch < c; ++ch) {
        std::memcpy(tensor.data() + ch * h * w,
                     channels[ch].data,
                     h * w * sizeof(float));
    }

    return tensor;
}

}  // namespace edgeai
