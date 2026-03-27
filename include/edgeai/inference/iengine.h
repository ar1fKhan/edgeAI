#pragma once

/**
 * @file iengine.h
 * @brief Abstract inference engine interface — enables swapping backends.
 *
 * Implement this interface for any inference runtime:
 *   - ONNX Runtime (OnnxEngine)
 *   - TensorRT
 *   - OpenVINO
 *   - TFLite
 *   - Custom FPGA accelerators
 */

#include <string>
#include <vector>

namespace edgeai {

class IInferenceEngine {
public:
    virtual ~IInferenceEngine() = default;

    /// Load model from file
    virtual bool load_model() = 0;

    /// Run inference on a preprocessed input tensor
    /// @param input_tensor Preprocessed float data [1, 3, H, W] in CHW format
    /// @return Raw output tensor as vector of floats
    [[nodiscard]] virtual std::vector<float> infer(const std::vector<float>& input_tensor) = 0;

    /// Check if model is loaded and ready
    [[nodiscard]] virtual bool is_loaded() const = 0;

    /// Get input dimensions
    [[nodiscard]] virtual int input_width() const = 0;
    [[nodiscard]] virtual int input_height() const = 0;
};

}  // namespace edgeai
