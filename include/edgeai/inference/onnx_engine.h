#pragma once

/**
 * @file onnx_engine.h
 * @brief ONNX Runtime inference engine — pure tensor-in, tensor-out.
 *
 * The engine does NOT know about images, preprocessing, or postprocessing.
 * It only accepts a preprocessed float tensor and returns raw model output.
 *
 * Flow: Preprocessor → Tensor → OnnxEngine → Raw Output → Postprocessor
 *
 * Supports CPU, CUDA, TensorRT, and OpenVINO execution providers.
 */

#include <memory>
#include <string>
#include <vector>
#include "edgeai/inference/iengine.h"
#include "edgeai/common/types.h"

#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace edgeai {

class OnnxEngine : public IInferenceEngine {
public:
    explicit OnnxEngine(const InferenceConfig& config);
    ~OnnxEngine() override;

    // Non-copyable
    OnnxEngine(const OnnxEngine&) = delete;
    OnnxEngine& operator=(const OnnxEngine&) = delete;

    /// Load model from file
    bool load_model() override;

    /// Run inference on a preprocessed input tensor
    /// @param input_tensor Preprocessed float data [1, 3, H, W] in CHW format
    /// @return Raw output tensor as vector of floats
    [[nodiscard]] std::vector<float> infer(const std::vector<float>& input_tensor) override;

    /// Check if model is loaded and ready
    [[nodiscard]] bool is_loaded() const override { return model_loaded_; }

    /// Get input dimensions
    [[nodiscard]] int input_width() const override { return config_.input_width; }
    [[nodiscard]] int input_height() const override { return config_.input_height; }

private:
    InferenceConfig config_;
    bool            model_loaded_ = false;

#ifdef HAS_ONNXRUNTIME
    std::unique_ptr<Ort::Env>             env_;
    std::unique_ptr<Ort::Session>         session_;
    std::unique_ptr<Ort::SessionOptions>  session_options_;
    Ort::AllocatorWithDefaultOptions      allocator_;

    std::vector<std::string>           input_name_strings_;
    std::vector<std::string>           output_name_strings_;
    std::vector<const char*>           input_names_;
    std::vector<const char*>           output_names_;
    std::vector<int64_t>               input_shape_;
    std::vector<int64_t>               output_shape_;
#endif

    void setup_execution_provider();
    static bool verify_model_file(const std::string& path);
};

}  // namespace edgeai
