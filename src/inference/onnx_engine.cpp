/**
 * @file onnx_engine.cpp
 * @brief ONNX Runtime inference engine — pure tensor-in, tensor-out.
 *
 * No image handling. No preprocessing. No postprocessing.
 * Only tensor → model → tensor.
 */

#include "edgeai/inference/onnx_engine.h"
#include "edgeai/common/logger.h"
#include "edgeai/common/timer.h"

#include <algorithm>
#include <numeric>
#include <cmath>

namespace edgeai {

OnnxEngine::OnnxEngine(const InferenceConfig& config)
    : config_(config) {}

OnnxEngine::~OnnxEngine() = default;

bool OnnxEngine::load_model() {
#ifdef HAS_ONNXRUNTIME
    try {
        ScopedTimer timer("Model Load");

        // Create ONNX Runtime environment
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "EdgeAI_DefectDetector");

        // Session options
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(config_.num_threads);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Enable execution provider
        setup_execution_provider();

        // Load model
        session_ = std::make_unique<Ort::Session>(*env_, config_.model_path.c_str(), *session_options_);

        // Get input info
        auto input_count = session_->GetInputCount();
        for (size_t i = 0; i < input_count; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator_);
            input_name_strings_.emplace_back(name.get());

            auto type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_shape_ = tensor_info.GetShape();
        }

        // Get output info
        auto output_count = session_->GetOutputCount();
        for (size_t i = 0; i < output_count; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator_);
            output_name_strings_.emplace_back(name.get());

            auto type_info = session_->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_shape_ = tensor_info.GetShape();
        }

        // Build const char* arrays from owned strings
        input_names_.clear();
        for (const auto& s : input_name_strings_)
            input_names_.push_back(s.c_str());
        output_names_.clear();
        for (const auto& s : output_name_strings_)
            output_names_.push_back(s.c_str());

        model_loaded_ = true;
        LOG_INFO("Inference", "Model loaded: " + config_.model_path
                 + " | Input: [" + std::to_string(input_shape_[0]) + ","
                 + std::to_string(input_shape_[1]) + ","
                 + std::to_string(input_shape_[2]) + ","
                 + std::to_string(input_shape_[3]) + "]"
                 + " | Provider: " + config_.execution_provider);
        return true;

    } catch (const Ort::Exception& e) {
        LOG_ERROR("Inference", "ONNX Runtime error: " + std::string(e.what()));
        return false;
    }
#else
    LOG_WARN("Inference", "ONNX Runtime not available — running in stub mode");
    model_loaded_ = true;  // stub mode for development
    return true;
#endif
}

std::vector<float> OnnxEngine::infer(const std::vector<float>& input_tensor) {
#ifdef HAS_ONNXRUNTIME
    if (!model_loaded_ || !session_) {
        LOG_ERROR("Inference", "Model not loaded");
        return {};
    }

    try {
        // Create input tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto ort_input = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input_tensor.data()),
            input_tensor.size(),
            input_shape_.data(),
            input_shape_.size()
        );

        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &ort_input,
            1,
            output_names_.data(),
            output_names_.size()
        );

        // Extract output
        auto* output_data = output_tensors[0].GetTensorData<float>();
        auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        auto output_shape = output_info.GetShape();

        size_t total_elements = 1;
        for (auto dim : output_shape) {
            total_elements *= static_cast<size_t>(dim);
        }

        return std::vector<float>(output_data, output_data + total_elements);

    } catch (const Ort::Exception& e) {
        LOG_ERROR("Inference", "Inference error: " + std::string(e.what()));
        return {};
    }
#else
    // Stub mode: return empty (no detections)
    (void)input_tensor;
    return {};
#endif
}

void OnnxEngine::setup_execution_provider() {
#ifdef HAS_ONNXRUNTIME
    if (config_.execution_provider == "cuda" && config_.use_gpu) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = config_.gpu_device_id;
        session_options_->AppendExecutionProvider_CUDA(cuda_options);
        LOG_INFO("Inference", "Using CUDA execution provider (GPU " 
                 + std::to_string(config_.gpu_device_id) + ")");
    }
    // TensorRT and OpenVINO would be added similarly with their respective options
    else {
        LOG_INFO("Inference", "Using CPU execution provider with " 
                 + std::to_string(config_.num_threads) + " threads");
    }
#endif
}

}  // namespace edgeai
