/**
 * @file benchmark.cpp
 * @brief Inference benchmark tool — measures latency across N iterations.
 *
 * Usage: benchmark_inference --model models/defect_detector.onnx --iterations 1000
 */

#include <iostream>
#include <numeric>
#include <vector>
#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "edgeai/inference/onnx_engine.h"
#include "edgeai/inference/preprocessing.h"
#include "edgeai/common/timer.h"
#include "edgeai/common/logger.h"
#include "edgeai/common/types.h"

int main(int argc, char* argv[]) {
    std::string model_path = "models/defect_detector.onnx";
    int iterations = 100;
    int warmup = 10;
    bool use_gpu = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc)      model_path = argv[++i];
        else if (arg == "--iterations" && i + 1 < argc) iterations = std::stoi(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) warmup = std::stoi(argv[++i]);
        else if (arg == "--gpu")                    use_gpu = true;
    }

    edgeai::Logger::instance().set_level(edgeai::LogLevel::INFO);

    std::cout << "╔══════════════════════════════════════════╗" << std::endl;
    std::cout << "║  EdgeAI Inference Benchmark              ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════╣" << std::endl;
    std::cout << "║ Model:      " << model_path << std::endl;
    std::cout << "║ Iterations: " << iterations << std::endl;
    std::cout << "║ Warmup:     " << warmup << std::endl;
    std::cout << "║ GPU:        " << (use_gpu ? "Yes" : "No") << std::endl;
    std::cout << "╠══════════════════════════════════════════╣" << std::endl;

    // Setup inference
    edgeai::InferenceConfig config;
    config.model_path = model_path;
    config.use_gpu = use_gpu;
    config.execution_provider = use_gpu ? "cuda" : "cpu";

    edgeai::OnnxEngine engine(config);
    if (!engine.load_model()) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    edgeai::Preprocessor preprocessor(config.input_width, config.input_height);

    // Generate synthetic test image (simulates paint can photo)
    cv::Mat test_image(1080, 1920, CV_8UC3);
    cv::randu(test_image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    auto prep = preprocessor.process(test_image);

    // Warmup
    std::cout << "║ Running warmup..." << std::endl;
    for (int i = 0; i < warmup; ++i) {
        (void)engine.infer(prep.tensor);
    }

    // Benchmark
    std::cout << "║ Benchmarking..." << std::endl;
    std::vector<double> latencies;
    latencies.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        edgeai::ScopedTimer timer("inference", false);
        (void)engine.infer(prep.tensor);
        latencies.push_back(timer.elapsed_ms());
    }

    // Calculate statistics
    std::sort(latencies.begin(), latencies.end());
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double mean = sum / latencies.size();

    double sq_sum = 0.0;
    for (auto l : latencies) sq_sum += (l - mean) * (l - mean);
    double stddev = std::sqrt(sq_sum / latencies.size());

    double p50 = latencies[latencies.size() / 2];
    double p95 = latencies[static_cast<size_t>(latencies.size() * 0.95)];
    double p99 = latencies[static_cast<size_t>(latencies.size() * 0.99)];
    double min_lat = latencies.front();
    double max_lat = latencies.back();

    std::cout << "╠══════════════════════════════════════════╣" << std::endl;
    std::cout << "║         RESULTS                          ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════╣" << std::endl;
    std::cout << "║ Mean:    " << mean << " ms" << std::endl;
    std::cout << "║ Stddev:  " << stddev << " ms" << std::endl;
    std::cout << "║ Min:     " << min_lat << " ms" << std::endl;
    std::cout << "║ Max:     " << max_lat << " ms" << std::endl;
    std::cout << "║ P50:     " << p50 << " ms" << std::endl;
    std::cout << "║ P95:     " << p95 << " ms" << std::endl;
    std::cout << "║ P99:     " << p99 << " ms" << std::endl;
    std::cout << "║ FPS:     " << 1000.0 / mean << std::endl;
    std::cout << "╚══════════════════════════════════════════╝" << std::endl;

    return 0;
}
