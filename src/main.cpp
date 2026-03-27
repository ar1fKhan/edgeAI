/**
 * @file main.cpp
 * @brief EdgeAI Paint Can Defect Inspector — Composition Root
 *
 * This is the only place where concrete implementations are wired up.
 * The pipeline itself only knows about interfaces (ICamera, IInferenceEngine, etc.)
 *
 * To swap a module:
 *   Replace std::make_unique<CameraManager>   → std::make_unique<YourCamera>
 *   Replace std::make_unique<OnnxEngine>      → std::make_unique<TensorRTEngine>
 *   Replace std::make_unique<DefectDatabase>  → std::make_unique<PostgresStore>
 *   Replace std::make_unique<GpioController>  → std::make_unique<PlcController>
 *
 * Usage:
 *   edge_inspector --config configs/inference.cfg
 *   edge_inspector --model models/defect_detector.onnx --camera 0
 *   edge_inspector --image test_image.jpg  (single image mode)
 */

#include <csignal>
#include <cstring>
#include <iostream>
#include <string>
#include <atomic>

#include <opencv2/imgcodecs.hpp>

// ── Pipeline (uses interfaces only) ────────────────────────────
#include "edgeai/pipeline/defect_pipeline.h"

// ── Concrete implementations (wired up HERE only) ──────────────
#include "edgeai/camera/camera_manager.h"
#include "edgeai/inference/onnx_engine.h"
#include "edgeai/inference/preprocessing.h"
#include "edgeai/inference/postprocessing.h"
#include "edgeai/decision/decision_engine.h"
#include "edgeai/db/defect_database.h"
#include "edgeai/io/gpio_controller.h"

// ── Common ─────────────────────────────────────────────────────
#include "edgeai/common/config_loader.h"
#include "edgeai/common/logger.h"
#include "edgeai/common/types.h"

static std::atomic<bool> g_running{true};

void signal_handler(int sig) {
    (void)sig;
    g_running = false;
}

struct CliArgs {
    std::string config_path  = "";
    std::string model_path   = "models/defect_detector.onnx";
    std::string image_path   = "";        // single-image mode
    std::string video_path   = "";        // video file debug mode
    int         camera_id    = 0;
    bool        display      = false;
    bool        verbose      = false;
    bool        loop_video   = false;     // loop video playback
    bool        help         = false;
};

CliArgs parse_args(int argc, char* argv[]) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc)     args.config_path = argv[++i];
        else if (arg == "--model" && i + 1 < argc)  args.model_path = argv[++i];
        else if (arg == "--camera" && i + 1 < argc)  args.camera_id = std::stoi(argv[++i]);
        else if (arg == "--video" && i + 1 < argc)   args.video_path = argv[++i];
        else if (arg == "--image" && i + 1 < argc)   args.image_path = argv[++i];
        else if (arg == "--loop")                     args.loop_video = true;
        else if (arg == "--display")                  args.display = true;
        else if (arg == "--verbose" || arg == "-v")   args.verbose = true;
        else if (arg == "--help" || arg == "-h")      args.help = true;
    }
    return args;
}

void print_usage() {
    std::cout << R"(
EdgeAI Paint Can Defect Inspector v1.0.0
========================================

Usage:
  edge_inspector [options]

Options:
  --config <path>    Configuration file path
  --model <path>     ONNX model file path (default: models/defect_detector.onnx)
  --camera <id>      Camera device ID (default: 0)
  --video <path>     Use video file instead of camera (debug mode)
  --loop             Loop video playback (use with --video)
  --image <path>     Single image mode (process one image and exit)
  --display          Show live detection window
  --verbose, -v      Enable verbose logging
  --help, -h         Show this help message

Examples:
  edge_inspector --config configs/inference.cfg
  edge_inspector --model models/defect_detector.onnx --camera 0 --display
  edge_inspector --video data/test_videos/sample.mp4 --loop --verbose
  edge_inspector --image test_can.jpg --model models/defect_detector.onnx

)" << std::endl;
}

int run_single_image(const CliArgs& args) {
    LOG_INFO("Main", "Single image mode: " + args.image_path);

    cv::Mat image = cv::imread(args.image_path);
    if (image.empty()) {
        LOG_ERROR("Main", "Failed to load image: " + args.image_path);
        return 1;
    }

    edgeai::PipelineConfig config;
    if (!args.config_path.empty()) {
        config = edgeai::ConfigLoader::load(args.config_path);
    }
    config.inference.model_path = args.model_path;

    // ── Wire up components (single-image: no camera, no store, no GPIO) ──
    auto engine = std::make_unique<edgeai::OnnxEngine>(config.inference);
    auto preprocessor = std::make_unique<edgeai::Preprocessor>(
        config.inference.input_width, config.inference.input_height);
    auto postprocessor = std::make_unique<edgeai::Postprocessor>(
        config.inference.conf_threshold,
        config.inference.nms_threshold,
        config.inference.num_classes);
    auto decision = std::make_unique<edgeai::DecisionEngine>(
        config.inference.conf_threshold,
        config.inference.review_threshold);

    edgeai::DefectPipeline pipeline(
        nullptr,                        // no camera
        std::move(engine),
        std::move(preprocessor),
        std::move(postprocessor),
        std::move(decision),
        nullptr,                        // no store
        nullptr,                        // no reject controller
        config
    );

    if (!pipeline.initialize()) {
        LOG_ERROR("Main", "Pipeline initialization failed");
        return 1;
    }

    auto result = pipeline.process_frame(image);

    std::cout << "\n╔══════════════════════════════════════════╗" << std::endl;
    std::cout << "║      INSPECTION RESULT                   ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════╣" << std::endl;
    std::cout << "║ Verdict:    " << edgeai::verdict_to_string(result.verdict) << std::endl;
    std::cout << "║ Detections: " << result.detections.size() << std::endl;

    for (const auto& det : result.detections) {
        std::cout << "║   → " << edgeai::defect_type_to_string(det.type)
                  << " (confidence: " << static_cast<int>(det.confidence * 100) << "%)"
                  << std::endl;
    }

    auto inf_ms = std::chrono::duration<double, std::milli>(result.inference_time).count();
    auto tot_ms = std::chrono::duration<double, std::milli>(result.total_time).count();
    std::cout << "║ Inference:  " << inf_ms << " ms" << std::endl;
    std::cout << "║ Total:      " << tot_ms << " ms" << std::endl;
    std::cout << "╚══════════════════════════════════════════╝" << std::endl;

    return (result.verdict == edgeai::Verdict::Pass) ? 0 : 2;
}

int run_pipeline(const CliArgs& args) {
    edgeai::PipelineConfig config;

    if (!args.config_path.empty()) {
        config = edgeai::ConfigLoader::load(args.config_path);
    }

    // Override with CLI args
    config.inference.model_path = args.model_path;
    config.camera.device_id = args.camera_id;
    config.enable_display = args.display;

    // Video debug mode
    if (!args.video_path.empty()) {
        config.camera.video_path = args.video_path;
        config.camera.loop_video = args.loop_video;
    }

    // ── Wire up all components (composition root) ──────────────
    auto camera = std::make_unique<edgeai::CameraManager>(config.camera);
    auto engine = std::make_unique<edgeai::OnnxEngine>(config.inference);
    auto preprocessor = std::make_unique<edgeai::Preprocessor>(
        config.inference.input_width, config.inference.input_height);
    auto postprocessor = std::make_unique<edgeai::Postprocessor>(
        config.inference.conf_threshold,
        config.inference.nms_threshold,
        config.inference.num_classes);

    auto decision = std::make_unique<edgeai::DecisionEngine>(
        config.inference.conf_threshold,
        config.inference.review_threshold);

    std::unique_ptr<edgeai::IDefectStore> store =
        std::make_unique<edgeai::DefectDatabase>(config.database_path);

    std::unique_ptr<edgeai::IRejectController> rejector;
    if (config.enable_gpio) {
        rejector = std::make_unique<edgeai::GpioController>(
            config.gpio_reject_pin, config.gpio_pulse_ms);
    }

    edgeai::DefectPipeline pipeline(
        std::move(camera),
        std::move(engine),
        std::move(preprocessor),
        std::move(postprocessor),
        std::move(decision),
        std::move(store),
        std::move(rejector),
        config
    );

    // Register result callback for real-time logging
    pipeline.on_result([](const edgeai::InspectionResult& result) {
        if (result.verdict == edgeai::Verdict::Reject) {
            auto inf_ms = std::chrono::duration<double, std::milli>(result.inference_time).count();
            LOG_INFO("Result", "REJECT Frame #" + std::to_string(result.frame_id)
                     + " | " + std::to_string(result.detections.size()) + " defect(s)"
                     + " | " + std::to_string(inf_ms) + " ms");
        }
    });

    if (!pipeline.initialize()) {
        LOG_FATAL("Main", "Pipeline initialization failed");
        return 1;
    }

    pipeline.start();

    LOG_INFO("Main", "Pipeline running. Press Ctrl+C to stop.");

    // Main loop — wait for signal
    while (g_running.load() && pipeline.is_running()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // Periodic stats output
        auto stats = pipeline.stats();
        if (stats.total_frames > 0 && stats.total_frames % 100 == 0) {
            LOG_INFO("Stats",
                "Frames: " + std::to_string(stats.total_frames)
                + " | Defects: " + std::to_string(stats.defective_frames)
                + " (" + std::to_string(stats.defect_rate) + "%)"
                + " | Avg latency: " + std::to_string(stats.avg_total_ms) + " ms"
                + " | FPS: " + std::to_string(stats.throughput_fps));
        }
    }

    pipeline.stop();

    // Final report
    auto stats = pipeline.stats();
    std::cout << "\n╔══════════════════════════════════════════╗" << std::endl;
    std::cout << "║         SESSION SUMMARY                  ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════╣" << std::endl;
    std::cout << "║ Total Inspected: " << stats.total_frames << std::endl;
    std::cout << "║ Passed:          " << stats.passed_frames << std::endl;
    std::cout << "║ Rejected:        " << stats.defective_frames << std::endl;
    std::cout << "║ Review:          " << stats.review_frames << std::endl;
    std::cout << "║ Defect Rate:     " << stats.defect_rate << "%" << std::endl;
    std::cout << "║ Avg Inference:   " << stats.avg_inference_ms << " ms" << std::endl;
    std::cout << "║ Avg Total:       " << stats.avg_total_ms << " ms" << std::endl;
    std::cout << "║ Throughput:      " << stats.throughput_fps << " fps" << std::endl;
    std::cout << "║ ── Defect Breakdown ──" << std::endl;
    std::cout << "║   Dents:         " << stats.dent_count << std::endl;
    std::cout << "║   Wrong Label:   " << stats.wrong_label_count << std::endl;
    std::cout << "║   Missing Label: " << stats.missing_label_count << std::endl;
    std::cout << "║   Seal Defect:   " << stats.seal_defect_count << std::endl;
    std::cout << "║   Color Mismatch:" << stats.color_mismatch_count << std::endl;
    std::cout << "╚══════════════════════════════════════════╝" << std::endl;

    return 0;
}

int main(int argc, char* argv[]) {
    // Register signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    auto args = parse_args(argc, argv);

    if (args.help) {
        print_usage();
        return 0;
    }

    // Configure logging
    if (args.verbose) {
        edgeai::Logger::instance().set_level(edgeai::LogLevel::DEBUG);
    }
    edgeai::Logger::instance().set_file("edgeai.log");

    LOG_INFO("Main", "EdgeAI Paint Can Defect Inspector v1.0.0");
    LOG_INFO("Main", "All processing runs locally — zero cloud dependency");

    // Route to appropriate mode
    if (!args.image_path.empty()) {
        return run_single_image(args);
    }
    return run_pipeline(args);
}
