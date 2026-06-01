#pragma once

/**
 * @file defect_pipeline.h
 * @brief Pipeline orchestrator — infrastructure only, no domain logic.
 *
 * The pipeline:
 *   - Orchestrates threads (capture, processing)
 *   - Passes data between modules
 *   - Handles lifecycle (start/stop/cleanup)
 *
 * The pipeline does NOT:
 *   - Contain inference logic (delegated to IInferenceEngine)
 *   - Contain business logic (delegated to IDecisionEngine)
 *   - Contain storage logic (delegated to IDefectStore)
 *   - Know about specific defect types or thresholds
 *
 * ── Thread Ownership ───────────────────────────────────────────
 *   Thread 1 (capture):    ICamera::grab_frame() → FrameQueue
 *   Thread 2 (processing): FrameQueue → Preprocess → Infer → Postprocess
 *                           → Decide → Store/Reject/Display
 *
 * All components are injected via interfaces for testability and swapping.
 */

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

// ── Interfaces (depend only on these — not concrete types) ─────
#include "edgeai/camera/icamera.h"
#include "edgeai/inference/iengine.h"
#include "edgeai/inference/ipreprocessor.h"
#include "edgeai/inference/ipostprocessor.h"
#include "edgeai/decision/idecision_engine.h"
#include "edgeai/db/idefect_store.h"
#include "edgeai/io/ireject_controller.h"

// ── Pipeline internals ────────────────────────────────────────
#include "edgeai/pipeline/frame_queue.h"

// ── Common ─────────────────────────────────────────────────────
#include "edgeai/common/types.h"
#include "edgeai/common/timer.h"

namespace edgeai {

class DefectPipeline {
public:
    using ResultCallback = std::function<void(const InspectionResult&)>;

    /**
     * @brief Construct pipeline with injected components.
     *
     * @param camera           Camera source (nullable for single-image mode)
     * @param engine           Inference backend (required)
     * @param preprocessor     Image preprocessor (required)
     * @param postprocessor    Model output decoder (required)
     * @param decision         Business logic engine (required)
     * @param store            Defect storage backend (nullable — logging disabled)
     * @param rejector         Reject mechanism (nullable — rejection disabled)
     * @param config           Pipeline configuration
     */
    DefectPipeline(
        std::unique_ptr<ICamera>            camera,
        std::unique_ptr<IInferenceEngine>   engine,
        std::unique_ptr<IPreprocessor>      preprocessor,
        std::unique_ptr<IPostprocessor>     postprocessor,
        std::unique_ptr<IDecisionEngine>    decision,
        std::unique_ptr<IDefectStore>       store,
        std::unique_ptr<IRejectController>  rejector,
        const PipelineConfig&               config
    );

    ~DefectPipeline();

    // Non-copyable
    DefectPipeline(const DefectPipeline&) = delete;
    DefectPipeline& operator=(const DefectPipeline&) = delete;

    /// Initialize all injected components (open camera, load model, etc.)
    bool initialize();

    /// Start the pipeline (non-blocking, spawns threads)
    void start();

    /// Stop the pipeline gracefully
    void stop();

    /// Check if pipeline is running
    [[nodiscard]] bool is_running() const { return running_.load(); }

    /// Get current pipeline statistics (thread-safe snapshot)
    [[nodiscard]] PipelineStats stats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return stats_;
    }

    /// Register a callback for each inspection result
    /// Must be called BEFORE start() — not safe to call while running.
    void on_result(ResultCallback callback) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        result_callback_ = std::move(callback);
    }

    /// Process a single frame (for testing/benchmarking, bypasses camera)
    [[nodiscard]] InspectionResult process_frame(const cv::Mat& image);

private:
    PipelineConfig                      config_;
    std::atomic<bool>                   running_{false};

    // ── Injected Components (interfaces → swappable) ───────────
    std::unique_ptr<ICamera>            camera_;
    std::unique_ptr<IInferenceEngine>   engine_;
    std::unique_ptr<IPreprocessor>      preprocessor_;
    std::unique_ptr<IPostprocessor>     postprocessor_;
    std::unique_ptr<IDecisionEngine>    decision_;
    std::unique_ptr<IDefectStore>       store_;
    std::unique_ptr<IRejectController>  rejector_;

    // ── Threading ──────────────────────────────────────────────
    FrameQueue                          frame_queue_;
    std::thread                         capture_thread_;
    std::thread                         inference_thread_;
    std::thread                         display_thread_;
    std::thread                         watchdog_thread_;

    // ── Display slot (latest annotated frame for display thread) ──
    std::mutex                          display_mutex_;
    std::condition_variable             display_cv_;
    std::optional<cv::Mat>              display_frame_;

    // ── Watchdog heartbeat ─────────────────────────────────────
    std::atomic<int64_t>                last_inference_ms_{0};

    // ── Stats (guarded by stats_mutex_) ────────────────────────
    mutable std::mutex                  stats_mutex_;
    PipelineStats                       stats_;
    LatencyTracker                      latency_tracker_;

    // ── Callbacks (guarded by stats_mutex_) ────────────────────
    ResultCallback                      result_callback_;

    // Thread functions
    void capture_loop();
    void inference_loop();
    void display_loop();
    void watchdog_loop();

    // Helpers
    void handle_result(const InspectionResult& result, const cv::Mat& image);
    std::string save_defect_image(const cv::Mat& image, uint64_t frame_id);
    void draw_detections(cv::Mat& image, const std::vector<Detection>& detections);
    void prune_if_needed();
};

}  // namespace edgeai
