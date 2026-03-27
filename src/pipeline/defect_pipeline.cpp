/**
 * @file defect_pipeline.cpp
 * @brief Multi-threaded defect detection pipeline with dependency injection.
 *
 * All swappable components (camera, inference, storage, reject mechanism)
 * are injected through interfaces — no concrete types created here.
 */

#include "edgeai/pipeline/defect_pipeline.h"
#include "edgeai/common/logger.h"

#include <algorithm>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

namespace fs = std::filesystem;

namespace edgeai {

DefectPipeline::DefectPipeline(
    std::unique_ptr<ICamera>            camera,
    std::unique_ptr<IInferenceEngine>   engine,
    std::unique_ptr<IPreprocessor>      preprocessor,
    std::unique_ptr<IPostprocessor>     postprocessor,
    std::unique_ptr<IDecisionEngine>    decision,
    std::unique_ptr<IDefectStore>       store,
    std::unique_ptr<IRejectController>  rejector,
    const PipelineConfig&               config)
    : config_(config)
    , camera_(std::move(camera))
    , engine_(std::move(engine))
    , preprocessor_(std::move(preprocessor))
    , postprocessor_(std::move(postprocessor))
    , decision_(std::move(decision))
    , store_(std::move(store))
    , rejector_(std::move(rejector))
    , frame_queue_(config.queue_capacity)
{}

DefectPipeline::~DefectPipeline() {
    stop();
}

bool DefectPipeline::initialize() {
    LOG_INFO("Pipeline", "Initializing defect detection pipeline...");

    // ── Camera (optional — not needed for single-image mode) ───
    if (camera_ && !camera_->open()) {
        LOG_ERROR("Pipeline", "Failed to initialize camera");
        return false;
    }

    // ── Inference Engine ───────────────────────────────────────
    if (engine_ && !engine_->load_model()) {
        LOG_ERROR("Pipeline", "Failed to load model");
        return false;
    }

    // ── Defect Store (optional) ────────────────────────────────
    if (store_ && !store_->open()) {
        LOG_WARN("Pipeline", "Failed to open defect store — logging disabled");
        store_.reset();
    }

    // ── Reject Controller (optional) ───────────────────────────
    if (rejector_ && !rejector_->initialize()) {
        LOG_WARN("Pipeline", "Reject controller init failed — rejection disabled");
        rejector_.reset();
    }

    // ── Create defect image directory ──────────────────────────
    if (config_.save_defect_images) {
        fs::create_directories(config_.defect_image_dir);
    }

    LOG_INFO("Pipeline", "Pipeline initialized successfully");
    return true;
}

void DefectPipeline::start() {
    if (running_.load()) {
        LOG_WARN("Pipeline", "Pipeline already running");
        return;
    }

    running_ = true;
    frame_queue_.reset();

    LOG_INFO("Pipeline", "Starting pipeline threads...");
    capture_thread_ = std::thread(&DefectPipeline::capture_loop, this);
    inference_thread_ = std::thread(&DefectPipeline::inference_loop, this);

    LOG_INFO("Pipeline", "Pipeline started — capture and inference threads running");
}

void DefectPipeline::stop() {
    bool was_running = running_.exchange(false);
    frame_queue_.stop();

    if (capture_thread_.joinable()) capture_thread_.join();
    if (inference_thread_.joinable()) inference_thread_.join();

    if (!was_running) return;  // already stopped — skip logging/cleanup

    LOG_INFO("Pipeline", "Stopping pipeline...");

    if (camera_) camera_->close();

    // Read stats under lock for the final log message
    PipelineStats final_stats;
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        final_stats = stats_;
    }

    LOG_INFO("Pipeline", "Pipeline stopped. Stats: "
             + std::to_string(final_stats.total_frames) + " frames, "
             + std::to_string(final_stats.defective_frames) + " defects ("
             + std::to_string(final_stats.defect_rate) + "%), "
             + "avg latency: " + std::to_string(final_stats.avg_total_ms) + " ms");
}

InspectionResult DefectPipeline::process_frame(const cv::Mat& image) {
    ScopedTimer total_timer("TotalLatency", false);
    InspectionResult result;
    result.frame_id = 0;
    result.timestamp = Clock::now();

    // Preprocess
    auto prep = preprocessor_->process(image);

    // Inference
    ScopedTimer inf_timer("InferenceLatency", false);
    auto raw_output = engine_->infer(prep.tensor);
    result.inference_time = inf_timer.elapsed();

    if (!raw_output.empty()) {
        // Parse output
        int cols = 5 + config_.inference.num_classes;
        int rows = static_cast<int>(raw_output.size()) / cols;

        auto detections = postprocessor_->process(
            raw_output, rows, cols,
            prep.scale_x, prep.scale_y,
            prep.pad_x, prep.pad_y,
            prep.orig_w, prep.orig_h
        );

        result.detections = postprocessor_->apply_nms(detections);
    }

    // Decision — business logic (separate from model decoding)
    result.verdict = decision_->decide(result.detections);
    result.total_time = total_timer.elapsed();

    return result;
}

// ── Private Thread Functions ───────────────────────────────────

void DefectPipeline::capture_loop() {
    LOG_INFO("Pipeline", "Capture thread started");

    while (running_.load()) {
        try {
            // Exit if camera was closed (e.g. end of video file)
            if (!camera_->is_open()) {
                LOG_INFO("Pipeline", "Camera closed — stopping capture loop");
                running_.store(false);
                break;
            }

            auto frame = camera_->grab_frame();
            if (!frame.is_valid()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // Push frame to queue (drop if full to avoid latency buildup)
            if (!frame_queue_.push(std::move(frame), std::chrono::milliseconds(5))) {
                LOG_DEBUG("Pipeline", "Frame queue full — dropping frame");
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Pipeline", "Capture thread exception: " + std::string(e.what()));
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } catch (...) {
            LOG_ERROR("Pipeline", "Capture thread unknown exception");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    LOG_INFO("Pipeline", "Capture thread stopped");
}

void DefectPipeline::inference_loop() {
    LOG_INFO("Pipeline", "Inference thread started");
    uint64_t frames_since_prune = 0;
    constexpr uint64_t prune_interval = 1000;  // prune check every N frames

    while (running_.load()) {
        try {
            auto frame_opt = frame_queue_.pop(std::chrono::milliseconds(50));
            if (!frame_opt.has_value()) continue;

            auto& frame = frame_opt.value();
            if (!frame.is_valid()) continue;

            ScopedTimer total_timer("TotalLatency", false);

            // Preprocess
            auto prep = preprocessor_->process(frame.image);

            // Inference
            ScopedTimer inf_timer("InferenceLatency", false);
            auto raw_output = engine_->infer(prep.tensor);
            auto inference_time = inf_timer.elapsed();

            // Build result
            InspectionResult result;
            result.frame_id = frame.id;
            result.timestamp = frame.captured_at;
            result.inference_time = inference_time;

            if (!raw_output.empty()) {
                int cols = 5 + config_.inference.num_classes;
                int rows = static_cast<int>(raw_output.size()) / cols;

                auto detections = postprocessor_->process(
                    raw_output, rows, cols,
                    prep.scale_x, prep.scale_y,
                    prep.pad_x, prep.pad_y,
                    prep.orig_w, prep.orig_h
                );

                result.detections = postprocessor_->apply_nms(detections);
            }

            // Decision — business logic (separate from model decoding)
            result.verdict = decision_->decide(result.detections);
            result.total_time = total_timer.elapsed();

            // Handle result
            handle_result(result, frame.image);

            // Update stats (thread-safe)
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.update(result);
            }

            // Periodic disk pruning (images + DB)
            if (++frames_since_prune >= prune_interval) {
                frames_since_prune = 0;
                prune_if_needed();
            }

            // Display if enabled
            if (config_.enable_display) {
                cv::Mat display = frame.image.clone();
                draw_detections(display, result.detections);

                // Add stats overlay
                std::string status = verdict_to_string(result.verdict);
                auto color = (result.verdict == Verdict::Pass)
                    ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                cv::putText(display, status, cv::Point(10, 40),
                            cv::FONT_HERSHEY_SIMPLEX, 1.2, color, 3);

                double fps;
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    fps = stats_.throughput_fps;
                }
                std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
                cv::putText(display, fps_text, cv::Point(10, 80),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

                cv::imshow("EdgeAI Inspector", display);
                if (cv::waitKey(1) == 27) {  // ESC to quit
                    running_ = false;
                }
            }

            // Invoke callback
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                if (result_callback_) {
                    result_callback_(result);
                }
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Pipeline", "Inference thread exception: " + std::string(e.what()));
        } catch (...) {
            LOG_ERROR("Pipeline", "Inference thread unknown exception");
        }
    }

    LOG_INFO("Pipeline", "Inference thread stopped");
}

void DefectPipeline::handle_result(const InspectionResult& result, const cv::Mat& image) {
    // Log to defect store
    if (store_) {
        InspectionResult store_result = result;
        if (config_.save_defect_images && result.has_defects()) {
            store_result.image_path = save_defect_image(image, result.frame_id);
        }
        store_->insert_result(store_result);
    }

    // Trigger reject controller
    if (result.verdict == Verdict::Reject && rejector_ && rejector_->is_initialized()) {
        rejector_->trigger_reject();
        LOG_INFO("Pipeline", "REJECT — Frame #" + std::to_string(result.frame_id)
                 + " | " + std::to_string(result.detections.size()) + " defect(s) detected"
                 + " | max confidence: " + std::to_string(result.max_confidence()));
    }
}

std::string DefectPipeline::save_defect_image(const cv::Mat& image, uint64_t frame_id) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
    localtime_r(&time, &tm);

    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);

    std::string filename = config_.defect_image_dir + "/defect_"
        + std::string(buf) + "_" + std::to_string(frame_id) + ".jpg";

    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
    cv::imwrite(filename, image, params);

    return filename;
}

void DefectPipeline::draw_detections(cv::Mat& image,
                                      const std::vector<Detection>& detections) {
    static const cv::Scalar colors[] = {
        {0, 0, 255},    // Dent - Red
        {255, 0, 0},    // WrongLabel - Blue
        {0, 165, 255},  // MissingLabel - Orange
        {255, 255, 0},  // SealDefect - Cyan
        {128, 0, 128},  // ColorMismatch - Purple
    };

    for (const auto& det : detections) {
        auto rect = det.bbox.to_pixel_rect(image.cols, image.rows);
        int color_idx = det.class_id % 5;

        cv::rectangle(image, rect, colors[color_idx], 2);

        std::string label = std::string(defect_type_to_string(det.type))
            + " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";

        int baseline;
        auto text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);

        cv::rectangle(image,
            cv::Point(rect.x, rect.y - text_size.height - 6),
            cv::Point(rect.x + text_size.width, rect.y),
            colors[color_idx], cv::FILLED);

        cv::putText(image, label,
            cv::Point(rect.x, rect.y - 4),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    }
}

void DefectPipeline::prune_if_needed() {
    // Prune database records
    if (store_ && config_.max_db_records > 0) {
        store_->prune(config_.max_db_records);
    }

    // Prune old defect images (keep at most max_defect_images, delete oldest first)
    if (config_.save_defect_images && config_.max_defect_images > 0) {
        try {
            std::vector<fs::directory_entry> images;
            if (fs::exists(config_.defect_image_dir)) {
                for (const auto& entry : fs::directory_iterator(config_.defect_image_dir)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
                        images.push_back(entry);
                    }
                }
            }

            if (static_cast<int>(images.size()) > config_.max_defect_images) {
                // Sort by last write time (oldest first)
                std::sort(images.begin(), images.end(),
                    [](const fs::directory_entry& a, const fs::directory_entry& b) {
                        return fs::last_write_time(a) < fs::last_write_time(b);
                    });

                int to_delete = static_cast<int>(images.size()) - config_.max_defect_images;
                for (int i = 0; i < to_delete; ++i) {
                    std::error_code ec;
                    fs::remove(images[i].path(), ec);
                }
                LOG_INFO("Pipeline", "Pruned " + std::to_string(to_delete) + " old defect images");
            }
        } catch (const std::exception& e) {
            LOG_WARN("Pipeline", "Image pruning error: " + std::string(e.what()));
        }
    }
}

}  // namespace edgeai
