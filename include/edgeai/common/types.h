#pragma once

/**
 * @file types.h
 * @brief Core type definitions for the EdgeAI defect detection system.
 * 
 * All shared structs, enums, and type aliases used across the pipeline.
 */

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include <opencv2/core.hpp>

namespace edgeai {

// ── Time Aliases ───────────────────────────────────────────────

using Clock     = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Duration  = std::chrono::microseconds;

// ── Defect Classification ──────────────────────────────────────

enum class DefectType : uint8_t {
    None          = 0,
    Dent          = 1,
    WrongLabel    = 2,
    MissingLabel  = 3,
    SealDefect    = 4,
    ColorMismatch = 5,
    Unknown       = 255
};

inline const char* defect_type_to_string(DefectType type) {
    switch (type) {
        case DefectType::None:          return "none";
        case DefectType::Dent:          return "dent";
        case DefectType::WrongLabel:    return "wrong_label";
        case DefectType::MissingLabel:  return "missing_label";
        case DefectType::SealDefect:    return "seal_defect";
        case DefectType::ColorMismatch: return "color_mismatch";
        default:                        return "unknown";
    }
}

inline DefectType string_to_defect_type(const std::string& s) {
    if (s == "dent")           return DefectType::Dent;
    if (s == "wrong_label")    return DefectType::WrongLabel;
    if (s == "missing_label")  return DefectType::MissingLabel;
    if (s == "seal_defect")    return DefectType::SealDefect;
    if (s == "color_mismatch") return DefectType::ColorMismatch;
    if (s == "none")           return DefectType::None;
    return DefectType::Unknown;
}

// ── Bounding Box ───────────────────────────────────────────────

struct BoundingBox {
    float x;       // top-left x (normalized 0-1)
    float y;       // top-left y (normalized 0-1)
    float width;   // box width  (normalized 0-1)
    float height;  // box height (normalized 0-1)

    [[nodiscard]] cv::Rect to_pixel_rect(int img_w, int img_h) const {
        return cv::Rect(
            static_cast<int>(x * img_w),
            static_cast<int>(y * img_h),
            static_cast<int>(width * img_w),
            static_cast<int>(height * img_h)
        );
    }

    [[nodiscard]] float area() const { return width * height; }

    [[nodiscard]] float iou(const BoundingBox& other) const {
        float x1 = std::max(x, other.x);
        float y1 = std::max(y, other.y);
        float x2 = std::min(x + width, other.x + other.width);
        float y2 = std::min(y + height, other.y + other.height);

        float inter_w = std::max(0.0f, x2 - x1);
        float inter_h = std::max(0.0f, y2 - y1);
        float inter_area = inter_w * inter_h;

        float union_area = area() + other.area() - inter_area;
        return (union_area > 0.0f) ? inter_area / union_area : 0.0f;
    }
};

// ── Detection Result ───────────────────────────────────────────

struct Detection {
    DefectType  type;
    float       confidence;
    BoundingBox bbox;
    int         class_id;
};

// ── Inspection Verdict ─────────────────────────────────────────

enum class Verdict : uint8_t {
    Pass   = 0,
    Reject = 1,
    Review = 2   // borderline confidence → manual review
};

inline const char* verdict_to_string(Verdict v) {
    switch (v) {
        case Verdict::Pass:   return "PASS";
        case Verdict::Reject: return "REJECT";
        case Verdict::Review: return "REVIEW";
        default:              return "UNKNOWN";
    }
}

// ── Inspection Result (per frame) ──────────────────────────────

struct InspectionResult {
    uint64_t               frame_id;
    TimePoint              timestamp;
    Verdict                verdict;
    std::vector<Detection> detections;
    Duration               inference_time;
    Duration               total_time;
    std::string            image_path;   // saved defect image path

    [[nodiscard]] bool has_defects() const {
        return !detections.empty();
    }

    [[nodiscard]] float max_confidence() const {
        float max_conf = 0.0f;
        for (const auto& d : detections) {
            max_conf = std::max(max_conf, d.confidence);
        }
        return max_conf;
    }
};

// ── Camera Frame ───────────────────────────────────────────────

struct Frame {
    uint64_t  id;
    cv::Mat   image;
    TimePoint captured_at;

    [[nodiscard]] bool is_valid() const {
        return !image.empty();
    }
};

// ── Pipeline Statistics ────────────────────────────────────────

struct PipelineStats {
    uint64_t total_frames      = 0;
    uint64_t defective_frames  = 0;
    uint64_t passed_frames     = 0;
    uint64_t review_frames     = 0;
    double   avg_inference_ms  = 0.0;
    double   avg_total_ms      = 0.0;
    double   defect_rate       = 0.0;  // percentage
    double   throughput_fps    = 0.0;

    // Per-defect-type counters
    uint64_t dent_count          = 0;
    uint64_t wrong_label_count   = 0;
    uint64_t missing_label_count = 0;
    uint64_t seal_defect_count   = 0;
    uint64_t color_mismatch_count = 0;

    void update(const InspectionResult& result) {
        ++total_frames;
        if (result.verdict == Verdict::Reject) ++defective_frames;
        else if (result.verdict == Verdict::Pass) ++passed_frames;
        else ++review_frames;

        auto inf_ms = std::chrono::duration<double, std::milli>(result.inference_time).count();
        auto tot_ms = std::chrono::duration<double, std::milli>(result.total_time).count();
        avg_inference_ms = ((avg_inference_ms * (total_frames - 1)) + inf_ms) / total_frames;
        avg_total_ms = ((avg_total_ms * (total_frames - 1)) + tot_ms) / total_frames;

        defect_rate = (static_cast<double>(defective_frames) / total_frames) * 100.0;
        throughput_fps = 1000.0 / avg_total_ms;

        for (const auto& det : result.detections) {
            switch (det.type) {
                case DefectType::Dent:          ++dent_count; break;
                case DefectType::WrongLabel:    ++wrong_label_count; break;
                case DefectType::MissingLabel:  ++missing_label_count; break;
                case DefectType::SealDefect:    ++seal_defect_count; break;
                case DefectType::ColorMismatch: ++color_mismatch_count; break;
                default: break;
            }
        }
    }
};

// ── Configuration Structures ───────────────────────────────────

struct CameraConfig {
    int         device_id       = 0;
    int         width           = 1920;
    int         height          = 1080;
    int         fps             = 30;
    std::string backend         = "v4l2";  // v4l2, gige, usb3
    bool        trigger_mode    = false;
    int         exposure_us     = 10000;
    float       gain            = 1.0f;
    std::string video_path      = "";     // non-empty → read from video file instead of camera
    bool        loop_video      = false;  // loop video playback (useful for debugging)
};

struct InferenceConfig {
    std::string model_path      = "models/defect_detector.onnx";
    int         input_width     = 640;
    int         input_height    = 640;
    int         num_classes     = 5;
    float       conf_threshold  = 0.98f;   // 98% confidence - high threshold for undertrained synthetic model
    float       nms_threshold   = 0.15f;   // Very strict NMS
    float       review_threshold = 0.95f;  // 95% for manual review
    int         num_threads     = 4;
    bool        use_gpu         = false;
    int         gpu_device_id   = 0;
    std::string execution_provider = "cpu"; // cpu, cuda, tensorrt, openvino
};

struct PipelineConfig {
    CameraConfig    camera;
    InferenceConfig inference;
    int             queue_capacity     = 32;
    std::string     defect_image_dir   = "data/defects";
    std::string     database_path      = "data/defects.db";
    bool            save_defect_images = true;
    bool            enable_display     = false;
    bool            enable_gpio        = false;
    int             gpio_reject_pin    = 17;
    int             gpio_pulse_ms      = 200;
    int             max_defect_images  = 10000;   // 0 = unlimited
    int             max_db_records     = 100000;  // 0 = unlimited
};

}  // namespace edgeai
