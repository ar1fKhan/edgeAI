/**
 * @file camera_manager.cpp
 * @brief Camera acquisition implementation.
 */

#include "edgeai/camera/camera_manager.h"
#include "edgeai/common/logger.h"
#include <opencv2/videoio.hpp>
#include <thread>

namespace edgeai {

CameraManager::CameraManager(const CameraConfig& config)
    : config_(config) {}

CameraManager::~CameraManager() {
    close();
}

CameraManager::CameraManager(CameraManager&& other) noexcept
    : config_(std::move(other.config_))
    , capture_(std::move(other.capture_))
    , frame_counter_(other.frame_counter_.load())
    , opened_(other.opened_) {
    other.opened_ = false;
}

CameraManager& CameraManager::operator=(CameraManager&& other) noexcept {
    if (this != &other) {
        close();
        config_ = std::move(other.config_);
        capture_ = std::move(other.capture_);
        frame_counter_.store(other.frame_counter_.load());
        opened_ = other.opened_;
        other.opened_ = false;
    }
    return *this;
}

bool CameraManager::open() {
    capture_ = std::make_unique<cv::VideoCapture>();

    // ── Video file mode ────────────────────────────────────
    if (!config_.video_path.empty()) {
        LOG_INFO("Camera", "Opening video file: " + config_.video_path);

        if (!capture_->open(config_.video_path)) {
            LOG_ERROR("Camera", "Failed to open video file: " + config_.video_path);
            return false;
        }

        is_video_ = true;
        video_fps_ = capture_->get(cv::CAP_PROP_FPS);
        if (video_fps_ <= 0) video_fps_ = 30.0;

        int total_frames = static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_COUNT));
        opened_ = true;

        LOG_INFO("Camera", "Video opened. Resolution: "
                 + std::to_string(static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_WIDTH)))
                 + "x" + std::to_string(static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_HEIGHT)))
                 + " @ " + std::to_string(static_cast<int>(video_fps_)) + " fps"
                 + " | " + std::to_string(total_frames) + " frames"
                 + (config_.loop_video ? " [LOOP]" : ""));
        return true;
    }

    // ── Live camera mode ───────────────────────────────────
    int backend = backend_id();
    LOG_INFO("Camera", "Opening camera device " + std::to_string(config_.device_id)
             + " with backend: " + config_.backend);

    if (!capture_->open(config_.device_id, backend)) {
        LOG_ERROR("Camera", "Failed to open camera device " + std::to_string(config_.device_id));
        return false;
    }

    configure_camera();
    opened_ = true;

    LOG_INFO("Camera", "Camera opened successfully. Resolution: "
             + std::to_string(static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_WIDTH)))
             + "x" + std::to_string(static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_HEIGHT)))
             + " @ " + std::to_string(static_cast<int>(capture_->get(cv::CAP_PROP_FPS))) + " fps");

    return true;
}

void CameraManager::close() {
    if (capture_ && capture_->isOpened()) {
        capture_->release();
        LOG_INFO("Camera", "Camera closed");
    }
    opened_ = false;
}

Frame CameraManager::grab_frame() {
    Frame frame;
    frame.id = ++frame_counter_;
    frame.captured_at = Clock::now();

    if (!capture_ || !capture_->isOpened()) {
        LOG_ERROR("Camera", "Attempted to grab frame from closed camera");
        return frame;
    }

    if (!capture_->read(frame.image)) {
        // ── End of video file handling ──────────────────────
        if (is_video_) {
            if (config_.loop_video) {
                capture_->set(cv::CAP_PROP_POS_FRAMES, 0);
                LOG_INFO("Camera", "Video looped — rewinding to start");
                if (!capture_->read(frame.image)) {
                    LOG_ERROR("Camera", "Failed to read after video rewind");
                    frame.image = cv::Mat();
                }
            } else {
                // Log once, then close so capture_loop sees is_open() == false
                LOG_INFO("Camera", "End of video file reached after "
                         + std::to_string(frame.id - 1) + " frames");
                close();
            }
        } else {
            LOG_WARN("Camera", "Failed to grab frame #" + std::to_string(frame.id));
            frame.image = cv::Mat();
        }
    }

    // Throttle video playback to match original FPS
    if (is_video_ && frame.is_valid()) {
        auto frame_duration = std::chrono::milliseconds(
            static_cast<int>(1000.0 / video_fps_));
        std::this_thread::sleep_for(frame_duration);
    }

    return frame;
}

bool CameraManager::is_open() const {
    return opened_ && capture_ && capture_->isOpened();
}

void CameraManager::configure_camera() {
    if (!capture_) return;

    capture_->set(cv::CAP_PROP_FRAME_WIDTH, config_.width);
    capture_->set(cv::CAP_PROP_FRAME_HEIGHT, config_.height);
    capture_->set(cv::CAP_PROP_FPS, config_.fps);

    // Set buffer size to 1 to minimize latency (always get latest frame)
    capture_->set(cv::CAP_PROP_BUFFERSIZE, 1);

    // Disable auto-exposure for consistent lighting in industrial setting
    if (config_.exposure_us > 0) {
        capture_->set(cv::CAP_PROP_AUTO_EXPOSURE, 1);  // manual mode
        capture_->set(cv::CAP_PROP_EXPOSURE, config_.exposure_us);
    }

    if (config_.gain > 0) {
        capture_->set(cv::CAP_PROP_GAIN, config_.gain);
    }
}

int CameraManager::backend_id() const {
    if (config_.backend == "v4l2")  return cv::CAP_V4L2;
    if (config_.backend == "gige")  return cv::CAP_GSTREAMER;
    if (config_.backend == "usb3")  return cv::CAP_V4L2;
    if (config_.backend == "gstreamer") return cv::CAP_GSTREAMER;
    return cv::CAP_ANY;
}

}  // namespace edgeai
