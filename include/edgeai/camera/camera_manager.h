#pragma once

/**
 * @file camera_manager.h
 * @brief Camera acquisition with frame buffering and trigger support.
 * 
 * Supports USB, V4L2, and GigE camera backends via OpenCV.
 * Provides a producer interface for the frame queue.
 */

#include <atomic>
#include <functional>
#include <memory>
#include <opencv2/videoio.hpp>
#include "edgeai/camera/icamera.h"
#include "edgeai/common/types.h"

namespace edgeai {

class CameraManager : public ICamera {
public:
    explicit CameraManager(const CameraConfig& config);
    ~CameraManager();

    // Non-copyable, moveable
    CameraManager(const CameraManager&) = delete;
    CameraManager& operator=(const CameraManager&) = delete;
    CameraManager(CameraManager&&) noexcept;
    CameraManager& operator=(CameraManager&&) noexcept;

    /// Open camera device and configure parameters
    bool open() override;

    /// Close camera and release resources
    void close() override;

    /// Grab a single frame (blocking). Returns empty Frame on failure.
    [[nodiscard]] Frame grab_frame() override;

    /// Check if camera is opened and operational
    [[nodiscard]] bool is_open() const override;

    /// Get current frame counter
    [[nodiscard]] uint64_t frame_count() const { return frame_counter_; }

private:
    CameraConfig              config_;
    std::unique_ptr<cv::VideoCapture> capture_;
    std::atomic<uint64_t>     frame_counter_{0};
    bool                      opened_ = false;
    bool                      is_video_ = false;
    double                    video_fps_ = 30.0;

    void configure_camera();
    int  backend_id() const;
};

}  // namespace edgeai
