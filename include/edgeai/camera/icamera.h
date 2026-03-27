#pragma once

/**
 * @file icamera.h
 * @brief Abstract camera interface — enables swapping real cameras with simulators.
 *
 * Implement this interface to support any camera backend:
 *   - USB/V4L2 (CameraManager)
 *   - GigE Vision
 *   - Simulated/recorded video
 *   - Multi-camera rigs
 */

#include "edgeai/common/types.h"

namespace edgeai {

class ICamera {
public:
    virtual ~ICamera() = default;

    /// Open camera device and configure parameters
    virtual bool open() = 0;

    /// Close camera and release resources
    virtual void close() = 0;

    /// Grab a single frame (blocking). Returns empty Frame on failure.
    [[nodiscard]] virtual Frame grab_frame() = 0;

    /// Check if camera is opened and operational
    [[nodiscard]] virtual bool is_open() const = 0;
};

}  // namespace edgeai
