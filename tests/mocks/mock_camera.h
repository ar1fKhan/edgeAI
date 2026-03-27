#pragma once

/**
 * @file mock_camera.h
 * @brief GMock implementation of ICamera for integration testing.
 */

#include <gmock/gmock.h>
#include "edgeai/camera/icamera.h"

namespace edgeai::testing {

class MockCamera : public ICamera {
public:
    MOCK_METHOD(bool, open, (), (override));
    MOCK_METHOD(void, close, (), (override));
    MOCK_METHOD(Frame, grab_frame, (), (override));
    MOCK_METHOD(bool, is_open, (), (const, override));
};

}  // namespace edgeai::testing
