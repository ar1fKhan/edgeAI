#pragma once

/**
 * @file mock_reject_controller.h
 * @brief GMock implementation of IRejectController for integration testing.
 */

#include <gmock/gmock.h>
#include "edgeai/io/ireject_controller.h"

namespace edgeai::testing {

class MockRejectController : public IRejectController {
public:
    MOCK_METHOD(bool, initialize, (), (override));
    MOCK_METHOD(void, trigger_reject, (), (override));
    MOCK_METHOD(void, cleanup, (), (override));
    MOCK_METHOD(bool, is_initialized, (), (const, override));
};

}  // namespace edgeai::testing
