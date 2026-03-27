#pragma once

/**
 * @file mock_engine.h
 * @brief GMock implementation of IInferenceEngine for integration testing.
 */

#include <gmock/gmock.h>
#include "edgeai/inference/iengine.h"

namespace edgeai::testing {

class MockEngine : public IInferenceEngine {
public:
    MOCK_METHOD(bool, load_model, (), (override));
    MOCK_METHOD(std::vector<float>, infer, (const std::vector<float>&), (override));
    MOCK_METHOD(bool, is_loaded, (), (const, override));
    MOCK_METHOD(int, input_width, (), (const, override));
    MOCK_METHOD(int, input_height, (), (const, override));
};

}  // namespace edgeai::testing
