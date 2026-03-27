#pragma once

/**
 * @file mock_decision_engine.h
 * @brief GMock implementation of IDecisionEngine for integration testing.
 */

#include <gmock/gmock.h>
#include "edgeai/decision/idecision_engine.h"

namespace edgeai::testing {

class MockDecisionEngine : public IDecisionEngine {
public:
    MOCK_METHOD(Verdict, decide, (const std::vector<Detection>&), (const, override));
};

}  // namespace edgeai::testing
