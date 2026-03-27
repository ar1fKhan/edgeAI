#pragma once

/**
 * @file mock_store.h
 * @brief GMock implementation of IDefectStore for integration testing.
 */

#include <gmock/gmock.h>
#include "edgeai/db/idefect_store.h"

namespace edgeai::testing {

class MockStore : public IDefectStore {
public:
    MOCK_METHOD(bool, open, (), (override));
    MOCK_METHOD(void, close, (), (override));
    MOCK_METHOD(bool, insert_result, (const InspectionResult&), (override));
    MOCK_METHOD(int64_t, prune, (int64_t max_records), (override));
};

}  // namespace edgeai::testing
