#pragma once

#include <gmock/gmock.h>
#include "edgeai/inference/ipreprocessor.h"

namespace edgeai {

class MockPreprocessor : public IPreprocessor {
public:
    MOCK_METHOD(PreprocessResult, process, (const cv::Mat& image), (const, override));
};

}  // namespace edgeai
