#pragma once

#include <gmock/gmock.h>
#include "edgeai/inference/ipostprocessor.h"

namespace edgeai {

class MockPostprocessor : public IPostprocessor {
public:
    MOCK_METHOD(std::vector<Detection>, process,
        (const std::vector<float>& output,
         int output_rows, int output_cols,
         float scale_x, float scale_y,
         int pad_x, int pad_y,
         int orig_w, int orig_h),
        (const, override));

    MOCK_METHOD(std::vector<Detection>, apply_nms,
        (std::vector<Detection>& detections),
        (const, override));
};

}  // namespace edgeai
