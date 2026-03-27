/**
 * @file test_preprocessing.cpp
 * @brief Unit tests for image preprocessing pipeline.
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "edgeai/inference/preprocessing.h"

using namespace edgeai;

class PreprocessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        preprocessor = std::make_unique<Preprocessor>(640, 640);
    }

    std::unique_ptr<Preprocessor> preprocessor;
};

TEST_F(PreprocessorTest, OutputTensorSize) {
    cv::Mat image(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    auto result = preprocessor->process(image);

    // Expected: 3 channels * 640 * 640
    EXPECT_EQ(result.tensor.size(), static_cast<size_t>(3 * 640 * 640));
}

TEST_F(PreprocessorTest, PreservesOriginalDimensions) {
    cv::Mat image(1080, 1920, CV_8UC3, cv::Scalar(100, 150, 200));
    auto result = preprocessor->process(image);

    EXPECT_EQ(result.orig_w, 1920);
    EXPECT_EQ(result.orig_h, 1080);
}

TEST_F(PreprocessorTest, NormalizedValues) {
    cv::Mat image(640, 640, CV_8UC3, cv::Scalar(255, 255, 255));
    auto result = preprocessor->process(image);

    // All values should be close to 1.0 (white normalized)
    for (float val : result.tensor) {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f + 1e-6f);
    }
}

TEST_F(PreprocessorTest, SquareImageNoPadding) {
    cv::Mat image(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    auto result = preprocessor->process(image);

    EXPECT_FLOAT_EQ(result.scale_x, 1.0f);
    EXPECT_FLOAT_EQ(result.scale_y, 1.0f);
    EXPECT_EQ(result.pad_x, 0);
    EXPECT_EQ(result.pad_y, 0);
}

TEST_F(PreprocessorTest, WideImageHasVerticalPadding) {
    cv::Mat image(480, 1920, CV_8UC3, cv::Scalar(0, 0, 0));
    auto result = preprocessor->process(image);

    // Width-dominated → vertical padding expected
    EXPECT_GT(result.pad_y, 0);
}

TEST_F(PreprocessorTest, SimplePreprocess) {
    cv::Mat image(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    auto tensor = preprocessor->simple_preprocess(image);

    EXPECT_EQ(tensor.size(), static_cast<size_t>(3 * 640 * 640));
}
