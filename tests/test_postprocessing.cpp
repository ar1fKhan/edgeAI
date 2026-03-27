/**
 * @file test_postprocessing.cpp
 * @brief Unit tests for post-processing: NMS, confidence filtering, coordinate mapping.
 *
 * Verdict tests live in test_decision.cpp (DecisionEngine).
 */

#include <gtest/gtest.h>
#include "edgeai/inference/postprocessing.h"

using namespace edgeai;

class PostprocessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // conf_threshold=0.5, nms_threshold=0.45, num_classes=5
        postprocessor = std::make_unique<Postprocessor>(0.5f, 0.45f, 5);
    }
    std::unique_ptr<Postprocessor> postprocessor;
};

TEST_F(PostprocessorTest, EmptyOutputProducesNoDetections) {
    std::vector<float> output;
    auto detections = postprocessor->process(output, 0, 10, 1.0f, 1.0f, 0, 0, 640, 640);
    EXPECT_TRUE(detections.empty());
}

TEST_F(PostprocessorTest, LowConfidenceFiltered) {
    // Single detection: cx, cy, w, h, class0_conf=0.3, rest=0
    // 5 classes → row has 9 elements (4 bbox + 5 class scores)
    std::vector<float> output = {320, 320, 100, 100, 0.3f, 0.0f, 0.0f, 0.0f, 0.0f};

    auto detections = postprocessor->process(output, 1, 9, 1.0f, 1.0f, 0, 0, 640, 640);
    EXPECT_TRUE(detections.empty());  // 0.3 < 0.5 threshold
}

TEST_F(PostprocessorTest, HighConfidenceDetected) {
    std::vector<float> output = {320, 320, 100, 100, 0.9f, 0.0f, 0.0f, 0.0f, 0.0f};

    auto detections = postprocessor->process(output, 1, 9, 1.0f, 1.0f, 0, 0, 640, 640);
    EXPECT_EQ(detections.size(), 1u);
    EXPECT_EQ(detections[0].type, DefectType::Dent);  // class 0
    EXPECT_FLOAT_EQ(detections[0].confidence, 0.9f);
}

TEST_F(PostprocessorTest, NMS_SuppressesDuplicates) {
    std::vector<Detection> dets;
    // Two overlapping detections of same class
    dets.push_back({DefectType::Dent, 0.9f, {0.1f, 0.1f, 0.3f, 0.3f}, 0});
    dets.push_back({DefectType::Dent, 0.7f, {0.12f, 0.12f, 0.3f, 0.3f}, 0});

    auto result = postprocessor->apply_nms(dets);
    EXPECT_EQ(result.size(), 1u);  // second suppressed due to high IoU
    EXPECT_FLOAT_EQ(result[0].confidence, 0.9f);  // higher confidence kept
}

TEST_F(PostprocessorTest, NMS_KeepsDifferentClasses) {
    std::vector<Detection> dets;
    // Two overlapping detections of DIFFERENT classes
    dets.push_back({DefectType::Dent, 0.9f, {0.1f, 0.1f, 0.3f, 0.3f}, 0});
    dets.push_back({DefectType::WrongLabel, 0.8f, {0.12f, 0.12f, 0.3f, 0.3f}, 1});

    auto result = postprocessor->apply_nms(dets);
    EXPECT_EQ(result.size(), 2u);  // different classes → both kept
}
