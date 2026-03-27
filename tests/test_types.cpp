/**
 * @file test_types.cpp
 * @brief Unit tests for core type definitions.
 */

#include <gtest/gtest.h>
#include "edgeai/common/types.h"

using namespace edgeai;

// ── DefectType Tests ───────────────────────────────────────────

TEST(DefectTypeTest, StringConversion) {
    EXPECT_STREQ(defect_type_to_string(DefectType::Dent), "dent");
    EXPECT_STREQ(defect_type_to_string(DefectType::WrongLabel), "wrong_label");
    EXPECT_STREQ(defect_type_to_string(DefectType::MissingLabel), "missing_label");
    EXPECT_STREQ(defect_type_to_string(DefectType::SealDefect), "seal_defect");
    EXPECT_STREQ(defect_type_to_string(DefectType::ColorMismatch), "color_mismatch");
    EXPECT_STREQ(defect_type_to_string(DefectType::None), "none");
}

TEST(DefectTypeTest, StringToType) {
    EXPECT_EQ(string_to_defect_type("dent"), DefectType::Dent);
    EXPECT_EQ(string_to_defect_type("wrong_label"), DefectType::WrongLabel);
    EXPECT_EQ(string_to_defect_type("unknown_type"), DefectType::Unknown);
}

// ── BoundingBox Tests ──────────────────────────────────────────

TEST(BoundingBoxTest, PixelRect) {
    BoundingBox bbox{0.1f, 0.2f, 0.3f, 0.4f};
    auto rect = bbox.to_pixel_rect(1000, 500);

    EXPECT_EQ(rect.x, 100);
    EXPECT_EQ(rect.y, 100);
    EXPECT_EQ(rect.width, 300);
    EXPECT_EQ(rect.height, 200);
}

TEST(BoundingBoxTest, Area) {
    BoundingBox bbox{0.0f, 0.0f, 0.5f, 0.5f};
    EXPECT_FLOAT_EQ(bbox.area(), 0.25f);
}

TEST(BoundingBoxTest, IoU_Identical) {
    BoundingBox a{0.1f, 0.1f, 0.5f, 0.5f};
    BoundingBox b{0.1f, 0.1f, 0.5f, 0.5f};
    EXPECT_FLOAT_EQ(a.iou(b), 1.0f);
}

TEST(BoundingBoxTest, IoU_NoOverlap) {
    BoundingBox a{0.0f, 0.0f, 0.1f, 0.1f};
    BoundingBox b{0.5f, 0.5f, 0.1f, 0.1f};
    EXPECT_FLOAT_EQ(a.iou(b), 0.0f);
}

TEST(BoundingBoxTest, IoU_Partial) {
    BoundingBox a{0.0f, 0.0f, 0.4f, 0.4f};
    BoundingBox b{0.2f, 0.2f, 0.4f, 0.4f};
    float iou = a.iou(b);
    EXPECT_GT(iou, 0.0f);
    EXPECT_LT(iou, 1.0f);
}

// ── Verdict Tests ──────────────────────────────────────────────

TEST(VerdictTest, StringConversion) {
    EXPECT_STREQ(verdict_to_string(Verdict::Pass), "PASS");
    EXPECT_STREQ(verdict_to_string(Verdict::Reject), "REJECT");
    EXPECT_STREQ(verdict_to_string(Verdict::Review), "REVIEW");
}

// ── InspectionResult Tests ─────────────────────────────────────

TEST(InspectionResultTest, HasDefects) {
    InspectionResult result;
    EXPECT_FALSE(result.has_defects());

    result.detections.push_back({DefectType::Dent, 0.95f, {}, 0});
    EXPECT_TRUE(result.has_defects());
}

TEST(InspectionResultTest, MaxConfidence) {
    InspectionResult result;
    EXPECT_FLOAT_EQ(result.max_confidence(), 0.0f);

    result.detections.push_back({DefectType::Dent, 0.7f, {}, 0});
    result.detections.push_back({DefectType::WrongLabel, 0.95f, {}, 1});
    result.detections.push_back({DefectType::SealDefect, 0.5f, {}, 3});
    EXPECT_FLOAT_EQ(result.max_confidence(), 0.95f);
}

// ── PipelineStats Tests ────────────────────────────────────────

TEST(PipelineStatsTest, Update) {
    PipelineStats stats;
    
    InspectionResult result;
    result.verdict = Verdict::Reject;
    result.inference_time = std::chrono::microseconds(15000);  // 15ms
    result.total_time = std::chrono::microseconds(25000);      // 25ms
    result.detections.push_back({DefectType::Dent, 0.9f, {}, 0});

    stats.update(result);

    EXPECT_EQ(stats.total_frames, 1u);
    EXPECT_EQ(stats.defective_frames, 1u);
    EXPECT_EQ(stats.dent_count, 1u);
    EXPECT_GT(stats.defect_rate, 0.0);
}
