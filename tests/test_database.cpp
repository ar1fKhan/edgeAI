/**
 * @file test_database.cpp
 * @brief Unit tests for SQLite defect database.
 */

#include <gtest/gtest.h>
#include <filesystem>
#include "edgeai/db/defect_database.h"

using namespace edgeai;

class DatabaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        db_path_ = "/tmp/test_edgeai_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + ".db";
        db_ = std::make_unique<DefectDatabase>(db_path_);
        ASSERT_TRUE(db_->open());
    }

    void TearDown() override {
        db_->close();
        std::filesystem::remove(db_path_);
    }

    std::string db_path_;
    std::unique_ptr<DefectDatabase> db_;
};

TEST_F(DatabaseTest, OpenAndClose) {
    // Already opened in SetUp
    EXPECT_EQ(db_->total_records(), 0);
}

TEST_F(DatabaseTest, InsertAndRetrieve) {
    InspectionResult result;
    result.frame_id = 42;
    result.timestamp = Clock::now();
    result.verdict = Verdict::Reject;
    result.inference_time = std::chrono::microseconds(15000);
    result.total_time = std::chrono::microseconds(25000);
    result.image_path = "/data/defects/test.jpg";

    Detection det;
    det.type = DefectType::Dent;
    det.confidence = 0.92f;
    det.bbox = {0.1f, 0.2f, 0.3f, 0.4f};
    det.class_id = 0;
    result.detections.push_back(det);

    ASSERT_TRUE(db_->insert_result(result));
    EXPECT_EQ(db_->total_records(), 1);

    auto defects = db_->get_recent_defects(10);
    ASSERT_EQ(defects.size(), 1u);
    EXPECT_EQ(defects[0].defect_type, "dent");
    EXPECT_NEAR(defects[0].confidence, 0.92f, 0.01f);
}

TEST_F(DatabaseTest, DefectDistribution) {
    // Insert multiple defect types
    for (int i = 0; i < 5; ++i) {
        InspectionResult result;
        result.frame_id = i;
        result.verdict = Verdict::Reject;
        result.inference_time = std::chrono::microseconds(10000);
        result.total_time = std::chrono::microseconds(20000);

        Detection det;
        det.type = DefectType::Dent;
        det.confidence = 0.9f;
        det.bbox = {};
        det.class_id = 0;
        result.detections.push_back(det);
        db_->insert_result(result);
    }

    for (int i = 0; i < 3; ++i) {
        InspectionResult result;
        result.frame_id = 10 + i;
        result.verdict = Verdict::Reject;
        result.inference_time = std::chrono::microseconds(10000);
        result.total_time = std::chrono::microseconds(20000);

        Detection det;
        det.type = DefectType::WrongLabel;
        det.confidence = 0.85f;
        det.bbox = {};
        det.class_id = 1;
        result.detections.push_back(det);
        db_->insert_result(result);
    }

    auto dist = db_->get_defect_distribution();
    ASSERT_GE(dist.size(), 2u);
    EXPECT_EQ(dist[0].first, "dent");
    EXPECT_EQ(dist[0].second, 5);
}

TEST_F(DatabaseTest, OverallDefectRate) {
    // 2 rejects + 3 passes = 40% defect rate
    for (int i = 0; i < 2; ++i) {
        InspectionResult result;
        result.frame_id = i;
        result.verdict = Verdict::Reject;
        result.inference_time = std::chrono::microseconds(10000);
        result.total_time = std::chrono::microseconds(20000);
        result.detections.push_back({DefectType::Dent, 0.9f, {}, 0});
        db_->insert_result(result);
    }
    for (int i = 0; i < 3; ++i) {
        InspectionResult result;
        result.frame_id = 10 + i;
        result.verdict = Verdict::Pass;
        result.inference_time = std::chrono::microseconds(10000);
        result.total_time = std::chrono::microseconds(20000);
        db_->insert_result(result);
    }

    double rate = db_->overall_defect_rate();
    EXPECT_NEAR(rate, 40.0, 0.1);
}
