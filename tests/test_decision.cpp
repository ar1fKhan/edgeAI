/**
 * @file test_decision.cpp
 * @brief Unit tests for DecisionEngine — business logic verdict determination.
 *
 * Tests the threshold-based decision logic that was separated from
 * the Postprocessor (which now only handles NMS + decode).
 */

#include <gtest/gtest.h>
#include "edgeai/decision/decision_engine.h"

using namespace edgeai;

class DecisionEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // reject_threshold=0.5, review_threshold=0.3
        engine = std::make_unique<DecisionEngine>(0.5f, 0.3f);
    }
    std::unique_ptr<DecisionEngine> engine;
};

TEST_F(DecisionEngineTest, EmptyDetections_Pass) {
    std::vector<Detection> dets;
    auto verdict = engine->decide(dets);
    EXPECT_EQ(verdict, Verdict::Pass);
}

TEST_F(DecisionEngineTest, HighConfidence_Reject) {
    std::vector<Detection> dets;
    dets.push_back({DefectType::Dent, 0.9f, {}, 0});

    auto verdict = engine->decide(dets);
    EXPECT_EQ(verdict, Verdict::Reject);
}

TEST_F(DecisionEngineTest, ExactRejectThreshold_Reject) {
    std::vector<Detection> dets;
    dets.push_back({DefectType::Dent, 0.5f, {}, 0});

    auto verdict = engine->decide(dets);
    EXPECT_EQ(verdict, Verdict::Reject);  // >= reject_threshold
}

TEST_F(DecisionEngineTest, BorderlineConfidence_Review) {
    std::vector<Detection> dets;
    dets.push_back({DefectType::Dent, 0.4f, {}, 0});  // between 0.3 and 0.5

    auto verdict = engine->decide(dets);
    EXPECT_EQ(verdict, Verdict::Review);
}

TEST_F(DecisionEngineTest, ExactReviewThreshold_Review) {
    std::vector<Detection> dets;
    dets.push_back({DefectType::Dent, 0.3f, {}, 0});

    auto verdict = engine->decide(dets);
    EXPECT_EQ(verdict, Verdict::Review);  // >= review_threshold
}

TEST_F(DecisionEngineTest, BelowAllThresholds_Pass) {
    std::vector<Detection> dets;
    dets.push_back({DefectType::Dent, 0.1f, {}, 0});  // below review_threshold

    auto verdict = engine->decide(dets);
    EXPECT_EQ(verdict, Verdict::Pass);
}

TEST_F(DecisionEngineTest, MixedConfidences_RejectWins) {
    std::vector<Detection> dets;
    dets.push_back({DefectType::Dent, 0.4f, {}, 0});        // review
    dets.push_back({DefectType::WrongLabel, 0.8f, {}, 1});  // reject

    auto verdict = engine->decide(dets);
    EXPECT_EQ(verdict, Verdict::Reject);  // reject takes priority
}

TEST_F(DecisionEngineTest, MultipleReviewOnly_Review) {
    std::vector<Detection> dets;
    dets.push_back({DefectType::Dent, 0.35f, {}, 0});
    dets.push_back({DefectType::SealDefect, 0.42f, {}, 3});

    auto verdict = engine->decide(dets);
    EXPECT_EQ(verdict, Verdict::Review);
}

TEST_F(DecisionEngineTest, CustomThresholds) {
    // Strict thresholds for safety-critical domain
    auto strict = std::make_unique<DecisionEngine>(0.3f, 0.1f);

    std::vector<Detection> dets;
    dets.push_back({DefectType::Dent, 0.25f, {}, 0});

    auto verdict = strict->decide(dets);
    EXPECT_EQ(verdict, Verdict::Review);  // 0.25 >= 0.1 review, < 0.3 reject
}
