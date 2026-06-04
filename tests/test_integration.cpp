/**
 * @file test_integration.cpp
 * @brief Full integration tests for the EdgeAI pipeline using GMock.
 *
 * Tests the complete flow: Camera → Queue → Preprocess → Infer → Postprocess
 * → Decide → Store → Reject, with all external dependencies mocked.
 *
 * Test categories:
 *   1. Single-frame flow (process_frame — synchronous)
 *   2. Multi-threaded pipeline (start/stop — asynchronous with real threads)
 *   3. Edge cases (empty output, failures, nullables)
 *   4. Stats accumulation across multiple frames
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <thread>
#include <chrono>
#include <atomic>

#include <opencv2/core.hpp>

#include "edgeai/pipeline/defect_pipeline.h"
#include "edgeai/inference/preprocessing.h"
#include "edgeai/inference/postprocessing.h"
#include "edgeai/decision/decision_engine.h"

// ── Mocks ──────────────────────────────────────────────────────
#include "mocks/mock_camera.h"
#include "mocks/mock_engine.h"
#include "mocks/mock_decision_engine.h"
#include "mocks/mock_store.h"
#include "mocks/mock_reject_controller.h"

using namespace edgeai;
using namespace edgeai::testing;

using ::testing::_;
using ::testing::Return;
using ::testing::Invoke;
using ::testing::AtLeast;
using ::testing::NiceMock;
using ::testing::StrictMock;
using ::testing::HasSubstr;

// ═══════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════

namespace {

/// Create a default PipelineConfig suitable for testing
PipelineConfig make_test_config() {
    PipelineConfig config;
    config.inference.input_width    = 640;
    config.inference.input_height   = 640;
    config.inference.num_classes    = 5;
    config.inference.conf_threshold = 0.5f;
    config.inference.nms_threshold  = 0.45f;
    config.inference.review_threshold = 0.3f;
    config.queue_capacity          = 8;
    config.enable_display          = false;
    config.save_defect_images      = false;
    return config;
}

/// Create a dummy BGR image (solid color) for testing
cv::Mat make_test_image(int width = 640, int height = 480) {
    return cv::Mat(height, width, CV_8UC3, cv::Scalar(128, 128, 128));
}

/// Build a fake YOLO output tensor with 1 detection.
/// Format: [cx, cy, w, h, class0_conf, class1_conf, ..., classN_conf]
/// YOLOv8 has no objectness score: cols = 4 + num_classes.
std::vector<float> make_yolo_output(int num_classes, int class_id,
                                     float confidence,
                                     float cx = 320.0f, float cy = 320.0f,
                                     float w = 100.0f, float h = 100.0f) {
    int cols = 4 + num_classes;
    std::vector<float> output(cols, 0.0f);
    output[0] = cx;
    output[1] = cy;
    output[2] = w;
    output[3] = h;
    if (class_id >= 0 && class_id < num_classes) {
        output[4 + class_id] = confidence;
    }
    return output;
}

/// Build a fake YOLO output tensor with multiple detections
std::vector<float> make_multi_detection_output(
    int num_classes,
    const std::vector<std::pair<int, float>>& class_conf_pairs) {

    std::vector<float> output;
    int cols = 4 + num_classes;  // YOLOv8: no objectness score
    float offset = 0.0f;
    for (const auto& [class_id, conf] : class_conf_pairs) {
        std::vector<float> row(cols, 0.0f);
        row[0] = 100.0f + offset;   // cx — offset to avoid NMS suppression
        row[1] = 100.0f + offset;   // cy
        row[2] = 50.0f;             // w
        row[3] = 50.0f;             // h
        if (class_id >= 0 && class_id < num_classes) {
            row[4 + class_id] = conf;
        }
        output.insert(output.end(), row.begin(), row.end());
        offset += 200.0f;
    }
    return output;
}

/// Create a valid Frame for mock camera
Frame make_test_frame(uint64_t id = 1) {
    Frame f;
    f.id = id;
    f.image = make_test_image();
    f.captured_at = Clock::now();
    return f;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════
//  1. SINGLE-FRAME INTEGRATION TESTS (process_frame)
//
//  These test the synchronous path: image → result,
//  with real Preprocessor + Postprocessor but mocked Engine + Decision.
// ═══════════════════════════════════════════════════════════════

class SingleFrameIntegration : public ::testing::Test {
protected:
    PipelineConfig config_ = make_test_config();
};

// ── 1a. Clean frame, no defects → PASS ─────────────────────────

TEST_F(SingleFrameIntegration, CleanFrame_NoDefects_Pass) {
    auto engine   = std::make_unique<NiceMock<MockEngine>>();
    auto decision = std::make_unique<NiceMock<MockDecisionEngine>>();

    // Engine returns empty output (no detections)
    EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));
    EXPECT_CALL(*engine, infer(_)).WillOnce(Return(std::vector<float>{}));

    // Decision sees empty detections → PASS
    EXPECT_CALL(*decision, decide(_)).WillOnce(Return(Verdict::Pass));

    auto preprocessor  = std::make_unique<Preprocessor>(640, 640);
    auto postprocessor = std::make_unique<Postprocessor>(0.5f, 0.45f, 5);

    DefectPipeline pipeline(
        nullptr,
        std::move(engine),
        std::move(preprocessor),
        std::move(postprocessor),
        std::move(decision),
        nullptr,
        nullptr,
        config_
    );

    ASSERT_TRUE(pipeline.initialize());

    auto result = pipeline.process_frame(make_test_image());
    EXPECT_EQ(result.verdict, Verdict::Pass);
    EXPECT_TRUE(result.detections.empty());
    EXPECT_FALSE(result.has_defects());
}

// ── 1b. High confidence defect → REJECT ────────────────────────

TEST_F(SingleFrameIntegration, HighConfidenceDefect_Reject) {
    auto engine   = std::make_unique<NiceMock<MockEngine>>();
    auto decision = std::make_unique<NiceMock<MockDecisionEngine>>();

    auto fake_output = make_yolo_output(5, /*class_id=*/0, /*conf=*/0.95f);

    EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));
    EXPECT_CALL(*engine, infer(_)).WillOnce(Return(fake_output));
    EXPECT_CALL(*decision, decide(_)).WillOnce(Return(Verdict::Reject));

    DefectPipeline pipeline(
        nullptr,
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::move(decision),
        nullptr, nullptr,
        config_
    );

    ASSERT_TRUE(pipeline.initialize());
    auto result = pipeline.process_frame(make_test_image());

    EXPECT_EQ(result.verdict, Verdict::Reject);
    EXPECT_FALSE(result.detections.empty());
    EXPECT_TRUE(result.has_defects());
    EXPECT_GE(result.max_confidence(), 0.9f);
}

// ── 1c. Borderline confidence → REVIEW ─────────────────────────

TEST_F(SingleFrameIntegration, BorderlineConfidence_Review) {
    auto engine   = std::make_unique<NiceMock<MockEngine>>();
    auto decision = std::make_unique<NiceMock<MockDecisionEngine>>();

    // Confidence between review_threshold (0.3) and conf_threshold (0.5)
    // Postprocessor filters below conf_threshold, so use a confidence
    // slightly above that but below "high confidence" to trigger REVIEW
    auto fake_output = make_yolo_output(5, 0, 0.55f);

    EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));
    EXPECT_CALL(*engine, infer(_)).WillOnce(Return(fake_output));
    EXPECT_CALL(*decision, decide(_)).WillOnce(Return(Verdict::Review));

    DefectPipeline pipeline(
        nullptr,
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::move(decision),
        nullptr, nullptr,
        config_
    );

    ASSERT_TRUE(pipeline.initialize());
    auto result = pipeline.process_frame(make_test_image());

    EXPECT_EQ(result.verdict, Verdict::Review);
}

// ── 1d. Multiple defects of different classes ──────────────────

TEST_F(SingleFrameIntegration, MultipleDefects_DifferentClasses) {
    auto engine   = std::make_unique<NiceMock<MockEngine>>();
    auto decision = std::make_unique<NiceMock<MockDecisionEngine>>();

    auto fake_output = make_multi_detection_output(5, {
        {0, 0.9f},   // Dent
        {1, 0.85f},  // WrongLabel
        {3, 0.8f},   // SealDefect
    });

    EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));
    EXPECT_CALL(*engine, infer(_)).WillOnce(Return(fake_output));
    EXPECT_CALL(*decision, decide(_)).WillOnce(
        Invoke([](const std::vector<Detection>& dets) -> Verdict {
            return dets.size() >= 2 ? Verdict::Reject : Verdict::Pass;
        })
    );

    DefectPipeline pipeline(
        nullptr,
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::move(decision),
        nullptr, nullptr,
        config_
    );

    ASSERT_TRUE(pipeline.initialize());
    auto result = pipeline.process_frame(make_test_image());

    EXPECT_EQ(result.verdict, Verdict::Reject);
    EXPECT_GE(result.detections.size(), 3u);  // 3 detections, different classes
}

// ── 1e. All detections below threshold → empty → PASS ──────────

TEST_F(SingleFrameIntegration, AllBelowThreshold_NoDetections) {
    auto engine   = std::make_unique<NiceMock<MockEngine>>();
    auto decision = std::make_unique<NiceMock<MockDecisionEngine>>();

    // All confidences below conf_threshold (0.5)
    auto fake_output = make_multi_detection_output(5, {
        {0, 0.2f},
        {1, 0.1f},
        {2, 0.3f},
    });

    EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));
    EXPECT_CALL(*engine, infer(_)).WillOnce(Return(fake_output));
    EXPECT_CALL(*decision, decide(_)).WillOnce(Return(Verdict::Pass));

    DefectPipeline pipeline(
        nullptr,
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::move(decision),
        nullptr, nullptr,
        config_
    );

    ASSERT_TRUE(pipeline.initialize());
    auto result = pipeline.process_frame(make_test_image());

    EXPECT_EQ(result.verdict, Verdict::Pass);
    EXPECT_TRUE(result.detections.empty());
}

// ═══════════════════════════════════════════════════════════════
//  2. SINGLE-FRAME WITH REAL DECISION ENGINE
//
//  End-to-end with real Preprocessor + Postprocessor + DecisionEngine.
//  Only the inference engine is mocked.
// ═══════════════════════════════════════════════════════════════

class EndToEndSingleFrame : public ::testing::Test {
protected:
    PipelineConfig config_ = make_test_config();
};

TEST_F(EndToEndSingleFrame, RealDecision_EmptyOutput_Pass) {
    auto engine = std::make_unique<NiceMock<MockEngine>>();
    EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));
    EXPECT_CALL(*engine, infer(_)).WillOnce(Return(std::vector<float>{}));

    DefectPipeline pipeline(
        nullptr,
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::make_unique<DecisionEngine>(0.5f, 0.3f),
        nullptr, nullptr,
        config_
    );

    ASSERT_TRUE(pipeline.initialize());
    auto result = pipeline.process_frame(make_test_image());

    EXPECT_EQ(result.verdict, Verdict::Pass);
    EXPECT_TRUE(result.detections.empty());
}

TEST_F(EndToEndSingleFrame, RealDecision_HighConfDent_Reject) {
    auto engine = std::make_unique<NiceMock<MockEngine>>();
    auto fake_output = make_yolo_output(5, 0, 0.92f);  // Dent @ 92%

    EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));
    EXPECT_CALL(*engine, infer(_)).WillOnce(Return(fake_output));

    DefectPipeline pipeline(
        nullptr,
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::make_unique<DecisionEngine>(0.5f, 0.3f),
        nullptr, nullptr,
        config_
    );

    ASSERT_TRUE(pipeline.initialize());
    auto result = pipeline.process_frame(make_test_image());

    EXPECT_EQ(result.verdict, Verdict::Reject);
    ASSERT_EQ(result.detections.size(), 1u);
    EXPECT_EQ(result.detections[0].type, DefectType::Dent);
    EXPECT_FLOAT_EQ(result.detections[0].confidence, 0.92f);
}

TEST_F(EndToEndSingleFrame, RealDecision_BorderlineConf_Review) {
    auto engine = std::make_unique<NiceMock<MockEngine>>();
    // Confidence 0.35 — above review_threshold (0.3) but below reject (0.5)
    // BUT Postprocessor conf_threshold is 0.5, so this detection would be filtered!
    // We need a confidence that passes postprocessor (>=0.5) but triggers Review
    // in the DecisionEngine. With reject=0.5 and review=0.3, anything >=0.5 is Reject.
    //
    // For a Review, we need: DecisionEngine with reject_threshold=0.7, review=0.3
    // and confidence between 0.5 and 0.7 (passes postprocessor, hits Review range).

    auto fake_output = make_yolo_output(5, 2, 0.55f);  // MissingLabel @ 55%

    EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));
    EXPECT_CALL(*engine, infer(_)).WillOnce(Return(fake_output));

    // Set reject=0.7, review=0.3 so 0.55 falls in Review range
    DefectPipeline pipeline(
        nullptr,
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::make_unique<DecisionEngine>(/*reject=*/0.7f, /*review=*/0.3f),
        nullptr, nullptr,
        config_
    );

    ASSERT_TRUE(pipeline.initialize());
    auto result = pipeline.process_frame(make_test_image());

    EXPECT_EQ(result.verdict, Verdict::Review);
    ASSERT_EQ(result.detections.size(), 1u);
    EXPECT_EQ(result.detections[0].type, DefectType::MissingLabel);
}

// ═══════════════════════════════════════════════════════════════
//  3. MULTI-THREADED PIPELINE INTEGRATION TESTS
//
//  Tests start()/stop() with real threads, mocked components.
//  MockCamera feeds frames, MockEngine returns canned output,
//  MockStore verifies results are logged.
// ═══════════════════════════════════════════════════════════════

class PipelineThreadedIntegration : public ::testing::Test {
protected:
    PipelineConfig config_ = make_test_config();
};

// ── 3a. Pipeline processes frames from mock camera ─────────────

TEST_F(PipelineThreadedIntegration, ProcessesFramesFromCamera) {
    auto camera   = std::make_unique<NiceMock<MockCamera>>();
    auto engine   = std::make_unique<NiceMock<MockEngine>>();
    auto store    = std::make_unique<NiceMock<MockStore>>();
    auto decision = std::make_unique<NiceMock<MockDecisionEngine>>();

    // Camera produces frames, then signals stop
    std::atomic<int> frame_count{0};
    const int total_frames = 5;

    EXPECT_CALL(*camera, open()).WillOnce(Return(true));
    EXPECT_CALL(*camera, close()).Times(1);

    // Return valid frames, then block to avoid busy-loop after done
    ON_CALL(*camera, grab_frame()).WillByDefault(
        Invoke([&]() -> Frame {
            int count = frame_count.fetch_add(1);
            if (count < total_frames) {
                return make_test_frame(count + 1);
            }
            // After all frames delivered, return invalid to slow down capture thread
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return Frame{};  // invalid
        })
    );

    // Engine returns empty output (clean frames)
    EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));
    ON_CALL(*engine, infer(_)).WillByDefault(Return(std::vector<float>{}));

    // Decision always passes
    ON_CALL(*decision, decide(_)).WillByDefault(Return(Verdict::Pass));

    // Store should receive results
    EXPECT_CALL(*store, open()).WillOnce(Return(true));
    EXPECT_CALL(*store, insert_result(_)).Times(AtLeast(1)).WillRepeatedly(Return(true));

    DefectPipeline pipeline(
        std::move(camera),
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::move(decision),
        std::move(store),
        nullptr,
        config_
    );

    ASSERT_TRUE(pipeline.initialize());
    pipeline.start();
    EXPECT_TRUE(pipeline.is_running());

    // Wait for frames to be processed
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    pipeline.stop();
    EXPECT_FALSE(pipeline.is_running());

    auto stats = pipeline.stats();
    EXPECT_GE(stats.total_frames, 1u);
}

// ── 3b. Reject triggered on defective frames ──────────────────

TEST_F(PipelineThreadedIntegration, RejectTriggeredOnDefect) {
    auto camera   = std::make_unique<NiceMock<MockCamera>>();
    auto engine   = std::make_unique<NiceMock<MockEngine>>();
    auto store    = std::make_unique<NiceMock<MockStore>>();
    auto decision = std::make_unique<NiceMock<MockDecisionEngine>>();
    auto rejector = std::make_unique<NiceMock<MockRejectController>>();

    std::atomic<int> frame_count{0};
    const int total_frames = 3;

    EXPECT_CALL(*camera, open()).WillOnce(Return(true));
    EXPECT_CALL(*camera, close()).Times(1);
    ON_CALL(*camera, grab_frame()).WillByDefault(
        Invoke([&]() -> Frame {
            int count = frame_count.fetch_add(1);
            if (count < total_frames) {
                return make_test_frame(count + 1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return Frame{};
        })
    );

    // Engine returns defect output
    auto fake_output = make_yolo_output(5, 0, 0.92f);
    EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));
    ON_CALL(*engine, infer(_)).WillByDefault(Return(fake_output));

    // Decision always rejects
    ON_CALL(*decision, decide(_)).WillByDefault(Return(Verdict::Reject));

    // Store accepts results
    EXPECT_CALL(*store, open()).WillOnce(Return(true));
    ON_CALL(*store, insert_result(_)).WillByDefault(Return(true));

    // Reject controller should be triggered
    EXPECT_CALL(*rejector, initialize()).WillOnce(Return(true));
    EXPECT_CALL(*rejector, is_initialized()).WillRepeatedly(Return(true));
    EXPECT_CALL(*rejector, trigger_reject()).Times(AtLeast(1));

    DefectPipeline pipeline(
        std::move(camera),
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::move(decision),
        std::move(store),
        std::move(rejector),
        config_
    );

    ASSERT_TRUE(pipeline.initialize());
    pipeline.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    pipeline.stop();

    auto stats = pipeline.stats();
    EXPECT_GE(stats.defective_frames, 1u);
}

// ── 3c. Result callback invoked with correct data ──────────────

TEST_F(PipelineThreadedIntegration, ResultCallbackInvoked) {
    auto camera   = std::make_unique<NiceMock<MockCamera>>();
    auto engine   = std::make_unique<NiceMock<MockEngine>>();
    auto decision = std::make_unique<NiceMock<MockDecisionEngine>>();

    std::atomic<int> frame_count{0};
    EXPECT_CALL(*camera, open()).WillOnce(Return(true));
    EXPECT_CALL(*camera, close()).Times(1);
    ON_CALL(*camera, grab_frame()).WillByDefault(
        Invoke([&]() -> Frame {
            int count = frame_count.fetch_add(1);
            if (count < 2) return make_test_frame(count + 1);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return Frame{};
        })
    );

    EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));
    ON_CALL(*engine, infer(_)).WillByDefault(Return(std::vector<float>{}));
    ON_CALL(*decision, decide(_)).WillByDefault(Return(Verdict::Pass));

    DefectPipeline pipeline(
        std::move(camera),
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::move(decision),
        nullptr, nullptr,
        config_
    );

    // Register callback
    std::atomic<int> callback_count{0};
    std::vector<Verdict> verdicts;
    std::mutex verdicts_mutex;
    pipeline.on_result([&](const InspectionResult& result) {
        callback_count.fetch_add(1);
        std::lock_guard<std::mutex> lock(verdicts_mutex);
        verdicts.push_back(result.verdict);
    });

    ASSERT_TRUE(pipeline.initialize());
    pipeline.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    pipeline.stop();

    EXPECT_GE(callback_count.load(), 1);
    std::lock_guard<std::mutex> lock(verdicts_mutex);
    for (auto v : verdicts) {
        EXPECT_EQ(v, Verdict::Pass);
    }
}

// ── 3d. Stats accumulate correctly ─────────────────────────────

TEST_F(PipelineThreadedIntegration, StatsAccumulateCorrectly) {
    auto camera   = std::make_unique<NiceMock<MockCamera>>();
    auto engine   = std::make_unique<NiceMock<MockEngine>>();
    auto decision = std::make_unique<NiceMock<MockDecisionEngine>>();

    std::atomic<int> frame_count{0};
    const int total_frames = 6;

    EXPECT_CALL(*camera, open()).WillOnce(Return(true));
    EXPECT_CALL(*camera, close()).Times(1);
    EXPECT_CALL(*camera, is_open()).WillRepeatedly(Return(true));
    ON_CALL(*camera, grab_frame()).WillByDefault(
        Invoke([&]() -> Frame {
            int count = frame_count.fetch_add(1);
            if (count < total_frames) return make_test_frame(count + 1);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return Frame{};
        })
    );

    EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));
    ON_CALL(*engine, infer(_)).WillByDefault(Return(std::vector<float>{}));

    // Alternate verdicts: Pass, Reject, Review, Pass, Reject, Review
    std::atomic<int> decide_count{0};
    ON_CALL(*decision, decide(_)).WillByDefault(
        Invoke([&](const std::vector<Detection>&) -> Verdict {
            int n = decide_count.fetch_add(1) % 3;
            if (n == 0) return Verdict::Pass;
            if (n == 1) return Verdict::Reject;
            return Verdict::Review;
        })
    );

    DefectPipeline pipeline(
        std::move(camera),
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::move(decision),
        nullptr, nullptr,
        config_
    );

    ASSERT_TRUE(pipeline.initialize());
    pipeline.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    pipeline.stop();

    auto stats = pipeline.stats();
    EXPECT_GE(stats.total_frames, 3u);  // at least some processed
    // Stats should have a mix of verdicts
    EXPECT_GE(stats.passed_frames + stats.defective_frames + stats.review_frames,
              stats.total_frames);
    EXPECT_GT(stats.avg_total_ms, 0.0);
}

// ═══════════════════════════════════════════════════════════════
//  4. EDGE CASES & FAILURE HANDLING
// ═══════════════════════════════════════════════════════════════

class PipelineEdgeCases : public ::testing::Test {
protected:
    PipelineConfig config_ = make_test_config();
};

// ── 4a. Engine fails to load → initialize returns false ────────

TEST_F(PipelineEdgeCases, EngineLoadFailure_InitReturnsFalse) {
    auto engine   = std::make_unique<NiceMock<MockEngine>>();
    auto decision = std::make_unique<NiceMock<MockDecisionEngine>>();

    EXPECT_CALL(*engine, load_model()).WillOnce(Return(false));

    DefectPipeline pipeline(
        nullptr,
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::move(decision),
        nullptr, nullptr,
        config_
    );

    EXPECT_FALSE(pipeline.initialize());
}

// ── 4b. Camera fails to open → initialize returns false ────────

TEST_F(PipelineEdgeCases, CameraOpenFailure_InitReturnsFalse) {
    auto camera   = std::make_unique<NiceMock<MockCamera>>();
    auto engine   = std::make_unique<NiceMock<MockEngine>>();
    auto decision = std::make_unique<NiceMock<MockDecisionEngine>>();

    EXPECT_CALL(*camera, open()).WillOnce(Return(false));

    DefectPipeline pipeline(
        std::move(camera),
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::move(decision),
        nullptr, nullptr,
        config_
    );

    EXPECT_FALSE(pipeline.initialize());
}

// ── 4c. Store fails to open → graceful fallback (store disabled) ───

TEST_F(PipelineEdgeCases, StoreOpenFailure_GracefulFallback) {
    auto engine   = std::make_unique<NiceMock<MockEngine>>();
    auto decision = std::make_unique<NiceMock<MockDecisionEngine>>();

    // Use a raw pointer to set up expectations BEFORE moving into pipeline
    auto store_ptr = new NiceMock<MockStore>();
    auto store     = std::unique_ptr<IDefectStore>(store_ptr);

    EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));
    EXPECT_CALL(*store_ptr, open()).WillOnce(Return(false));
    // NOTE: After store_.open() returns false, the pipeline resets store_ (destroys it).
    // We cannot check insert_result expectations after destruction.
    // The test verifies that pipeline.initialize() still succeeds.

    DefectPipeline pipeline(
        nullptr,
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::move(decision),
        std::move(store),
        nullptr,
        config_
    );

    // Should still succeed — store failure is non-fatal
    EXPECT_TRUE(pipeline.initialize());
}

// ── 4d. Rejector fails to init → graceful fallback ─────────────

TEST_F(PipelineEdgeCases, RejectorInitFailure_GracefulFallback) {
    auto engine   = std::make_unique<NiceMock<MockEngine>>();
    auto decision = std::make_unique<NiceMock<MockDecisionEngine>>();

    auto rejector_ptr = new NiceMock<MockRejectController>();
    auto rejector     = std::unique_ptr<IRejectController>(rejector_ptr);

    EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));
    EXPECT_CALL(*rejector_ptr, initialize()).WillOnce(Return(false));
    // After rejector init fails, pipeline resets rejector_ (destroys it).

    DefectPipeline pipeline(
        nullptr,
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::move(decision),
        nullptr,
        std::move(rejector),
        config_
    );

    EXPECT_TRUE(pipeline.initialize());
}

// ── 4e. Null camera + null store + null rejector (minimal pipeline) ─

TEST_F(PipelineEdgeCases, MinimalPipeline_AllNullables) {
    auto engine   = std::make_unique<NiceMock<MockEngine>>();
    auto decision = std::make_unique<NiceMock<MockDecisionEngine>>();

    EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));
    EXPECT_CALL(*engine, infer(_)).WillOnce(Return(std::vector<float>{}));
    EXPECT_CALL(*decision, decide(_)).WillOnce(Return(Verdict::Pass));

    DefectPipeline pipeline(
        nullptr, std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::move(decision),
        nullptr, nullptr,
        config_
    );

    ASSERT_TRUE(pipeline.initialize());
    auto result = pipeline.process_frame(make_test_image());
    EXPECT_EQ(result.verdict, Verdict::Pass);
}

// ── 4f. Pipeline start/stop lifecycle — double start is no-op ──

TEST_F(PipelineEdgeCases, DoubleStartIsNoOp) {
    auto camera   = std::make_unique<NiceMock<MockCamera>>();
    auto engine   = std::make_unique<NiceMock<MockEngine>>();
    auto decision = std::make_unique<NiceMock<MockDecisionEngine>>();

    EXPECT_CALL(*camera, open()).WillOnce(Return(true));
    EXPECT_CALL(*camera, close()).Times(1);
    ON_CALL(*camera, grab_frame()).WillByDefault(
        Invoke([]() -> Frame {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return Frame{};
        })
    );
    EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));
    ON_CALL(*engine, infer(_)).WillByDefault(Return(std::vector<float>{}));
    ON_CALL(*decision, decide(_)).WillByDefault(Return(Verdict::Pass));

    DefectPipeline pipeline(
        std::move(camera),
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::move(decision),
        nullptr, nullptr,
        config_
    );

    ASSERT_TRUE(pipeline.initialize());
    pipeline.start();
    pipeline.start();  // second start should be no-op
    EXPECT_TRUE(pipeline.is_running());
    pipeline.stop();
    EXPECT_FALSE(pipeline.is_running());
}

// ── 4g. Pipeline stop without start is safe ────────────────────

TEST_F(PipelineEdgeCases, StopWithoutStartIsSafe) {
    auto engine   = std::make_unique<NiceMock<MockEngine>>();
    auto decision = std::make_unique<NiceMock<MockDecisionEngine>>();

    EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));

    DefectPipeline pipeline(
        nullptr,
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::move(decision),
        nullptr, nullptr,
        config_
    );

    ASSERT_TRUE(pipeline.initialize());
    pipeline.stop();  // should not crash
    EXPECT_FALSE(pipeline.is_running());
}

// ═══════════════════════════════════════════════════════════════
//  5. STRICT MOCK TESTS — verify exact call sequences
// ═══════════════════════════════════════════════════════════════

class StrictIntegration : public ::testing::Test {
protected:
    PipelineConfig config_ = make_test_config();
};

// ── 5a. Verify exact call sequence for single frame ────────────

TEST_F(StrictIntegration, ExactCallSequence_SingleFrame) {
    auto engine   = std::make_unique<StrictMock<MockEngine>>();
    auto decision = std::make_unique<StrictMock<MockDecisionEngine>>();

    // Exact sequence: load_model → infer → decide
    {
        ::testing::InSequence seq;
        EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));
        EXPECT_CALL(*engine, infer(_)).WillOnce(Return(std::vector<float>{}));
        EXPECT_CALL(*decision, decide(_)).WillOnce(Return(Verdict::Pass));
    }

    DefectPipeline pipeline(
        nullptr,
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::move(decision),
        nullptr, nullptr,
        config_
    );

    ASSERT_TRUE(pipeline.initialize());
    auto result = pipeline.process_frame(make_test_image());
    EXPECT_EQ(result.verdict, Verdict::Pass);
}

// ── 5b. Store receives correct verdict in logged result ────────

TEST_F(StrictIntegration, StoreReceivesCorrectVerdict) {
    auto engine   = std::make_unique<NiceMock<MockEngine>>();
    auto store    = std::make_unique<NiceMock<MockStore>>();
    auto decision = std::make_unique<NiceMock<MockDecisionEngine>>();

    auto fake_output = make_yolo_output(5, 0, 0.9f);

    EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));
    EXPECT_CALL(*engine, infer(_)).WillOnce(Return(fake_output));
    EXPECT_CALL(*decision, decide(_)).WillOnce(Return(Verdict::Reject));
    EXPECT_CALL(*store, open()).WillOnce(Return(true));

    // Verify the stored result has the correct verdict
    EXPECT_CALL(*store, insert_result(_)).WillOnce(
        Invoke([](const InspectionResult& result) -> bool {
            EXPECT_EQ(result.verdict, Verdict::Reject);
            EXPECT_FALSE(result.detections.empty());
            return true;
        })
    );

    // Need to use process_frame path — note: process_frame doesn't call
    // handle_result (that's only in the threaded path). So we test via
    // the threaded pipeline instead.
    // Let's use the threaded path with 1 frame.
    auto camera = std::make_unique<NiceMock<MockCamera>>();
    std::atomic<int> frame_count{0};
    EXPECT_CALL(*camera, open()).WillOnce(Return(true));
    EXPECT_CALL(*camera, close()).Times(1);
    EXPECT_CALL(*camera, is_open()).WillRepeatedly(Return(true));
    ON_CALL(*camera, grab_frame()).WillByDefault(
        Invoke([&]() -> Frame {
            int count = frame_count.fetch_add(1);
            if (count < 1) return make_test_frame(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return Frame{};
        })
    );

    DefectPipeline pipeline(
        std::move(camera),
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::move(decision),
        std::move(store),
        nullptr,
        config_
    );

    ASSERT_TRUE(pipeline.initialize());
    pipeline.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    pipeline.stop();
}

// ═══════════════════════════════════════════════════════════════
//  6. CROSS-MODULE VERIFICATION
//
//  Verify the data integrity across module boundaries.
// ═══════════════════════════════════════════════════════════════

class CrossModuleIntegration : public ::testing::Test {
protected:
    PipelineConfig config_ = make_test_config();
};

// ── 6a. Preprocessor output dimensions match engine expectations ──

TEST_F(CrossModuleIntegration, PreprocessorTensorDimensionsCorrect) {
    Preprocessor preprocessor(640, 640);
    auto image = make_test_image(800, 600);

    auto prep = preprocessor.process(image);

    // CHW tensor should be 3 * 640 * 640
    EXPECT_EQ(prep.tensor.size(), 3u * 640u * 640u);
    EXPECT_EQ(prep.orig_w, 800);
    EXPECT_EQ(prep.orig_h, 600);
    EXPECT_GT(prep.scale_x, 0.0f);
    EXPECT_GT(prep.scale_y, 0.0f);
}

// ── 6b. Postprocessor + DecisionEngine contract ────────────────

TEST_F(CrossModuleIntegration, PostprocessorOutputFeedsDecisionEngine) {
    Postprocessor postprocessor(0.5f, 0.45f, 5);
    DecisionEngine decision(0.5f, 0.3f);

    // Simulate engine output with high-confidence dent
    auto raw_output = make_yolo_output(5, 0, 0.88f);
    int cols = 5 + 5;  // 5 + num_classes, matching pipeline formula
    int rows = 1;

    auto detections = postprocessor.process(
        raw_output, rows, cols,
        1.0f, 1.0f, 0, 0, 640, 640
    );
    auto nms_result = postprocessor.apply_nms(detections);

    // Feed to decision engine
    auto verdict = decision.decide(nms_result);
    EXPECT_EQ(verdict, Verdict::Reject);
}

// ── 6c. Full chain: Preprocess → (mock) Infer → Postprocess → Decide ──

TEST_F(CrossModuleIntegration, FullDataChain) {
    Preprocessor preprocessor(640, 640);
    Postprocessor postprocessor(0.5f, 0.45f, 5);
    DecisionEngine decision(0.5f, 0.3f);

    // Step 1: Preprocess
    auto image = make_test_image();
    auto prep = preprocessor.process(image);
    EXPECT_FALSE(prep.tensor.empty());

    // Step 2: Simulate inference (engine returns canned output)
    auto raw_output = make_multi_detection_output(5, {{0, 0.85f}, {2, 0.75f}});

    // Step 3: Postprocess
    int cols = 4 + 5;  // 4 + num_classes (YOLOv8: no objectness score)
    int rows = static_cast<int>(raw_output.size()) / cols;
    auto detections = postprocessor.process(
        raw_output, rows, cols,
        prep.scale_x, prep.scale_y,
        prep.pad_x, prep.pad_y,
        prep.orig_w, prep.orig_h
    );
    auto nms_result = postprocessor.apply_nms(detections);

    EXPECT_EQ(nms_result.size(), 2u);

    // Step 4: Decision
    auto verdict = decision.decide(nms_result);
    EXPECT_EQ(verdict, Verdict::Reject);
}

// ── 6d. Process_frame latency is measured ──────────────────────

TEST_F(CrossModuleIntegration, LatencyIsMeasured) {
    auto engine   = std::make_unique<NiceMock<MockEngine>>();
    auto decision = std::make_unique<NiceMock<MockDecisionEngine>>();

    EXPECT_CALL(*engine, load_model()).WillOnce(Return(true));
    EXPECT_CALL(*engine, infer(_)).WillOnce(
        Invoke([](const std::vector<float>&) -> std::vector<float> {
            // Simulate 5ms inference
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            return {};
        })
    );
    EXPECT_CALL(*decision, decide(_)).WillOnce(Return(Verdict::Pass));

    DefectPipeline pipeline(
        nullptr,
        std::move(engine),
        std::make_unique<Preprocessor>(640, 640),
        std::make_unique<Postprocessor>(0.5f, 0.45f, 5),
        std::move(decision),
        nullptr, nullptr,
        config_
    );

    ASSERT_TRUE(pipeline.initialize());
    auto result = pipeline.process_frame(make_test_image());

    // Verify timing was captured
    auto inf_ms = std::chrono::duration<double, std::milli>(result.inference_time).count();
    auto tot_ms = std::chrono::duration<double, std::milli>(result.total_time).count();
    EXPECT_GE(inf_ms, 4.0);  // at least ~5ms simulated inference
    EXPECT_GE(tot_ms, inf_ms);  // total >= inference
}
