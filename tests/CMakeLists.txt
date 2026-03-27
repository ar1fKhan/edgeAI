cmake_minimum_required(VERSION 3.20)

# ═══════════════════════════════════════════════════════════════
#  Unit Tests — each links only the module under test
# ═══════════════════════════════════════════════════════════════

# Types (common module only)
add_executable(test_types test_types.cpp)
target_link_libraries(test_types PRIVATE edgeai_common gtest_main)
add_test(NAME TypesTest COMMAND test_types)

# Preprocessing (preprocessing module)
add_executable(test_preprocessing test_preprocessing.cpp)
target_link_libraries(test_preprocessing PRIVATE edgeai_preprocessing gtest_main)
add_test(NAME PreprocessingTest COMMAND test_preprocessing)

# Postprocessing (postprocessing module — NMS + decode only)
add_executable(test_postprocessing test_postprocessing.cpp)
target_link_libraries(test_postprocessing PRIVATE edgeai_postprocessing gtest_main)
add_test(NAME PostprocessingTest COMMAND test_postprocessing)

# Decision (decision module — business logic verdicts)
add_executable(test_decision test_decision.cpp)
target_link_libraries(test_decision PRIVATE edgeai_decision gtest_main)
add_test(NAME DecisionTest COMMAND test_decision)

# Database / Storage (storage module)
add_executable(test_database test_database.cpp)
target_link_libraries(test_database PRIVATE edgeai_storage gtest_main)
add_test(NAME DatabaseTest COMMAND test_database)

# Frame Queue (header-only template, needs common for types)
add_executable(test_frame_queue test_frame_queue.cpp)
target_link_libraries(test_frame_queue PRIVATE edgeai_common gtest_main)
add_test(NAME FrameQueueTest COMMAND test_frame_queue)

# ═══════════════════════════════════════════════════════════════
#  Integration Tests — full pipeline with GMock
#
#  Links pipeline + all concrete modules + GMock.
#  Mock classes in tests/mocks/ replace external dependencies.
# ═══════════════════════════════════════════════════════════════

add_executable(test_integration test_integration.cpp)
target_include_directories(test_integration PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_integration PRIVATE
    edgeai_pipeline
    edgeai_preprocessing
    edgeai_postprocessing
    edgeai_decision
    edgeai_common
    gmock_main
)
add_test(NAME IntegrationTest COMMAND test_integration)
