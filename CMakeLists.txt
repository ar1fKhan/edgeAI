cmake_minimum_required(VERSION 3.20)
project(EdgeAI_DefectDetector 
    VERSION 1.0.0 
    LANGUAGES CXX
    DESCRIPTION "Real-time Edge AI Paint Can Defect Detection"
)

# ── C++17 Standard ──────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Build Configuration ────────────────────────────────────────
option(ENABLE_TENSORRT  "Enable TensorRT backend (NVIDIA only)" OFF)
option(ENABLE_OPENVINO  "Enable OpenVINO backend (Intel only)"  OFF)
option(ENABLE_GPIO      "Enable GPIO/PLC hardware I/O"          OFF)
option(ENABLE_TESTS     "Build unit tests"                      ON)
option(ENABLE_SANITIZER "Enable AddressSanitizer"               OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── Compiler Flags ──────────────────────────────────────────────
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic -Wshadow")

if(ENABLE_SANITIZER)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

# ── Output Directories ─────────────────────────────────────────
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ── Find Dependencies ──────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui dnn)
find_package(SQLite3 REQUIRED)
find_package(Threads REQUIRED)

# ONNX Runtime — search standard paths
find_library(ONNXRUNTIME_LIB onnxruntime 
    PATHS /usr/lib /usr/local/lib /opt/onnxruntime/lib
)
find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
    PATHS /usr/include /usr/local/include /opt/onnxruntime/include
    PATH_SUFFIXES onnxruntime
)

if(NOT ONNXRUNTIME_LIB OR NOT ONNXRUNTIME_INCLUDE)
    message(WARNING "ONNX Runtime not found — building with stub inference engine")
    set(USE_ONNX_STUB ON)
else()
    set(USE_ONNX_STUB OFF)
    message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIB}")
endif()

# ═══════════════════════════════════════════════════════════════
#  Module Libraries — each layer is independently compiled,
#  linked, tested, and swappable.
#
#  Dependency Graph:
#
#    edgeai_common          (types, logger, timer, config)
#        ↑
#    ┌───┼───────┬──────────┬───────────────┬──────────┬──────────┐
#    │   │       │          │               │          │          │
#  camera│  preproc  inference  postprocessing  decision  storage  io
#    │   │       │          │               │          │          │
#    └───┼───────┴──────────┴───────────────┴──────────┴──────────┘
#        ↑
#    edgeai_pipeline  (orchestrator — depends on interfaces only)
#        ↑
#    edge_inspector   (composition root — wires concrete types)
#
# ═══════════════════════════════════════════════════════════════

# ── 1. Common Module ───────────────────────────────────────────
#    Leaf module: types, logger, timer, config_loader
#    Everything depends on this.
add_library(edgeai_common STATIC
    src/common/logger.cpp
    src/common/timer.cpp
    src/common/config_loader.cpp
)
target_include_directories(edgeai_common PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(edgeai_common PUBLIC
    ${OpenCV_LIBS}
    Threads::Threads
)

# ── 2. Camera Module ──────────────────────────────────────────
#    ICamera interface + CameraManager implementation
#    Swap: replace CameraManager with SimulatedCamera, GigECamera, etc.
add_library(edgeai_camera STATIC
    src/camera/camera_manager.cpp
)
target_link_libraries(edgeai_camera PUBLIC edgeai_common)

# ── 3. Preprocessing Module ───────────────────────────────────
#    Letterbox resize, BGR→RGB, normalize, HWC→CHW
add_library(edgeai_preprocessing STATIC
    src/inference/preprocessing.cpp
)
target_link_libraries(edgeai_preprocessing PUBLIC edgeai_common)

# ── 4. Inference Module ───────────────────────────────────────
#    IInferenceEngine interface + OnnxEngine implementation
#    Pure tensor-in, tensor-out — no image or decision knowledge.
#    Swap: replace OnnxEngine with TensorRTEngine, OpenVINOEngine, etc.
add_library(edgeai_inference STATIC
    src/inference/onnx_engine.cpp
)
target_link_libraries(edgeai_inference PUBLIC
    edgeai_common
)
if(NOT USE_ONNX_STUB)
    target_include_directories(edgeai_inference PUBLIC ${ONNXRUNTIME_INCLUDE})
    target_link_libraries(edgeai_inference PUBLIC ${ONNXRUNTIME_LIB})
    target_compile_definitions(edgeai_inference PUBLIC HAS_ONNXRUNTIME)
endif()

# ── 5. Postprocessing Module ──────────────────────────────────
#    NMS, confidence filtering, raw model output → Detection structs
add_library(edgeai_postprocessing STATIC
    src/inference/postprocessing.cpp
)
target_link_libraries(edgeai_postprocessing PUBLIC edgeai_common)

# ── 6. Decision Module ────────────────────────────────────────
#    IDecisionEngine interface + threshold-based DecisionEngine
#    Business logic: given detections, determine Pass/Reject/Review
#    Swap: subclass IDecisionEngine for domain-specific rules.
add_library(edgeai_decision STATIC
    src/decision/decision_engine.cpp
)
target_link_libraries(edgeai_decision PUBLIC edgeai_common)

# ── 7. Storage Module ─────────────────────────────────────────
#    IDefectStore interface + DefectDatabase (SQLite) implementation
#    Swap: replace DefectDatabase with PostgresStore, InfluxStore, etc.
add_library(edgeai_storage STATIC
    src/db/defect_database.cpp
)
target_link_libraries(edgeai_storage PUBLIC
    edgeai_common
    SQLite::SQLite3
)

# ── 8. IO Module ──────────────────────────────────────────────
#    IRejectController interface + GpioController implementation
#    Swap: replace GpioController with PlcController, ModbusController, etc.
add_library(edgeai_io STATIC
    src/io/gpio_controller.cpp
)
target_link_libraries(edgeai_io PUBLIC edgeai_common)
if(ENABLE_GPIO)
    target_compile_definitions(edgeai_io PUBLIC ENABLE_GPIO_HW)
endif()

# ── 9. Pipeline Module (orchestrator) ─────────────────────────
#    Depends on INTERFACES (ICamera, IInferenceEngine, IPreprocessor,
#    IPostprocessor, IDecisionEngine, IDefectStore, IRejectController)
#    not on concrete implementations.
#    Concrete types are linked only at the application level.
add_library(edgeai_pipeline STATIC
    src/pipeline/defect_pipeline.cpp
    src/pipeline/frame_queue.cpp
)
target_link_libraries(edgeai_pipeline PUBLIC
    edgeai_common
)

# ── Main Executable (composition root) ─────────────────────────
#    This is where concrete implementations are linked.
#    The pipeline library only knows about interfaces.
add_executable(edge_inspector src/main.cpp)
target_link_libraries(edge_inspector PRIVATE
    edgeai_pipeline
    edgeai_camera
    edgeai_inference
    edgeai_preprocessing
    edgeai_postprocessing
    edgeai_decision
    edgeai_storage
    edgeai_io
)

# ── Benchmark Tool ─────────────────────────────────────────────
add_executable(benchmark_inference src/tools/benchmark.cpp)
target_link_libraries(benchmark_inference PRIVATE
    edgeai_inference
    edgeai_preprocessing
    edgeai_common
)

# ── Camera Capture Tool ────────────────────────────────────────
add_executable(capture_images src/tools/capture_images.cpp)
target_link_libraries(capture_images PRIVATE
    edgeai_camera
    edgeai_common
)

# ── Tests ───────────────────────────────────────────────────────
if(ENABLE_TESTS)
    enable_testing()
    # Build GTest + GMock from source (Ubuntu ships source in /usr/src/googletest)
    # This provides both GTest::gtest_main and GTest::gmock targets.
    if(EXISTS /usr/src/googletest/CMakeLists.txt)
        add_subdirectory(/usr/src/googletest ${CMAKE_BINARY_DIR}/googletest EXCLUDE_FROM_ALL)
        add_subdirectory(tests)
    else()
        find_package(GTest QUIET)
        if(GTest_FOUND)
            add_subdirectory(tests)
        else()
            message(STATUS "GTest not found — skipping tests")
        endif()
    endif()
endif()

# ── Install ─────────────────────────────────────────────────────
install(TARGETS edge_inspector DESTINATION bin)
install(DIRECTORY models/ DESTINATION share/edgeai/models)
install(DIRECTORY configs/ DESTINATION etc/edgeai)
