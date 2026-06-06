# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Configure (Debug includes -Wall -Wextra -Wpedantic -Wshadow)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug

# Build all targets
cmake --build build

# Run all tests
cd build && ctest --output-on-failure

# Run a single test binary
./build/tests/test_decision
./build/tests/test_postprocessing
./build/tests/test_integration   # full pipeline with GMock

# Run the main application
./build/bin/edge_inspector --model models/defect_detector.onnx --camera 0
./build/bin/edge_inspector --model models/defect_detector.onnx --video path/to/video.mp4 --loop

# Latency profiler
./build/bin/benchmark_inference

# Camera capture tool
./build/bin/capture_images
```

### Build Options

| CMake Flag | Default | Purpose |
|------------|---------|---------|
| `ENABLE_TESTS` | ON | Google Test unit + integration tests |
| `ENABLE_TENSORRT` | OFF | NVIDIA TensorRT execution provider |
| `ENABLE_OPENVINO` | OFF | Intel OpenVINO execution provider |
| `ENABLE_GPIO` | OFF | sysfs GPIO hardware I/O (`ENABLE_GPIO_HW` define) |
| `ENABLE_SANITIZER` | OFF | AddressSanitizer |

If ONNX Runtime headers/libs are not found, the build proceeds with a stub inference engine (`USE_ONNX_STUB`). Check `build/CMakeCache.txt` if inference behaves unexpectedly.

### Python (training + dashboard)

```bash
cd python && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train/train_model.py --data ../configs/dataset.yaml --epochs 100
python dashboard/app.py  # http://localhost:5000
```

## Architecture

The system is a **real-time visual defect detection pipeline** for a conveyor belt (current domain: paint cans). The C++ runtime is domain-agnostic — domain knowledge lives only in the trained `.onnx` model, the `DefectType` enum in [include/edgeai/common/types.h](include/edgeai/common/types.h), and `DecisionEngine` thresholds.

### Module Libraries (9 static libs)

Each library is compiled, linked, and tested independently. Concrete implementations are injected only at the composition root (`src/main.cpp`).

```
edgeai_common          ← types, logger, timer, config_loader (foundation)
    ↑
    ├── edgeai_camera            ICamera → CameraManager (OpenCV)
    ├── edgeai_preprocessing     Preprocessor (letterbox, normalize, HWC→CHW)
    ├── edgeai_inference         IInferenceEngine → OnnxEngine (tensor-in/out only)
    ├── edgeai_postprocessing    Postprocessor (NMS, decode → Detection structs)
    ├── edgeai_decision          IDecisionEngine → DecisionEngine (Pass/Reject/Review)
    ├── edgeai_storage           IDefectStore → DefectDatabase (SQLite WAL)
    └── edgeai_io                IRejectController → GpioController (sysfs)
            ↑
    edgeai_pipeline              DefectPipeline (thread orchestrator, interfaces only)
            ↑
    edge_inspector               composition root — wires all concrete types
```

### Key Interfaces (swap points)

Located in `include/edgeai/`:

- `camera/icamera.h` — `ICamera`
- `inference/iengine.h` — `IInferenceEngine`
- `decision/idecision_engine.h` — `IDecisionEngine`
- `db/idefect_store.h` — `IDefectStore`
- `io/ireject_controller.h` — `IRejectController`

To swap any module (e.g., ONNX → TensorRT), implement the corresponding interface and change the `make_unique<>` call in `src/main.cpp`. The pipeline library has zero knowledge of concrete implementations.

### Threading Model

Two threads with a bounded SPSC `FrameQueue`:

- **Capture thread**: calls `ICamera::capture()`, pushes to `FrameQueue`. Blocks on full queue (back-pressure).
- **Inference thread**: pops frames, runs preprocess → infer → postprocess → decide → store → reject in sequence.

`running_` is `std::atomic<bool>` for lock-free shutdown. Stats are guarded by `stats_mutex_`. No other shared mutable state exists between threads.

### Data Flow Types (`include/edgeai/common/types.h`)

- `Frame` — raw camera frame with `cv::Mat` + timestamp
- `Detection` — single defect: `DefectType`, confidence, normalized `BoundingBox`, class_id
- `InspectionResult` — per-frame result: `Verdict` (Pass/Reject/Review), detections, timing
- `PipelineStats` — running counters updated via `PipelineStats::update(InspectionResult)`

`BoundingBox` coordinates are **normalized 0–1**; use `to_pixel_rect(img_w, img_h)` to convert.

### Decision Logic

`DecisionEngine` uses two thresholds from `InferenceConfig`:
- `conf_threshold` (default 0.5) — at or above → `Reject`
- `review_threshold` (default 0.3) — between review and conf → `Review`
- Below review threshold → `Pass`

For domain-specific rules, implement `IDecisionEngine` and inject in `main.cpp`.

## Conventions

- **Logging**: use `LOG_INFO`, `LOG_WARN`, `LOG_ERROR`, `LOG_DEBUG` macros from `include/edgeai/common/logger.h` (`edgeai::Logger` singleton, thread-safe).
- **Config**: key-value pairs via `ConfigLoader`; use `safe_stoi` / `safe_stof` with range bounds — never raw `stoi`. Runtime config in `configs/inference.cfg`.
- **SQL**: always use prepared statements (`sqlite3_prepare_v2` / `sqlite3_bind_*`). DB rotation via `DefectDatabase::prune()` at `max_db_records`.
- **GPIO**: `GpioController` uses sysfs (legacy). If extending GPIO support, prefer `libgpiod`. Watch for detached thread lifetimes on `stop()`.
- **ONNX Engine**: `OnnxEngine` takes `vector<float>` input and returns `vector<float>` output. It has no OpenCV or image knowledge — preprocessing/postprocessing are separate modules.
- **Error handling at boundaries**: validate config and external inputs; trust internal module contracts.

## Testing

Unit tests link only their target module. Integration test (`test_integration.cpp`) uses GMock with mock headers from `tests/mocks/`. When adding a new module or interface, provide a corresponding mock in `tests/mocks/` following the existing pattern.

To add a new defect domain: update `DefectType` enum + `defect_type_to_string` / `string_to_defect_type` in `types.h`, update `class_id_to_type()` in `src/inference/postprocessing.cpp`, retrain the model, and swap the `.onnx` file.
