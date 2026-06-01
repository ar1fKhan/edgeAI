# EdgeAI Defect Detection System

> **Domain-agnostic** real-time edge AI quality control platform.  
> Modular architecture — swap camera, inference backend, decision logic, storage, or I/O without touching the pipeline.  
> **Zero cloud dependency** — all inference runs locally on edge hardware.

---

## Overview

A production-grade visual inspection system that detects defects on items moving along a conveyor belt using computer vision and deep learning at the edge. Designed for sub-50ms latency per frame, enabling real-time reject decisions at production line speeds.

The C++ runtime is **domain-agnostic** — the same pipeline handles paint cans, potatoes, PCBs, or pharmaceuticals. Domain knowledge lives in three places:

1. **Trained model** (`.onnx` file)
2. **`DefectType` enum** in `types.h` (~10 lines)
3. **`IDecisionEngine` implementation** (threshold tuning per domain)

> **Honest caveat:** while the pipeline code is domain-agnostic, a real domain pivot
> requires hardware tuning (lighting, camera positioning, trigger timing, reject
> mechanism) that is domain-specific and cannot be abstracted in software.

### Current Domain: Paint Can Inspection

| Class | Description |
|-------|-------------|
| `dent` | Physical dents or deformations on the can body |
| `wrong_label` | Incorrect label applied (wrong product/SKU) |
| `missing_label` | Label partially or fully missing |
| `seal_defect` | Lid/seal not properly applied |
| `color_mismatch` | Paint color doesn't match expected specification |

---

## Modular Architecture

### Layer Diagram

```
┌──────────────────────────────┐
│     Application (main.cpp)   │  ← Composition Root: wires concrete types
│     "edge_inspector"         │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│     Pipeline Layer           │  ← edgeai_pipeline (interfaces only)
│     DefectPipeline           │     Knows ICamera, IInferenceEngine,
│     FrameQueue               │     IDecisionEngine, IDefectStore,
│                              │     IRejectController
└──────────────┬───────────────┘
               │
   ┌───────┬───┼──────┬──────────┬──────────┬──────────┐
   │       │   │      │          │          │          │
┌──▼──┐ ┌──▼──┐ ┌──▼──┐ ┌───▼────┐ ┌───▼────┐ ┌──▼───┐ ┌──▼──┐
│Cam- │ │Pre- │ │Inf- │ │Post-   │ │Decis-  │ │Stor- │ │ I/O │
│era  │ │proc │ │er-  │ │process │ │ion     │ │age   │ │     │
│ Lib │ │ Lib │ │ence │ │  Lib   │ │  Lib   │ │ Lib  │ │ Lib │
└──┬──┘ └──┬──┘ └──┬──┘ └───┬────┘ └───┬────┘ └──┬───┘ └──┬──┘
   │       │      │         │          │          │        │
 ICamera   │  IInference   │     IDecision    IDefect  IReject
   │       │   Engine      │      Engine       Store   Controller
   │       └──────┼─────── │──────────┘          │        │
   │              │        │                     │        │
┌──▼──────────────▼────────▼─────────────────────▼────────▼────┐
│                       edgeai_common                          │
│          types.h  logger.h  timer.h  config_loader.h         │
└──────────────────────────────────────────────────────────────┘
               │
┌──────────────▼───────────────┐
│     Vendor SDK / Hardware    │
│   (OpenCV, ONNX RT, SQLite)  │
└──────────────────────────────┘
```

### Module Libraries

| Library | Interface | Default Implementation | Responsibility |
|---------|-----------|----------------------|----------------|
| `libedgeai_common.a` | — | types, logger, timer, config | Foundation layer |
| `libedgeai_camera.a` | `ICamera` | `CameraManager` (OpenCV) | Frame acquisition |
| `libedgeai_preprocessing.a` | — | `Preprocessor` | Letterbox, normalize, HWC→CHW |
| `libedgeai_inference.a` | `IInferenceEngine` | `OnnxEngine` | Tensor-in, tensor-out model inference |
| `libedgeai_postprocessing.a` | — | `Postprocessor` | NMS, confidence filtering, decode output |
| `libedgeai_decision.a` | `IDecisionEngine` | `DecisionEngine` | Business logic: Pass / Reject / Review |
| `libedgeai_storage.a` | `IDefectStore` | `DefectDatabase` (SQLite) | Defect logging |
| `libedgeai_io.a` | `IRejectController` | `GpioController` (sysfs) | Reject actuation |
| `libedgeai_pipeline.a` | — | `DefectPipeline` | Thread orchestrator |

### Design Principles

| Principle | Enforcement |
|-----------|-------------|
| **Inference engine has no image knowledge** | `OnnxEngine` accepts `vector<float>`, returns `vector<float>` — no OpenCV dependency |
| **Postprocessor has no business logic** | `Postprocessor` decodes model output → `Detection` structs. Verdict decision is in `DecisionEngine`. |
| **Pipeline is pure infrastructure** | `DefectPipeline` only orchestrates threads and data flow. Business logic delegated to `IDecisionEngine`. |
| **Common module stays thin** | Only types, logger, timer, config. No domain logic. |
| **Dependency injection everywhere** | All modules receive interfaces via constructor. Composition root (`main.cpp`) wires concrete types. |

### Dependency Graph

```
edgeai_common          ← foundation (types, logger, timer, config)
    ↑
    ├── edgeai_camera            (ICamera → CameraManager)
    ├── edgeai_preprocessing     (Preprocessor)
    ├── edgeai_inference         (IInferenceEngine → OnnxEngine)
    ├── edgeai_postprocessing    (Postprocessor — NMS + decode)
    ├── edgeai_decision          (IDecisionEngine → DecisionEngine)
    ├── edgeai_storage           (IDefectStore → DefectDatabase)
    └── edgeai_io                (IRejectController → GpioController)
            ↑
    edgeai_pipeline              (orchestrator — interfaces only)
            ↑
    edge_inspector               (composition root — wires concrete types)
```

### Why This Architecture Is Powerful

| Scenario | What Changes | What Stays |
|----------|-------------|------------|
| **Swap ONNX → TensorRT** | Only `edgeai_inference` implementation | All other 8 modules |
| **Replace real camera with simulator** | Only `edgeai_camera` implementation | All other 8 modules |
| **Unit test pipeline with mocks** | Inject mock `ICamera` / `IInferenceEngine` / `IDecisionEngine` | Zero production code changes |
| **Scale to multi-camera** | Multiple `ICamera` instances | Pipeline logic unchanged |
| **Switch SQLite → PostgreSQL** | Only `edgeai_storage` implementation | All other 8 modules |
| **Switch GPIO → PLC/Modbus** | Only `edgeai_io` implementation | All other 8 modules |
| **Change domain (cans → potatoes)** | Retrain model + update `DefectType` enum + tune `DecisionEngine` thresholds | All C++ runtime unchanged |
| **Different acceptance criteria per customer** | Implement custom `IDecisionEngine` | Pipeline, inference, postprocessing unchanged |

### Swapping a Module

All wiring happens in `main.cpp` (the composition root). To swap any module:

```cpp
// === Swap inference backend ===
// auto engine = std::make_unique<OnnxEngine>(config.inference);
auto engine = std::make_unique<TensorRTEngine>(config.inference);

// === Swap camera source ===
// auto camera = std::make_unique<CameraManager>(config.camera);
auto camera = std::make_unique<SimulatedCamera>("test_video.mp4");

// === Swap decision logic (e.g. stricter for pharma) ===
// auto decision = std::make_unique<DecisionEngine>(reject_thresh, review_thresh);
auto decision = std::make_unique<PharmaDecisionEngine>(pharma_rules);

// === Swap storage backend ===
// auto store = std::make_unique<DefectDatabase>(config.database_path);
auto store = std::make_unique<PostgresStore>(config.pg_connection_string);

// === Inject into pipeline (unchanged) ===
DefectPipeline pipeline(
    std::move(camera),
    std::move(engine),
    std::move(preprocessor),
    std::move(postprocessor),
    std::move(decision),
    std::move(store),
    std::move(rejector),
    config
);
```

---

## Threading Model

The pipeline uses a **4-thread** architecture with explicit ownership semantics:

```
Thread 1 (capture_thread_)
──────────────────────────
┌──────────────┐
│ ICamera      │──── FrameQueue (bounded SPSC) ──┐
└──────────────┘                                 │
                                                 ▼
Thread 2 (inference_thread_)              Thread 3 (display_thread_)
──────────────────────────────            ──────────────────────────
┌──────────────────────────┐              ┌──────────────────────┐
│ Preprocessor             │              │ Waits on display_cv_ │
│ IInferenceEngine         │─── display ─▶│ cv::imshow / waitKey │
│ Postprocessor (NMS)      │    slot      │ Owns the GUI window  │
│ IDecisionEngine          │              └──────────────────────┘
│ IDefectStore             │
│ IRejectController        │  Thread 4 (watchdog_thread_)
└──────────────────────────┘  ──────────────────────────
        │ updates              ┌──────────────────────────┐
        └─── last_inference ──▶│ Checks heartbeat every 1s│
             _ms_ (atomic)     │ Warns if stalled > N ms  │
                               └──────────────────────────┘
```

**Display slot** — the inference thread pushes the latest annotated `cv::Mat` to a mutex-protected `std::optional<cv::Mat>` and signals `display_cv_`. The display thread owns `cv::imshow` and `cv::waitKey`, keeping OpenCV's highgui off the inference hot path.

**GPIO pulse worker** — `GpioController` runs a fifth thread (`pulse_worker`) that blocks on a condition variable. `trigger_reject()` signals it rather than spawning a detached thread per reject, eliminating use-after-free on shutdown.

### Thread Ownership Rules

| Resource | Owner Thread | Access Pattern |
|----------|-------------|----------------|
| `camera_` | Capture thread | Exclusive |
| `frame_queue_` | Shared | SPSC: capture writes, inference reads |
| `preprocessor_`, `engine_`, `postprocessor_`, `decision_`, `store_`, `rejector_` | Inference thread | Exclusive |
| `display_frame_` | Shared | Inference writes, display reads — guarded by `display_mutex_` |
| `last_inference_ms_` | Shared | `std::atomic<int64_t>` — inference writes, watchdog reads |
| `stats_` | Shared | Guarded by `stats_mutex_`; callback copied under lock, invoked outside |
| `running_` | Shared | `std::atomic<bool>` — lock-free shutdown signal |

### Synchronization

| Primitive | Purpose |
|-----------|---------|
| `FrameQueue` (`mutex` + `cv`) | Bounded SPSC between capture and inference; back-pressure drops frames when full |
| `display_mutex_` + `display_cv_` | Latest-frame slot between inference and display thread |
| `stats_mutex_` | Guards `PipelineStats` and the result callback pointer |
| `last_inference_ms_` (`atomic<int64_t>`) | Lock-free watchdog heartbeat |
| `running_` (`atomic<bool>`) | Lock-free shutdown across all threads |

### Future: Split Preprocess + Inference

For higher throughput, preprocess and infer can be split onto separate threads:

```
Camera → [Queue] → Preprocess → [Queue] → Inference → [Queue] → Decision + Store + IO
```

Each module is already a separate library with no shared state, making this split mechanical.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | C++17 (inference engine), Python 3.10+ (training/dashboard) |
| **Inference** | ONNX Runtime (supports TensorRT / OpenVINO backends) |
| **Camera** | OpenCV 4.x with V4L2 / GigE Vision / USB3 |
| **Model** | YOLOv8 (Ultralytics) trained via PyTorch |
| **Database** | SQLite3 (WAL mode, local, no network) |
| **Dashboard** | Flask + Chart.js (local web UI) |
| **Build** | CMake 3.20+ |
| **Testing** | Google Test (per-module unit tests) |
| **Deployment** | Docker multi-stage builds |

---

## Hardware Requirements

### Recommended Edge Hardware

| Component | Specification |
|-----------|--------------|
| **Compute** | NVIDIA Jetson Orin Nano / Intel NUC with OpenVINO |
| **Camera** | Basler acA2040-90uc (USB3) or GigE industrial camera |
| **Lighting** | LED ring light / dome light for consistent illumination |
| **Trigger** | Photoelectric sensor for conveyor belt synchronization |
| **GPIO/PLC** | Digital output for reject mechanism actuation |

### Minimum Specs

- CPU: 4-core ARM Cortex-A78 or x86_64
- RAM: 8 GB
- GPU: NVIDIA with >=2GB VRAM (optional, CPU inference supported)
- Storage: 64 GB SSD

---

## Building

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install -y cmake build-essential \
    libopencv-dev libsqlite3-dev libgtest-dev

# Optional: ONNX Runtime (builds with stub if not found)
sudo apt-get install -y libonnxruntime-dev
```

### Build C++ Engine

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This produces 9 module libraries + 3 tools + 7 test binaries:

```
build/
├── lib/
│   ├── libedgeai_common.a
│   ├── libedgeai_camera.a
│   ├── libedgeai_preprocessing.a
│   ├── libedgeai_inference.a
│   ├── libedgeai_postprocessing.a
│   ├── libedgeai_decision.a
│   ├── libedgeai_storage.a
│   ├── libedgeai_io.a
│   └── libedgeai_pipeline.a
└── bin/
    ├── edge_inspector          # Main application
    ├── benchmark_inference     # Latency profiler
    ├── capture_images          # Camera capture tool
    ├── test_types
    ├── test_preprocessing
    ├── test_postprocessing
    ├── test_decision
    ├── test_database
    ├── test_frame_queue
    └── test_integration        # Full pipeline GMock tests
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_TENSORRT` | OFF | Enable TensorRT execution provider |
| `ENABLE_OPENVINO` | OFF | Enable OpenVINO execution provider |
| `ENABLE_GPIO` | OFF | Enable sysfs GPIO hardware I/O |
| `ENABLE_TESTS` | ON | Build unit tests |
| `ENABLE_SANITIZER` | OFF | Enable AddressSanitizer |

### Run Tests

```bash
cd build && ctest --output-on-failure
```

**68 tests across 7 suites — all passing.**

Unit tests link only the module they exercise. The integration test uses GMock to exercise the full pipeline end-to-end without hardware:

| Test binary | Tests | What It Covers |
|---|---|---|
| `test_types` | 11 | Core types, enums, config structs, IoU, stats |
| `test_preprocessing` | 6 | Letterbox, normalize, HWC→CHW tensor conversion |
| `test_postprocessing` | 5 | NMS, confidence filtering, coordinate mapping |
| `test_decision` | 9 | Verdict logic (Pass/Reject/Review), threshold edge cases |
| `test_database` | 4 | SQLite insert, retrieve, defect distribution, defect rate |
| `test_frame_queue` | 8 | Bounded SPSC queue: capacity, FIFO order, producer-consumer |
| `test_integration` | 25 | Full pipeline with GMock: single-frame, threaded, edge cases, strict call-sequence, cross-module data chain |

The integration test (`tests/mocks/`) provides GMock implementations of all 5 interfaces so the full pipeline can be tested without a camera, model, database, or GPIO hardware.

### Setup Python Environment

```bash
cd python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Training Pipeline

```bash
python python/train/train_model.py --data configs/dataset.yaml --epochs 100
```

### Run Edge Inference

```bash
# Live camera
./build/bin/edge_inspector --model models/defect_detector.onnx --camera 0

# Video file (debug)
./build/bin/edge_inspector --model models/defect_detector.onnx --video sample.mp4 --loop

# Single image
./build/bin/edge_inspector --model models/defect_detector.onnx --image test_can.jpg

# With display overlay and custom log path
./build/bin/edge_inspector --model models/defect_detector.onnx --camera 0 \
    --display --log /var/log/edgeai/inspector.log
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--model <path>` | `models/defect_detector.onnx` | ONNX model file |
| `--camera <id>` | `0` | Camera device ID (0–255) |
| `--video <path>` | — | Use video file instead of camera |
| `--loop` | off | Loop video playback |
| `--image <path>` | — | Single-image mode (process one frame and exit) |
| `--log <path>` | `edgeai.log` | Log file path |
| `--display` | off | Show live detection overlay window |
| `--config <path>` | — | Config file (overrides defaults) |
| `--verbose` / `-v` | off | Enable DEBUG log level |

### Run Dashboard

```bash
python python/dashboard/app.py
# Open http://localhost:5000
```

---

## Runtime Reliability

The following production-hardening measures are implemented in the current codebase:

| Area | Implementation |
|---|---|
| **GPIO thread safety** | `GpioController` runs a persistent `pulse_worker` thread blocked on a condition variable. `trigger_reject()` signals it; `cleanup()` sets a stop flag and joins. No detached threads, no use-after-free on shutdown. |
| **Database atomicity** | `insert_result()` wraps the inspection + detection inserts in `BEGIN`/`COMMIT`/`ROLLBACK`. A crash mid-write cannot leave orphaned inspection rows. |
| **Callback deadlock prevention** | The result callback is copied under `stats_mutex_` and invoked outside it, so a user callback that acquires any lock cannot deadlock. |
| **Display thread isolation** | `cv::imshow` and `cv::waitKey` run on a dedicated `display_loop()` thread. The inference thread only pushes to a latest-frame slot, keeping the inference hot path free of GUI blocking. |
| **Model file validation** | `OnnxEngine::verify_model_file()` checks that the `.onnx` file exists, is > 1 KB, and starts with the ONNX protobuf magic byte (`0x08`) before handing it to ONNX Runtime. |
| **CLI input safety** | `--camera` is parsed with `try/catch` and range-checked [0, 255]. Invalid input prints an error and shows help rather than throwing `std::invalid_argument`. |
| **Configurable log path** | Log file is set via `--log <path>` CLI flag or `pipeline.log_path` in the config file. Defaults to `edgeai.log` in the working directory. |
| **Inference watchdog** | `watchdog_loop()` checks `last_inference_ms_` (updated per frame) every second. If no frame is processed within `pipeline.inference_watchdog_timeout_ms` (default 10 000 ms), it logs a `WARN`. Set to `0` to disable. |

### Configuration Keys (inference.cfg)

```ini
# --- existing keys ---
inference.model_path         = models/defect_detector.onnx
inference.conf_threshold     = 0.5
inference.review_threshold   = 0.3
pipeline.queue_capacity      = 32
pipeline.max_db_records      = 100000

# --- added keys ---
pipeline.log_path                      = /var/log/edgeai/inspector.log
pipeline.inference_watchdog_timeout_ms = 10000   # 0 = disabled
```

---

## Docker Deployment

```bash
docker-compose up -d
```

| Container | Purpose |
|-----------|---------|
| `inference` | C++ edge inspector with camera access |
| `dashboard` | Flask web dashboard on port 5000 |

---

## Project Structure

```
edgeAI/
├── CMakeLists.txt                  # Root build — 9 module libraries
├── docker-compose.yml              # Multi-container deployment
├── Dockerfile.inference            # C++ inference container
├── Dockerfile.dashboard            # Python dashboard container
│
├── include/edgeai/
│   ├── common/                     # Foundation layer
│   │   ├── types.h                 #   Core types, enums, config structs
│   │   ├── logger.h                #   Thread-safe singleton logger
│   │   ├── timer.h                 #   ScopedTimer, LatencyTracker
│   │   └── config_loader.h         #   Key-value config parser
│   │
│   ├── camera/                     # Camera module
│   │   ├── icamera.h               #   ★ ICamera interface
│   │   └── camera_manager.h        #   OpenCV VideoCapture implementation
│   │
│   ├── inference/                  # Inference + pre/post processing
│   │   ├── iengine.h               #   ★ IInferenceEngine interface
│   │   ├── onnx_engine.h           #   ONNX Runtime (tensor-in, tensor-out)
│   │   ├── preprocessing.h         #   Letterbox, normalize, tensor convert
│   │   └── postprocessing.h        #   NMS, decode model output → Detections
│   │
│   ├── decision/                   # Business logic module
│   │   ├── idecision_engine.h      #   ★ IDecisionEngine interface
│   │   └── decision_engine.h       #   Threshold-based verdict determination
│   │
│   ├── db/                         # Storage module
│   │   ├── idefect_store.h         #   ★ IDefectStore interface
│   │   └── defect_database.h       #   SQLite implementation
│   │
│   ├── io/                         # I/O module
│   │   ├── ireject_controller.h    #   ★ IRejectController interface
│   │   └── gpio_controller.h       #   sysfs GPIO implementation
│   │
│   └── pipeline/                   # Pipeline module (orchestrator)
│       ├── defect_pipeline.h       #   Multi-threaded pipeline (uses interfaces)
│       └── frame_queue.h           #   Bounded SPSC queue template
│
├── src/
│   ├── main.cpp                    # ★ Composition root — wires all modules
│   ├── tools/benchmark.cpp         # Latency profiler
│   ├── camera/                     # Camera implementations
│   ├── inference/                  # Inference, preprocessing, postprocessing
│   ├── decision/                   # Decision engine implementation
│   ├── pipeline/                   # Pipeline implementation
│   ├── db/                         # Storage implementations
│   ├── io/                         # I/O implementations
│   └── common/                     # Common implementations
│
├── python/
│   ├── train/train_model.py        # YOLOv8 training + ONNX export
│   ├── data/generate_synthetic_data.py  # Synthetic dataset generator
│   ├── dashboard/app.py            # Flask monitoring dashboard
│   └── requirements.txt
│
├── configs/
│   ├── inference.cfg               # Runtime configuration
│   └── dataset.yaml                # Training dataset config
│
├── tests/                          # Per-module unit tests + integration tests
│   ├── CMakeLists.txt
│   ├── test_types.cpp
│   ├── test_preprocessing.cpp
│   ├── test_postprocessing.cpp     # NMS + decode tests
│   ├── test_decision.cpp           # Verdict logic tests
│   ├── test_database.cpp
│   ├── test_frame_queue.cpp
│   ├── test_integration.cpp        # Full pipeline GMock (25 tests)
│   └── mocks/                      # GMock implementations of all 5 interfaces
│       ├── mock_camera.h
│       ├── mock_inference_engine.h
│       ├── mock_decision_engine.h
│       ├── mock_defect_store.h
│       └── mock_reject_controller.h
│
└── docs/
    └── BUSINESS_MODEL.md           # Market analysis & revenue model
```

Items marked with ★ are the **key abstraction points** — 5 interfaces that enable module swapping.

---

## Performance Targets

| Metric | Target | Measured |
|--------|--------|----------|
| Inference Latency | < 30ms | TBD |
| End-to-End Latency | < 50ms | TBD |
| Detection Accuracy | > 95% mAP@0.5 | TBD |
| False Positive Rate | < 2% | TBD |
| Throughput | 60+ items/min | TBD |

---

## Adapting to a New Domain

To pivot from paint cans to another product (e.g., potato sorting, PCB inspection):

### Software Changes (~30 lines)

1. **Retrain the model** — new dataset, run `train_model.py`, export `.onnx`
2. **Update `DefectType` enum** in `types.h` (~10 lines)
3. **Update `class_id_to_type()`** in `postprocessing.cpp` (~10 lines)
4. **Tune `DecisionEngine` thresholds** or implement custom `IDecisionEngine` (~10 lines)
5. **Update dashboard labels** in `app.py`
6. **Swap the `.onnx` file** — `models/defect_detector.onnx`

**Zero changes** to camera, inference runtime, pipeline, storage, I/O, build system, or Docker deployment.

### Hardware Changes (domain-specific, not abstractable)

> **Important:** Real domain pivots require hardware work that cannot be abstracted in software:

| Domain Change | Hardware Considerations |
|--------------|------------------------|
| Cans → Potatoes | Different lighting (dome vs. bar), different belt speed, different reject mechanism (air puff vs. pusher) |
| Cans → PCBs | Microscope camera, backlighting, vibration isolation, different conveyor type |
| Indoor → Outdoor | Weatherproofing, IR lighting, temperature management |
| Single → Multi-camera | Synchronization, bandwidth planning, multi-angle mounting |

The software architecture handles these gracefully (implement a new `ICamera`, tune `IDecisionEngine`), but the physical setup requires domain expertise.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
