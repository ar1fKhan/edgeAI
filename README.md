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

The pipeline uses a **2-thread producer-consumer** architecture with explicit ownership semantics:

```
Thread 1 (capture_thread_)         Thread 2 (inference_thread_)
──────────────────────────         ──────────────────────────────
┌──────────────┐                   ┌──────────────────────────┐
│ ICamera      │                   │ Preprocessor             │
│  .capture()  │──── FrameQueue ──▶│  .preprocess()           │
│              │    (bounded SPSC) │                          │
└──────────────┘                   │ IInferenceEngine         │
                                   │  .infer()                │
Owns: camera_                      │                          │
Writes: frame_queue_               │ Postprocessor            │
                                   │  .process() + .apply_nms()│
                                   │                          │
                                   │ IDecisionEngine          │
                                   │  .decide()               │
                                   │                          │
                                   │ IDefectStore             │
                                   │  .log_result()           │
                                   │                          │
                                   │ IRejectController        │
                                   │  .trigger() (if reject)  │
                                   └──────────────────────────┘

                                   Owns: preprocessor_, engine_,
                                         postprocessor_, decision_,
                                         store_, rejector_
                                   Reads: frame_queue_
```

### Thread Ownership Rules

| Resource | Owner Thread | Access Pattern |
|----------|-------------|----------------|
| `camera_` | Capture thread | Exclusive write |
| `frame_queue_` | Shared | SPSC: capture writes, inference reads |
| `preprocessor_` | Inference thread | Exclusive |
| `engine_` | Inference thread | Exclusive |
| `postprocessor_` | Inference thread | Exclusive |
| `decision_` | Inference thread | Exclusive |
| `store_` | Inference thread | Exclusive |
| `rejector_` | Inference thread | Exclusive |
| `running_` | Shared | `std::atomic<bool>` — read by both, set by `stop()` |

### Synchronization

- **FrameQueue**: bounded single-producer/single-consumer queue with `std::mutex` + `std::condition_variable`. Back-pressure: capture blocks when queue is full (avoids unbounded memory growth).
- **`running_`**: `std::atomic<bool>` — enables lock-free shutdown signaling across threads.
- No other shared mutable state exists between threads.

### Future: 4-Stage Pipeline (planned)

For higher throughput, the architecture supports splitting into 4 threads:

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

This produces 9 module libraries + 2 executables:

```
build/
├── libedgeai_common.a
├── libedgeai_camera.a
├── libedgeai_preprocessing.a
├── libedgeai_inference.a
├── libedgeai_postprocessing.a
├── libedgeai_decision.a
├── libedgeai_storage.a
├── libedgeai_io.a
├── libedgeai_pipeline.a
└── bin/
    ├── edge_inspector          # Main application
    └── benchmark_inference     # Latency profiler
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

Tests link only the module they exercise:

| Test | Links Against | What It Tests |
|------|--------------|---------------|
| `test_types` | `edgeai_common` | Core types, enums, config structs |
| `test_preprocessing` | `edgeai_preprocessing` | Letterbox, normalize, tensor conversion |
| `test_postprocessing` | `edgeai_postprocessing` | NMS, confidence filtering, coordinate mapping |
| `test_decision` | `edgeai_decision` | Verdict determination (Pass/Reject/Review) |
| `test_database` | `edgeai_storage` | SQLite defect logging |
| `test_frame_queue` | `edgeai_common` | Bounded SPSC queue |

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
./build/bin/edge_inspector --model models/defect_detector.onnx --camera 0
```

### Run Dashboard

```bash
python python/dashboard/app.py
# Open http://localhost:5000
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
├── tests/                          # Per-module unit tests
│   ├── CMakeLists.txt
│   ├── test_types.cpp
│   ├── test_preprocessing.cpp
│   ├── test_postprocessing.cpp     # NMS + decode tests
│   ├── test_decision.cpp           # Verdict logic tests
│   ├── test_database.cpp
│   └── test_frame_queue.cpp
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

Proprietary — All rights reserved.
