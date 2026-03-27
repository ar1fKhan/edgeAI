# Project Guidelines

## Code Style
- C++17 code style with RAII, smart pointers, and interface-based dependency injection.
- Follow patterns in `include/edgeai/*` and `src/*` for naming and module boundaries.
- Logging via `edgeai::Logger` macros (`LOG_INFO`, `LOG_WARN`, etc.) in `include/edgeai/common/logger.h`.

## Architecture
- 7 core interfaces: `ICamera`, `IInferenceEngine`, `IPreprocessor`, `IPostprocessor`, `IDecisionEngine`, `IDefectStore`, `IRejectController`.
- Main orchestrator is `src/pipeline/defect_pipeline.cpp`; concrete wiring in `src/main.cpp`.
- `CameraManager` currently supports real cameras and video file sources (`camera.video_path`, `--video`, `--loop`).

## Build and Test
- Config: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug` (or Release) then `cmake --build build`.
- Run unit/integration: `cd build && ctest --output-on-failure`.
- Targets: `edge_inspector`, `capture_images`, `benchmark_inference`, test binaries in `tests/`.

## Project Conventions
- Config parsing in `include/edgeai/common/config_loader.h` uses `safe_stoi` and `safe_stof` with bounds.
- SQL interaction in `src/db/defect_database.cpp` uses prepared statements and `prune()`.
- Thread safety in pipeline with `stats_mutex_` and exception-safe capture/inference loops.
- Error semantics: tool should not rely on C-style string ownership; use `std::string`.

## Integration Points
- OpenCV 4.x for camera/video processing (`opencv2/videoio.hpp`, `opencv2/imgcodecs.hpp`).
- SQLite for persistence (`sqlite3`), DB rotation/pruning by record count.
- Optional GPIO via sysfs in `edgeai/io/gpio_controller.h` (legacy, should be replaced with libgpiod if extended).

## Security
- Avoid string concatenation in SQL; use prepared statements (`src/db/defect_database.cpp`).
- `ConfigLoader` validates ranges for camera and inference values.
- Watch for detached thread lifetimes in `GpioController` and `DefectPipeline` on stop.
