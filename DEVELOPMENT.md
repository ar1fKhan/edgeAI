# Developer Guide — EdgeAI Setup, Build & Test

> Comprehensive guide for new team members to set up, build, test, and verify the EdgeAI defect detection system.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup (5–10 minutes)](#initial-setup-5–10-minutes)
3. [Build Instructions](#build-instructions)
4. [Running Tests](#running-tests)
5. [Verification & Demo](#verification--demo)
6. [Model & Data Setup](#model--data-setup)
7. [IDE Setup (VS Code)](#ide-setup-vs-code)
8. [Development Workflow](#development-workflow)
9. [Troubleshooting](#troubleshooting)
10. [Next Steps](#next-steps)

---

## Prerequisites

### System Requirements

- **OS:** Ubuntu 20.04+ / Debian 11+
- **CPU:** 4+ cores (ARM64 or x86_64)
- **RAM:** 8 GB minimum
- **Disk:** 10 GB+ (includes dependencies, build artifacts, and synthetic training data)

### Verify Prerequisites

Run this script to check all dependencies:

```bash
#!/bin/bash
# Check prereqs
echo "✓ Checking prerequisites..."

# Check CMake
cmake --version 2>/dev/null && echo "  ✓ CMake found" || echo "  ✗ CMake NOT found"

# Check C++ compiler
g++ --version 2>/dev/null | head -1 && echo "  ✓ G++ found" || echo "  ✗ G++ NOT found"

# Check build-essential
dpkg -l | grep build-essential > /dev/null && echo "  ✓ build-essential found" || echo "  ✗ build-essential NOT found"

# Check OpenCV
pkg-config --modversion opencv4 2>/dev/null && echo "  ✓ OpenCV 4.x found" || echo "  ✗ OpenCV 4.x NOT found"

# Check SQLite3
sqlite3 --version 2>/dev/null && echo "  ✓ SQLite3 found" || echo "  ✗ SQLite3 NOT found"

# Check Python 3
python3 --version 2>/dev/null && echo "  ✓ Python 3 found" || echo "  ✗ Python 3 NOT found"

# Check git
git --version 2>/dev/null && echo "  ✓ Git found" || echo "  ✗ Git NOT found"

echo ""
```

### Install Dependencies

#### Ubuntu/Debian

```bash
# System packages
sudo apt-get update
sudo apt-get install -y \
    cmake \
    build-essential \
    git \
    libopencv-dev \
    libsqlite3-dev \
    libgtest-dev \
    python3 \
    python3-venv

# ONNX Runtime (optional, system will use stub if not available)
# For ARM64:
sudo apt-get install -y libonnxruntime-dev

# For x86_64, install from source (see troubleshooting)
```

#### macOS (Intel/Apple Silicon)

```bash
brew install cmake opencv sqlite3 python3 googletest
```

---

## Initial Setup (5–10 minutes)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd edgeAI
```

### Step 2: Verify Directory Structure

```bash
ls -la | grep -E "CMakeLists|README|src|include|tests|python"
```

Expected output:
```
CMakeLists.txt
README.md
DEVELOPMENT.md
src/
include/
tests/
python/
```

### Step 3: Initialize Python Virtual Environment

This is needed for model training + dashboard, but **NOT** for C++ build/test.

```bash
# Create virtual environment
python3 -m venv python_env

# Activate it
source python_env/bin/activate  # Linux/macOS
# or:
python_env\Scripts\activate    # Windows PowerShell

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r python/requirements.txt
```

**⏱️ Time:** ~2 minutes (downloads PyTorch, Ultralytics, ONNX, etc.)

### Step 4: Create Build Directory

```bash
mkdir -p build
cd build
```

---

## Build Instructions

### Configure CMake

```bash
# From build/ directory
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Or for Release (optimized):
cmake .. -DCMAKE_BUILD_TYPE=Release
```

**Expected output:**
```
-- The C compiler identification is GNU ...
-- The CXX compiler identification is GNU ...
-- Detecting CXX compile features - done
-- Found OpenCV: 4.x.x ...
-- Configuring done
-- Generating build files done
```

If **ONNX Runtime not found**, you'll see:
```
-- WARNING: ONNX Runtime not found. Using stub implementation.
```

This is fine for first build. See [Troubleshooting → ONNX Runtime](#onnx-runtime-not-found) for production setup.

### Compile All Targets

```bash
# From build/ directory
cmake --build . -j$(nproc)

# Or explicitly:
make -j4
```

**Expected output (final lines):**
```
[ 95%] Building CXX object CMakeFiles/edge_inspector.dir/src/main.cpp.o
[100%] Linking CXX executable bin/edge_inspector
[100%] Built target edge_inspector
```

### Build Artifacts

After successful build, you'll have:

```
build/
├── bin/
│   ├── edge_inspector              # Main inference application
│   ├── benchmark_inference         # Latency profiler
│   ├── capture_images              # Image capture utility
│   ├── test_types                  # Unit test: core types
│   ├── test_preprocessing          # Unit test: image preprocessing
│   ├── test_postprocessing         # Unit test: YOLO output parsing
│   ├── test_decision               # Unit test: decision logic
│   ├── test_database               # Unit test: SQLite storage
│   ├── test_frame_queue            # Unit test: thread-safe queue
│   └── test_integration            # Integration test: end-to-end pipeline
│
└── lib/
    ├── libedgeai_common.a
    ├── libedgeai_camera.a
    ├── libedgeai_preprocessing.a
    ├── libedgeai_inference.a
    ├── libedgeai_postprocessing.a
    ├── libedgeai_decision.a
    ├── libedgeai_storage.a
    ├── libedgeai_io.a
    └── libedgeai_pipeline.a
```

---

## Running Tests

### Run All Tests

```bash
# From build/ directory
ctest --output-on-failure
```

### Expected Output

On a fresh build with synthetic data + trained model:

```
Test project /home/user/edgeAI/build
    Start  1: test_types
1/25 Test #1: test_types ..................... Passed    0.05 sec
    Start  2: test_preprocessing
2/25 Test #2: test_preprocessing ............ Passed    0.08 sec
    Start  3: test_postprocessing
3/25 Test #3: test_postprocessing ........... Passed    0.12 sec
    ...
   22/25 Test #22: FrameQueueTest.BlockOnFull ... Passed    0.05 sec
   23/25 Test #23: SingleFrameIntegration.HighConfidenceDefect_Reject
23/25 Test #23: .................................... Passed    0.45 sec
   24/25 Test #24: EndToEndSingleFrame.RealDecision ...
24/25 Test #24: .................................... Passed    0.52 sec
   25/25 Test #25: IntegrationVideoPlayback.MultiFrame_VideoProcessing
25/25 Test #25: .................................... Passed    1.23 sec

100% tests passed, 0 tests failed out of 25

Total Test time (real) = 3.51 sec
```

### Success Criteria

- ✅ **25/25 tests PASS** (100% pass rate)
- ✅ Total test time < 5 seconds
- ✅ No segmentation faults or memory issues

### Individual Test Targets

Run specific test suites:

```bash
# Unit tests (fast, ~0.1s each)
./bin/test_types
./bin/test_preprocessing
./bin/test_postprocessing
./bin/test_decision
./bin/test_database

# Integration tests (slower, ~0.5–1.5s each)
./bin/test_frame_queue
./bin/test_integration
```

---

## Verification & Demo

### Step 1: Generate Model & Test Data

Before running inference demos, you must generate the trained model:

```bash
# From edgeAI/ root directory (not build/)
source python_env/bin/activate

# Generate synthetic dataset + train model + export to ONNX
python quick_setup.py
```

**⏱️ Time:** ~2–3 minutes

**Expected output:**
```
🚀 EdgeAI Quick Setup — Generating Synthetic Data & Training Model
📊 Step 1: Generating synthetic dataset...
  Creating 100 synthetic images (80 train, 20 val)...
  ✓ Training images: data/synthetic_dataset/images/train/
  ✓ Validation images: data/synthetic_dataset/images/val/

🧠 Step 2: Training YOLOv8 nano...
  Epoch 1/15: ...
  Epoch 15/15: ...
  ✓ Model trained

💾 Step 3: Exporting to ONNX...
  ✓ Export complete: models/defect_detector.onnx (11.7 MB)

🖼️  Step 4: Creating test images...
  ✓ Clean can: data/test_images/clean_can.jpg
  ✓ Defective can: data/test_images/defective_can.jpg

🎬 Step 5: Creating test video...
  ✓ Video: data/test_videos/sample.mp4 (30 frames)

🎉 SETUP COMPLETE!
```

All files created:
```
models/
└── defect_detector.onnx        ← Trained model

data/
├── synthetic_dataset/          ← Training dataset (100 images)
├── test_images/                ← Test images
│   ├── clean_can.jpg
│   └── defective_can.jpg
└── test_videos/
    └── sample.mp4              ← Test video (30 frames)
```

### Step 2: Test Single Image Inference

```bash
# From edgeAI/ root (or any directory)
./build/bin/edge_inspector \
    --image data/test_images/clean_can.jpg \
    --model models/defect_detector.onnx
```

**Expected output:**
```
[INFO] Loading model: models/defect_detector.onnx
[INFO] Inference latency: ~160-200ms on CPU (ARM64)
[INFO] Frame: 1
[INFO] Detections: 0 (clean image)
[INFO] Inference time: 0.19s
[INFO] Verdict: PASS ✓
```

### Step 3: Test Video Inference

```bash
./build/bin/edge_inspector \
    --video data/test_videos/sample.mp4 \
    --model models/defect_detector.onnx \
    --loop
```

**Expected output:**
```
[INFO] Loading model: models/defect_detector.onnx
[INFO] Processing video: data/test_videos/sample.mp4 (30 frames)
[INFO] Looping enabled
[INFO] Frame 1/30: [==>              ] Verdict: PASS
[INFO] Frame 2/30: [===>             ] Verdict: PASS
...
[INFO] Frame 30/30: [================] Verdict: PASS
[INFO] Video complete. Total processing time: 5.2s
[INFO] Average FPS: 5.7
```

### Step 4: Benchmark Inference Performance

```bash
./build/bin/benchmark_inference --model models/defect_detector.onnx --warmup 5 --iterations 50
```

**Expected output:**
```
[INFO] Benchmarking inference engine...
[INFO] Model: models/defect_detector.onnx
[INFO] Warmup: 5 iterations
[INFO] Executed: 50 iterations
[INFO] Min latency: 158ms
[INFO] Max latency: 196ms
[INFO] Mean latency: 175ms (±8ms std)
[INFO] Throughput: 5.7 items/sec
```

**Expected Performance (CPU, ARM64):**
- Per-frame latency: 160–200ms
- Throughput: ~5–6 items/sec
- Memory usage: <500 MB

---

## Model & Data Setup

### Where Does `defect_detector.onnx` Come From?

The trained model comes from **`quick_setup.py`** and is stored in `models/defect_detector.onnx`.

**Important:** The model is **NOT checked into Git** (see `.gitignore`). Every developer must generate it:

```bash
source python_env/bin/activate
python quick_setup.py
```

### What Is `quick_setup.py`?

Automated end-to-end pipeline:

1. **Generates synthetic dataset** — 100 images with 5 defect types
2. **Trains YOLOv8 nano** — 15 epochs on synthetic data
3. **Exports to ONNX** — compatible with ONNX Runtime
4. **Creates test artifacts** — clean/defective images + sample video

### Retraining With Real Data

For production, replace synthetic data with real images:

```bash
# 1. Prepare your dataset (YOLO format)
#    data/your_dataset/
#    ├── images/train/
#    │   ├── img1.jpg
#    │   └── ...
#    └── labels/train/
#        ├── img1.txt  (YOLO format: class cx cy w h)
#        └── ...

# 2. Update configs/dataset.yaml with your dataset path

# 3. Train
python python/train/train_model.py \
    --data configs/dataset.yaml \
    --epochs 100 \
    --img 640 \
    --batch 16
```

### Model Architecture

- **Base:** YOLOv8 nano (11.7 MB, ~1.1M parameters)
- **Input:** 640×640 RGB image
- **Output:** Bounding boxes + confidence scores (5 defect classes)
- **Inference Framework:** ONNX Runtime (CPU, 4 threads)

For faster inference on edge hardware, see [Troubleshooting → Enable TensorRT](#enable-tensorrt-for-faster-inference).

---

## IDE Setup (VS Code)

### Recommended Extensions

```json
{
  "recommendations": [
    "ms-vscode.cpptools",                    // C++ IntelliSense + Debug
    "ms-vscode.cmake-tools",                 // CMake integration
    "twxs.cmake",                             // CMake syntax highlighting
    "ms-python.python",                       // Python support
    "ms-python.vscode-pylance",               // Python type checking
    "eamodio.gitlens",                        // Git history
    "GitHub.Copilot"                          // Optional: AI pair programming
  ]
}
```

Install:
```bash
code --install-extension ms-vscode.cpptools
code --install-extension ms-vscode.cmake-tools
code --install-extension ms-python.python
```

### Configure CMake Tools

VS Code will auto-detect CMakeLists.txt. Check `.vscode/settings.json`:

```json
{
  "cmake.configureOnOpen": true,
  "cmake.sourceDirectory": "${workspaceFolder}",
  "cmake.buildDirectory": "${workspaceFolder}/build",
  "cmake.generator": "Unix Makefiles",
  "cmake.buildArgs": ["-j4"],
  "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools"
}
```

### Build & Test from VS Code

1. **Configure & Build:**
   - Press `Ctrl+Shift+P` → `CMake: Configure`
   - Press `Ctrl+Shift+B` → Build (or use task "Build Debug")

2. **Run Tests:**
   - `Ctrl+Shift+P` → `CMake: Run Tests`
   - Or from terminal: `ctest --output-on-failure`

3. **Debug Tests:**
   - In [`.vscode/launch.json`](.vscode/launch.json), select test target (pre-configured)
   - Press `F5` to run with debugger

### Recommended Workflow

1. **Open folder:** `File → Open Folder → ~/edgeAI`
2. **Configure CMake:** `Ctrl+Shift+P → CMake: Configure`
3. **Edit code** (VS Code indexes symbols via Pylance/Clangd)
4. **Build & test:** `Ctrl+Shift+B` (builds) → `ctest` (tests)
5. **Debug:** Set breakpoints, press `F5`

---

## Development Workflow

### Standard Workflow for New Features

#### 1. Create Feature Branch

```bash
git checkout -b feature/my-feature
```

#### 2. Make Changes

- **C++ changes:** Modify headers in `include/`, implementations in `src/`
- **Tests:** Add test cases in `tests/`
- **Python:** Update scripts in `python/` as needed

#### 3. Verify Locally

```bash
# Rebuild
cd build && cmake --build . -j4

# Run all tests
ctest --output-on-failure

# Run targeted tests for your feature
./bin/test_<module>    # e.g., test_decision, test_postprocessing
```

#### 4. Check Code Quality

```bash
# Optional: static analysis (if clang-tidy installed)
# clang-tidy src/my_changes.cpp -- -I include

# Code style (ensure consistent with existing)
# Review README.md code style section
```

#### 5. Commit & Push

```bash
git add -A
git commit -m "feat: add feature description"
git push origin feature/my-feature
```

#### 6. Create Pull Request

Ensure:
- ✅ All 25 tests pass
- ✅ No compiler warnings
- ✅ Code follows existing style (C++17, RAII, smart pointers)
- ✅ New public APIs documented

### Common Tasks

#### Add a New Defect Class

1. **Update enum** in `include/edgeai/common/types.h`:
   ```cpp
   enum class DefectType {
       DENT,
       WRONG_LABEL,
       MISSING_LABEL,
       SEAL_DEFECT,
       COLOR_MISMATCH,
       YOUR_NEW_CLASS    // ← Add here
   };
   ```

2. **Update mapper** in `src/inference/postprocessing.cpp`:
   ```cpp
   DefectType class_id_to_type(int class_id) {
       switch (class_id) {
           // ...
           case 5: return DefectType::YOUR_NEW_CLASS;
       }
   }
   ```

3. **Retrain model** with new class in dataset

4. **Update decision thresholds** in `src/decision/decision_engine.cpp` if needed

#### Swap Inference Backend (e.g., TensorRT)

1. **Create new `ITensorRtEngine` implementation** in `src/inference/tensorrt_engine.cpp`
2. **Update `main.cpp`** composition root:
   ```cpp
   // auto engine = std::make_unique<OnnxEngine>(config.inference);
   auto engine = std::make_unique<TensorRtEngine>(config.inference);
   ```
3. **Recompile:** `cmake --build . -j4`
4. **Test:** `ctest --output-on-failure`
5. **Pipeline, camera, storage, decision logic → unchanged**

#### Debug with Verbose Logging

Edit `include/edgeai/common/logger.h`:

```cpp
// Change log level
constexpr LogLevel MIN_LOG_LEVEL = LogLevel::DEBUG;  // Shows all logs
```

Then rebuild:
```bash
cmake --build . -j4
```

---

## Troubleshooting

### CMake Configuration Issues

#### `CMake not found`

```bash
# Check CMake version
cmake --version

# If not installed:
sudo apt-get install -y cmake
```

#### `Could NOT find OpenCV`

```bash
# Check OpenCV installation
pkg-config --modversion opencv4

# If missing:
sudo apt-get install -y libopencv-dev

# If on macOS:
brew install opencv
```

#### Google Test not found

```bash
sudo apt-get install -y libgtest-dev

# On some systems, you may need to build it:
cd /usr/src/gtest && sudo cmake . && sudo cmake --build . && sudo cp *.a /usr/lib/
```

### ONNX Runtime Not Found

If CMake says:
```
-- WARNING: ONNX Runtime not found. Using stub implementation.
```

You can continue development with the stub, but for real inference, install ONNX Runtime:

#### Option 1: System Package (Ubuntu)

```bash
sudo apt-get install -y libonnxruntime-dev libonnxruntime1
```

#### Option 2: Manual Installation (ARM64)

```bash
# Download ONNX Runtime v1.18.0 for ARM64
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-aarch64-1.18.0.tgz

# Extract
tar -xzf onnxruntime-linux-aarch64-1.18.0.tgz

# Install
sudo cp -r onnxruntime-linux-aarch64-1.18.0/lib/* /usr/local/lib/
sudo cp -r onnxruntime-linux-aarch64-1.18.0/include/* /usr/local/include/

# Update library cache
sudo ldconfig
```

#### Option 3: x86_64

```bash
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar -xzf onnxruntime-linux-x64-1.18.0.tgz
sudo cp -r onnxruntime-linux-x64-1.18.0/lib/* /usr/local/lib/
sudo cp -r onnxruntime-linux-x64-1.18.0/include/* /usr/local/include/
sudo ldconfig
```

Then reconfigure CMake:

```bash
cd build
rm CMakeCache.txt
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . -j4
```

### Build Failures

#### `fatal error: opencv2/core.hpp: No such file`

```bash
sudo apt-get install -y libopencv-dev
```

#### `undefined reference to 'sqlite3_open'`

```bash
sudo apt-get install -y libsqlite3-dev
```

#### Linker errors with ONNX Runtime

```bash
# Check if library is found
ldconfig -p | grep onnxruntime

# If not listed, add to library path:
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Rebuild
cd build && cmake --build . -j4
```

### Test Failures

#### `test_integration` fails with threshold errors

The integration tests have hardcoded confidence thresholds. If you modify `InferenceConfig` in `types.h`, update test fixtures in `tests/test_integration.cpp`:

```cpp
// Line ~150 in test_integration.cpp
ASSERT_GE(detection.confidence, 0.70f);  // Update threshold here
```

#### Segmentation fault in `test_integration`

**Possible causes:**

1. Model file missing:
   ```bash
   ls -la models/defect_detector.onnx
   ```

2. ONNX Runtime crash (stub mode trying to infer):
   - Install real ONNX Runtime (see above)

3. Memory corruption:
   ```bash
   # Rebuild with AddressSanitizer
   cmake .. -DENABLE_SANITIZER=ON
   ```

### Runtime Issues

#### `./edge_inspector: command not found`

```bash
# Ensure you're in the right directory
cd edgeAI
./build/bin/edge_inspector --help

# Or add to PATH
export PATH=$PWD/build/bin:$PATH
edge_inspector --help
```

#### `[ERROR] Could not open model: models/defect_detector.onnx`

```bash
# Verify model exists
ls -lh models/defect_detector.onnx

# Generate it
source python_env/bin/activate
python quick_setup.py
```

#### `[ERROR] Could not load camera`

This is expected if running headless (no camera attached). Use `--image` or `--video` instead:

```bash
# Single image
./build/bin/edge_inspector --image data/test_images/clean_can.jpg

# Video
./build/bin/edge_inspector --video data/test_videos/sample.mp4
```

### Python Environment Issues

#### `ModuleNotFoundError: No module named 'torch'`

```bash
# Ensure virtual env is activated
source python_env/bin/activate

# Install dependencies
pip install -r python/requirements.txt
```

#### Python version mismatch

```bash
# Check Python version
python --version

# Ensure >= 3.8
python3 --version

# Use python3 explicitly if needed
python3 -m venv python_env
source python_env/bin/activate
pip install -r python/requirements.txt
```

### Enable TensorRT for Faster Inference

For NVIDIA GPUs (Jetson, RTX):

```bash
# Download TensorRT to /opt/tensorrt/
# (Manual download from NVIDIA, requires registration)

# Configure with TensorRT
cmake .. -DENABLE_TENSORRT=ON

# Build
cmake --build . -j4

# TensorRT-enabled binary auto-selects execution provider in OnnxEngine
./build/bin/edge_inspector --image data/test_images/clean_can.jpg
# [INFO] TensorRT execution provider selected
```

---

## Next Steps

### For Feature Development

1. Read [`README.md`](README.md) architecture section
2. Review [`.github/copilot-instructions.md`](.github/copilot-instructions.md) for code style
3. Pick a module to extend (camera, inference, decision, storage, I/O)
4. Create branch & follow [Development Workflow](#development-workflow)

### For Model Improvement

1. Collect real defective product images (~200–500 per defect type)
2. Annotate in YOLO format (see [Model & Data Setup](#model--data-setup))
3. Update `configs/dataset.yaml`
4. Retrain: `python python/train/train_model.py --data configs/dataset.yaml --epochs 100`
5. Test: `ctest --output-on-failure`

### For Deployment

1. Build Release binary:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   cmake --build . -j4
   ```

2. Test on target hardware (Jetson, NUC, etc.)

3. Run performance benchmarks:
   ```bash
   ./build/bin/benchmark_inference --model models/defect_detector.onnx --iterations 100
   ```

4. Package with Docker:
   ```bash
   docker-compose up -d
   ```

### Resources

- **C++ Code Style:** See [`.github/copilot-instructions.md`](.github/copilot-instructions.md)
- **Architecture Deep Dive:** [README.md](README.md) modular architecture section
- **Threading Model:** [README.md](README.md) threading model section
- **ONNX Runtime Docs:** https://onnxruntime.ai/docs/
- **YOLOv8 Training:** https://docs.ultralytics.com/
- **CMake Guide:** https://cmake.org/cmake/help/latest/

---

## Support & Questions

- **Build issues:** Check [Troubleshooting](#troubleshooting)
- **Architecture questions:** See [README.md](README.md)
- **Code style:** Review examples in `src/` and `include/`
- **Test failures:** Run with verbose: `ctest --output-on-failure --verbose`

---

**Happy coding! 🚀**
