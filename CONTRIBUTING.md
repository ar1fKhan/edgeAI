# Contributing to EdgeAI

## How to Contribute

1. **Fork** the repository
2. **Clone** your fork locally
```bash
git clone https://github.com/YOUR_USERNAME/edgeAI.git
cd edgeAI
```
3. **Create a branch** for your change
```bash
git checkout -b feature/your-feature-name
```
4. **Make your changes** and ensure CI passes locally
```bash
# Install dependencies first — see README Prerequisites section
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```
5. **Push** to your fork
```bash
git push origin feature/your-feature-name
```
6. **Open a Pull Request** against `ar1fKhan/edgeAI:main`

---

## PR Requirements

- CI must pass (both `Build & Test (stub)` and `Build & Test (ONNX Runtime)` jobs)
- At least one approval required
- All review comments must be resolved
- Branch must be up to date with `main`

---

## Branch Naming

| Type | Pattern | Example |
|---|---|---|
| Feature | `feature/description` | `feature/tensorrt-backend` |
| Bug fix | `fix/description` | `fix/memory-leak-postprocessor` |
| Docs | `docs/description` | `docs/add-jetson-setup` |
| Test | `test/description` | `test/gpio-controller-coverage` |

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short summary>

<optional body>
```

| Type | When to use |
|---|---|
| `feat` | New feature or module |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `test` | Adding or fixing tests |
| `docs` | Documentation only |
| `ci` | CI/build system changes |
| `refactor` | Refactor without behaviour change |

**Examples:**
```
feat: add TensorRT execution provider backend
fix: correct NMS threshold comparison direction
perf: replace per-detection heap alloc with preallocated buffer
test: add mock for IPreprocessor interface
```

---

## Code Conventions

This project follows strict conventions — please match them or your PR will be asked to revise.

### Architecture rules

- **Interfaces live in `include/edgeai/`** — every swappable module has a pure-virtual interface (`ICamera`, `IInferenceEngine`, etc.)
- **Concrete implementations are wired only in `src/main.cpp`** — the pipeline library must never `#include` a concrete type
- **`edgeai_pipeline` depends on interfaces only** — if your change requires the pipeline to know about a concrete type, the design is wrong
- **Each module is its own static library** — don't add cross-module dependencies that aren't already in the graph

### Adding a new backend (the right way)

To add e.g. a TensorRT backend:

1. Implement `IInferenceEngine` in a new file `src/inference/tensorrt_engine.cpp`
2. Add a `CMake` option `ENABLE_TENSORRT` (already stubbed in `CMakeLists.txt`)
3. Change **one line** in `src/main.cpp`: `make_unique<TensorRTEngine>(...)`
4. Add a mock in `tests/mocks/` following the existing pattern
5. Add unit tests that exercise the new implementation directly

The pipeline library should have **zero changes**.

### C++ style

- Standard: **C++17** (`std::optional`, structured bindings, `if constexpr`)
- No raw `stoi` / `stof` — use `ConfigLoader::safe_stoi` / `safe_stof` with range bounds
- SQL: always `sqlite3_prepare_v2` + `sqlite3_bind_*` — no string-concatenated queries
- Logging: `LOG_INFO`, `LOG_WARN`, `LOG_ERROR`, `LOG_DEBUG` macros — never `std::cout` in library code
- No comments explaining *what* the code does — only *why* when non-obvious
- No `using namespace std` in headers

### Testing requirements

- Unit tests must link **only the module under test** — no pipeline, no concrete implementations
- Integration tests must use **GMock mocks from `tests/mocks/`** — no real hardware, files, or network
- New interfaces must have a corresponding mock in `tests/mocks/`
- New modules must have a test binary added to `tests/CMakeLists.txt`

---

## What We Welcome

- New execution provider backends (TensorRT, OpenVINO, CoreML)
- New camera source implementations (GigE, RTSP, RealSense)
- Performance improvements with `benchmark_inference` evidence (before/after numbers)
- New defect domain support (update `DefectType` enum + `class_id_to_type()` + model)
- Additional test coverage, especially threading edge cases
- Documentation improvements and hardware setup guides

## What We Won't Merge

- Changes that couple `edgeai_pipeline` to a concrete implementation
- Raw `stoi`/SQL string concatenation/`std::cout` in library code
- PRs that reduce test coverage
- Performance claims without benchmark numbers
- New dependencies without justification (this is an edge project — binary size and boot time matter)

---

## Getting Help

Open a [GitHub Issue](https://github.com/ar1fKhan/edgeAI/issues) for bugs, questions, or proposals before starting large changes.
