#pragma once

/**
 * @file timer.h
 * @brief High-resolution scoped timer for performance profiling.
 */

#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include "edgeai/common/logger.h"

namespace edgeai {

class ScopedTimer {
    using Clock    = std::chrono::steady_clock;
    using Duration = std::chrono::microseconds;

public:
    explicit ScopedTimer(const std::string& name, bool log_on_destruct = true)
        : name_(name), log_(log_on_destruct), start_(Clock::now()) {}

    ~ScopedTimer() {
        if (log_) {
            auto ms = elapsed_ms();
            LOG_DEBUG("Timer", name_ + ": " + std::to_string(ms) + " ms");
        }
    }

    [[nodiscard]] double elapsed_ms() const {
        auto now = Clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

    [[nodiscard]] Duration elapsed() const {
        return std::chrono::duration_cast<Duration>(Clock::now() - start_);
    }

    void reset() { start_ = Clock::now(); }

private:
    std::string                       name_;
    bool                              log_;
    std::chrono::steady_clock::time_point start_;
};

// ── Inline Performance Counter ─────────────────────────────────

class LatencyTracker {
public:
    void record(double ms) {
        ++count_;
        total_ms_ += ms;
        min_ms_ = std::min(min_ms_, ms);
        max_ms_ = std::max(max_ms_, ms);
    }

    [[nodiscard]] double avg_ms() const { return count_ > 0 ? total_ms_ / count_ : 0.0; }
    [[nodiscard]] double min_ms() const { return min_ms_; }
    [[nodiscard]] double max_ms() const { return max_ms_; }
    [[nodiscard]] uint64_t count() const { return count_; }

    void reset() {
        count_ = 0;
        total_ms_ = 0.0;
        min_ms_ = std::numeric_limits<double>::max();
        max_ms_ = 0.0;
    }

private:
    uint64_t count_    = 0;
    double   total_ms_ = 0.0;
    double   min_ms_   = std::numeric_limits<double>::max();
    double   max_ms_   = 0.0;
};

}  // namespace edgeai
