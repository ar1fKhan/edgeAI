#pragma once

/**
 * @file gpio_controller.h
 * @brief GPIO/PLC interface for triggering conveyor belt reject mechanism.
 * 
 * On Linux, uses sysfs GPIO interface.
 * Provides a timed pulse output to actuate pneumatic reject arm.
 */

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include "edgeai/io/ireject_controller.h"
#include "edgeai/common/types.h"
#include "edgeai/common/logger.h"

namespace edgeai {

class GpioController : public IRejectController {
public:
    GpioController(int pin, int pulse_duration_ms = 200);
    ~GpioController();

    // Non-copyable
    GpioController(const GpioController&) = delete;
    GpioController& operator=(const GpioController&) = delete;

    /// Initialize GPIO pin as output
    bool initialize() override;

    /// Trigger a reject pulse (non-blocking, runs in detached thread)
    void trigger_reject() override;

    /// Set pin high
    void set_high();

    /// Set pin low
    void set_low();

    /// Release GPIO resources
    void cleanup() override;

    /// Check if GPIO is operational
    [[nodiscard]] bool is_initialized() const override { return initialized_; }

    /// Get total reject count
    [[nodiscard]] uint64_t reject_count() const { return reject_count_.load(); }

private:
    int                    pin_;
    int                    pulse_ms_;
    bool                   initialized_ = false;
    std::atomic<uint64_t>  reject_count_{0};

    bool export_pin();
    bool set_direction(const std::string& direction);
    bool write_value(int value);
    std::string gpio_path() const;
};

}  // namespace edgeai
