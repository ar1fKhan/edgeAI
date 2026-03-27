/**
 * @file gpio_controller.cpp
 * @brief GPIO/PLC control for conveyor reject mechanism.
 */

#include "edgeai/io/gpio_controller.h"
#include <fstream>

namespace edgeai {

GpioController::GpioController(int pin, int pulse_duration_ms)
    : pin_(pin), pulse_ms_(pulse_duration_ms) {}

GpioController::~GpioController() {
    cleanup();
}

bool GpioController::initialize() {
#ifdef ENABLE_GPIO_HW
    if (!export_pin()) {
        LOG_WARN("GPIO", "Failed to export GPIO pin " + std::to_string(pin_));
        return false;
    }
    if (!set_direction("out")) {
        LOG_WARN("GPIO", "Failed to set GPIO direction");
        return false;
    }
    write_value(0);  // ensure low initially
    initialized_ = true;
    LOG_INFO("GPIO", "GPIO pin " + std::to_string(pin_) + " initialized for reject output");
    return true;
#else
    LOG_INFO("GPIO", "GPIO hardware disabled — running in simulation mode (pin "
             + std::to_string(pin_) + ")");
    initialized_ = true;
    return true;
#endif
}

void GpioController::trigger_reject() {
    if (!initialized_) return;

    ++reject_count_;
    LOG_INFO("GPIO", "Reject pulse triggered (count: " + std::to_string(reject_count_.load()) + ")");

    // Fire-and-forget pulse in detached thread to avoid blocking inference
    std::thread([this]() {
        set_high();
        std::this_thread::sleep_for(std::chrono::milliseconds(pulse_ms_));
        set_low();
    }).detach();
}

void GpioController::set_high() {
#ifdef ENABLE_GPIO_HW
    write_value(1);
#else
    LOG_DEBUG("GPIO", "SIM: Pin " + std::to_string(pin_) + " → HIGH");
#endif
}

void GpioController::set_low() {
#ifdef ENABLE_GPIO_HW
    write_value(0);
#else
    LOG_DEBUG("GPIO", "SIM: Pin " + std::to_string(pin_) + " → LOW");
#endif
}

void GpioController::cleanup() {
#ifdef ENABLE_GPIO_HW
    if (initialized_) {
        write_value(0);
        // Unexport pin
        std::ofstream unexport("/sys/class/gpio/unexport");
        if (unexport.is_open()) {
            unexport << pin_;
        }
    }
#endif
    initialized_ = false;
}

bool GpioController::export_pin() {
    std::ofstream exp("/sys/class/gpio/export");
    if (!exp.is_open()) return false;
    exp << pin_;
    return !exp.fail();
}

bool GpioController::set_direction(const std::string& direction) {
    std::ofstream dir(gpio_path() + "/direction");
    if (!dir.is_open()) return false;
    dir << direction;
    return !dir.fail();
}

bool GpioController::write_value(int value) {
    std::ofstream val(gpio_path() + "/value");
    if (!val.is_open()) return false;
    val << value;
    return !val.fail();
}

std::string GpioController::gpio_path() const {
    return "/sys/class/gpio/gpio" + std::to_string(pin_);
}

}  // namespace edgeai
