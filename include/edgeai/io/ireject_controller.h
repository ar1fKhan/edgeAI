#pragma once

/**
 * @file ireject_controller.h
 * @brief Abstract reject mechanism interface — enables swapping I/O backends.
 *
 * Implement this interface for any reject actuator:
 *   - sysfs GPIO (GpioController)
 *   - PLC / Modbus
 *   - Industrial fieldbus (EtherCAT, PROFINET)
 *   - Simulated (for testing)
 */

namespace edgeai {

class IRejectController {
public:
    virtual ~IRejectController() = default;

    /// Initialize the reject mechanism hardware
    virtual bool initialize() = 0;

    /// Trigger a reject pulse (non-blocking)
    virtual void trigger_reject() = 0;

    /// Release hardware resources
    virtual void cleanup() = 0;

    /// Check if the reject mechanism is operational
    [[nodiscard]] virtual bool is_initialized() const = 0;
};

}  // namespace edgeai
