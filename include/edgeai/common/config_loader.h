#pragma once

/**
 * @file config_loader.h
 * @brief Configuration loader with validation and safe parsing.
 * 
 * Parses simple key-value config files for camera, inference, and pipeline settings.
 * Format: key = value (one per line, # comments supported)
 *
 * All numeric parsing is wrapped in try/catch — malformed values fall back to
 * defaults with a warning instead of crashing.
 */

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include "edgeai/common/types.h"
#include "edgeai/common/logger.h"

namespace edgeai {

class ConfigLoader {
public:
    static PipelineConfig load(const std::string& filepath) {
        PipelineConfig config;
        std::unordered_map<std::string, std::string> kv;

        std::ifstream file(filepath);
        if (!file.is_open()) {
            LOG_WARN("Config", "Config file not found: " + filepath + " — using defaults");
            return config;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Strip comments and whitespace
            auto comment_pos = line.find('#');
            if (comment_pos != std::string::npos)
                line = line.substr(0, comment_pos);

            auto eq_pos = line.find('=');
            if (eq_pos == std::string::npos) continue;

            std::string key = trim(line.substr(0, eq_pos));
            std::string val = trim(line.substr(eq_pos + 1));
            if (!key.empty() && !val.empty()) {
                kv[key] = val;
            }
        }

        // ── Camera Settings ────────────────────────────────────
        if (auto it = kv.find("camera.device_id"); it != kv.end())
            config.camera.device_id = safe_stoi(it->first, it->second, 0, 0, 255);
        if (auto it = kv.find("camera.width"); it != kv.end())
            config.camera.width = safe_stoi(it->first, it->second, 1920, 1, 7680);
        if (auto it = kv.find("camera.height"); it != kv.end())
            config.camera.height = safe_stoi(it->first, it->second, 1080, 1, 4320);
        if (auto it = kv.find("camera.fps"); it != kv.end())
            config.camera.fps = safe_stoi(it->first, it->second, 30, 1, 240);
        if (auto it = kv.find("camera.backend"); it != kv.end())
            config.camera.backend = it->second;
        if (auto it = kv.find("camera.trigger_mode"); it != kv.end())
            config.camera.trigger_mode = (it->second == "true");
        if (auto it = kv.find("camera.exposure_us"); it != kv.end())
            config.camera.exposure_us = safe_stoi(it->first, it->second, 10000, 1, 10000000);
        if (auto it = kv.find("camera.video_path"); it != kv.end())
            config.camera.video_path = it->second;
        if (auto it = kv.find("camera.loop_video"); it != kv.end())
            config.camera.loop_video = (it->second == "true");

        // ── Inference Settings ─────────────────────────────────
        if (auto it = kv.find("inference.model_path"); it != kv.end())
            config.inference.model_path = it->second;
        if (auto it = kv.find("inference.input_width"); it != kv.end())
            config.inference.input_width = safe_stoi(it->first, it->second, 640, 32, 4096);
        if (auto it = kv.find("inference.input_height"); it != kv.end())
            config.inference.input_height = safe_stoi(it->first, it->second, 640, 32, 4096);
        if (auto it = kv.find("inference.num_classes"); it != kv.end())
            config.inference.num_classes = safe_stoi(it->first, it->second, 5, 1, 10000);
        if (auto it = kv.find("inference.conf_threshold"); it != kv.end())
            config.inference.conf_threshold = safe_stof(it->first, it->second, 0.5f, 0.0f, 1.0f);
        if (auto it = kv.find("inference.nms_threshold"); it != kv.end())
            config.inference.nms_threshold = safe_stof(it->first, it->second, 0.45f, 0.0f, 1.0f);
        if (auto it = kv.find("inference.num_threads"); it != kv.end())
            config.inference.num_threads = safe_stoi(it->first, it->second, 4, 1, 64);
        if (auto it = kv.find("inference.use_gpu"); it != kv.end())
            config.inference.use_gpu = (it->second == "true");
        if (auto it = kv.find("inference.execution_provider"); it != kv.end())
            config.inference.execution_provider = it->second;

        // ── Pipeline Settings ──────────────────────────────────
        if (auto it = kv.find("pipeline.queue_capacity"); it != kv.end())
            config.queue_capacity = safe_stoi(it->first, it->second, 32, 1, 1024);
        if (auto it = kv.find("pipeline.defect_image_dir"); it != kv.end())
            config.defect_image_dir = it->second;
        if (auto it = kv.find("pipeline.database_path"); it != kv.end())
            config.database_path = it->second;
        if (auto it = kv.find("pipeline.save_defect_images"); it != kv.end())
            config.save_defect_images = (it->second == "true");
        if (auto it = kv.find("pipeline.enable_display"); it != kv.end())
            config.enable_display = (it->second == "true");
        if (auto it = kv.find("pipeline.enable_gpio"); it != kv.end())
            config.enable_gpio = (it->second == "true");
        if (auto it = kv.find("pipeline.gpio_reject_pin"); it != kv.end())
            config.gpio_reject_pin = safe_stoi(it->first, it->second, 17, 0, 255);
        if (auto it = kv.find("pipeline.max_defect_images"); it != kv.end())
            config.max_defect_images = safe_stoi(it->first, it->second, 10000, 0, 1000000);
        if (auto it = kv.find("pipeline.max_db_records"); it != kv.end())
            config.max_db_records = safe_stoi(it->first, it->second, 100000, 0, 10000000);

        LOG_INFO("Config", "Configuration loaded from: " + filepath);
        return config;
    }

private:
    static std::string trim(const std::string& s) {
        auto start = s.find_first_not_of(" \t\r\n");
        auto end = s.find_last_not_of(" \t\r\n");
        return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
    }

    /// Safe integer parse with range validation. Returns default on failure.
    static int safe_stoi(const std::string& key, const std::string& val,
                         int default_val, int min_val, int max_val) {
        try {
            int v = std::stoi(val);
            if (v < min_val || v > max_val) {
                LOG_WARN("Config", key + " = " + val + " out of range ["
                         + std::to_string(min_val) + ", " + std::to_string(max_val)
                         + "] — using default " + std::to_string(default_val));
                return default_val;
            }
            return v;
        } catch (const std::exception&) {
            LOG_WARN("Config", "Invalid integer for " + key + " = '" + val
                     + "' — using default " + std::to_string(default_val));
            return default_val;
        }
    }

    /// Safe float parse with range validation. Returns default on failure.
    static float safe_stof(const std::string& key, const std::string& val,
                           float default_val, float min_val, float max_val) {
        try {
            float v = std::stof(val);
            if (v < min_val || v > max_val) {
                LOG_WARN("Config", key + " = " + val + " out of range — using default "
                         + std::to_string(default_val));
                return default_val;
            }
            return v;
        } catch (const std::exception&) {
            LOG_WARN("Config", "Invalid float for " + key + " = '" + val
                     + "' — using default " + std::to_string(default_val));
            return default_val;
        }
    }
};

}  // namespace edgeai
