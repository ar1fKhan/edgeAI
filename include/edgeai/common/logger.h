#pragma once

/**
 * @file logger.h
 * @brief Thread-safe logging with severity levels and size-based file rotation.
 *
 * Rotation: when the log file exceeds max_file_size_ bytes, the current file
 * is renamed to <name>.1.log, .2.log etc. (older files shift up), and a new
 * file is opened.  Up to max_rotated_files_ backups are kept.
 */

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace edgeai {

enum class LogLevel : uint8_t {
    TRACE = 0,
    DEBUG = 1,
    INFO  = 2,
    WARN  = 3,
    ERROR = 4,
    FATAL = 5
};

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    void set_level(LogLevel level) { level_ = level; }

    /// Configure file logging with optional rotation settings.
    /// @param path           Log file path
    /// @param max_bytes      Max file size before rotation (default 10 MB)
    /// @param max_backups    Number of rotated files to keep (default 5)
    void set_file(const std::string& path,
                  size_t max_bytes = 10 * 1024 * 1024,
                  int max_backups = 5) {
        std::lock_guard<std::mutex> lock(mutex_);
        file_path_ = path;
        max_file_size_ = max_bytes;
        max_rotated_files_ = max_backups;
        if (file_.is_open()) file_.close();
        file_.open(path, std::ios::app);
    }

    void log(LogLevel level, const std::string& component, const std::string& message,
             const char* file = __builtin_FILE(), int line = __builtin_LINE()) {
        if (level < level_) return;

        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        // Thread-safe time formatting (localtime_r instead of localtime)
        std::tm tm{};
        localtime_r(&time, &tm);

        // Get thread ID
        auto tid = std::this_thread::get_id();
        std::ostringstream tid_oss;
        tid_oss << tid;

        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S")
            << '.' << std::setfill('0') << std::setw(3) << ms.count()
            << " [" << level_str(level) << "] "
            << "[T:" << tid_oss.str() << "] "
            << "[" << component << "] "
            << message;

        if (level >= LogLevel::WARN) {
            oss << " (" << file << ":" << line << ")";
        }

        std::string formatted = oss.str();

        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << formatted << std::endl;
        if (file_.is_open()) {
            file_ << formatted << std::endl;
            // Check if rotation is needed
            if (max_file_size_ > 0 && file_size() >= max_file_size_) {
                rotate_files();
            }
        }
    }

private:
    Logger() = default;
    ~Logger() { if (file_.is_open()) file_.close(); }
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    static const char* level_str(LogLevel level) {
        switch (level) {
            case LogLevel::TRACE: return "TRACE";
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return "INFO ";
            case LogLevel::WARN:  return "WARN ";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::FATAL: return "FATAL";
            default:              return "?????";
        }
    }

    /// Get current log file size (must hold mutex_)
    size_t file_size() {
        if (!file_.is_open()) return 0;
        return static_cast<size_t>(file_.tellp());
    }

    /// Rotate log files: current → .1.log, .1 → .2, etc. (must hold mutex_)
    void rotate_files() {
        namespace fs = std::filesystem;

        file_.close();

        // Shift existing backups: .4→.5(delete), .3→.4, .2→.3, .1→.2
        for (int i = max_rotated_files_; i >= 1; --i) {
            std::string src = file_path_ + "." + std::to_string(i);
            if (i == max_rotated_files_) {
                // Delete oldest backup
                std::error_code ec;
                fs::remove(src, ec);
            } else {
                std::string dst = file_path_ + "." + std::to_string(i + 1);
                std::error_code ec;
                fs::rename(src, dst, ec);
            }
        }

        // Rotate current file → .1
        {
            std::error_code ec;
            fs::rename(file_path_, file_path_ + ".1", ec);
        }

        // Reopen fresh file
        file_.open(file_path_, std::ios::out | std::ios::trunc);
    }

    LogLevel      level_ = LogLevel::INFO;
    std::mutex    mutex_;
    std::ofstream file_;
    std::string   file_path_;
    size_t        max_file_size_ = 10 * 1024 * 1024;  // 10 MB default
    int           max_rotated_files_ = 5;
};

// ── Macros ─────────────────────────────────────────────────────

#define LOG_TRACE(comp, msg) edgeai::Logger::instance().log(edgeai::LogLevel::TRACE, comp, msg)
#define LOG_DEBUG(comp, msg) edgeai::Logger::instance().log(edgeai::LogLevel::DEBUG, comp, msg)
#define LOG_INFO(comp, msg)  edgeai::Logger::instance().log(edgeai::LogLevel::INFO,  comp, msg)
#define LOG_WARN(comp, msg)  edgeai::Logger::instance().log(edgeai::LogLevel::WARN,  comp, msg)
#define LOG_ERROR(comp, msg) edgeai::Logger::instance().log(edgeai::LogLevel::ERROR, comp, msg)
#define LOG_FATAL(comp, msg) edgeai::Logger::instance().log(edgeai::LogLevel::FATAL, comp, msg)

}  // namespace edgeai
