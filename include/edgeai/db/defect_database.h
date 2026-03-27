#pragma once

/**
 * @file defect_database.h
 * @brief SQLite-based local storage for defect records and analytics.
 * 
 * All data stays on-edge — no cloud dependency.
 * Provides query APIs for the local dashboard.
 */

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <sqlite3.h>
#include "edgeai/db/idefect_store.h"
#include "edgeai/common/types.h"

namespace edgeai {

struct DefectRecord {
    int64_t     id;
    uint64_t    frame_id;
    std::string timestamp;
    std::string defect_type;
    float       confidence;
    float       bbox_x, bbox_y, bbox_w, bbox_h;
    std::string verdict;
    double      inference_ms;
    std::string image_path;
};

struct DailyStats {
    std::string date;
    int64_t     total_inspected;
    int64_t     total_defects;
    int64_t     total_passed;
    double      defect_rate;
    double      avg_inference_ms;
};

class DefectDatabase : public IDefectStore {
public:
    explicit DefectDatabase(const std::string& db_path);
    ~DefectDatabase() override;

    // Non-copyable
    DefectDatabase(const DefectDatabase&) = delete;
    DefectDatabase& operator=(const DefectDatabase&) = delete;

    /// Open/create the database and initialize schema
    bool open() override;

    /// Close the database
    void close() override;

    /// Insert an inspection result with all its detections
    bool insert_result(const InspectionResult& result) override;

    /// Prune old records beyond max_records (oldest first).
    /// Also deletes associated detection rows.
    /// Returns number of inspection records deleted.
    int64_t prune(int64_t max_records) override;

    /// Query recent defect records
    [[nodiscard]] std::vector<DefectRecord> get_recent_defects(int limit = 100) const;

    /// Get daily statistics for the dashboard
    [[nodiscard]] std::vector<DailyStats> get_daily_stats(int days = 30) const;

    /// Get defect count by type
    [[nodiscard]] std::vector<std::pair<std::string, int64_t>> get_defect_distribution() const;

    /// Get total records count
    [[nodiscard]] int64_t total_records() const;

    /// Get overall defect rate
    [[nodiscard]] double overall_defect_rate() const;

private:
    std::string     db_path_;
    sqlite3*        db_ = nullptr;
    mutable std::mutex mutex_;

    bool create_tables();
    bool execute(const std::string& sql);
    static std::string current_timestamp();
};

}  // namespace edgeai
