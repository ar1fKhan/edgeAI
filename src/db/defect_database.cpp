/**
 * @file defect_database.cpp
 * @brief SQLite database implementation for defect logging.
 */

#include "edgeai/db/defect_database.h"
#include "edgeai/common/logger.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace edgeai {

DefectDatabase::DefectDatabase(const std::string& db_path)
    : db_path_(db_path) {}

DefectDatabase::~DefectDatabase() {
    close();
}

bool DefectDatabase::open() {
    std::lock_guard<std::mutex> lock(mutex_);

    int rc = sqlite3_open(db_path_.c_str(), &db_);
    if (rc != SQLITE_OK) {
        LOG_ERROR("Database", "Failed to open database: " + std::string(sqlite3_errmsg(db_)));
        return false;
    }

    // Enable WAL mode for better concurrent performance
    execute("PRAGMA journal_mode=WAL;");
    execute("PRAGMA synchronous=NORMAL;");
    execute("PRAGMA cache_size=10000;");

    if (!create_tables()) {
        LOG_ERROR("Database", "Failed to create tables");
        return false;
    }

    LOG_INFO("Database", "Database opened: " + db_path_);
    return true;
}

void DefectDatabase::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
        LOG_INFO("Database", "Database closed");
    }
}

bool DefectDatabase::create_tables() {
    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS inspections (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id        INTEGER NOT NULL,
            timestamp       TEXT NOT NULL,
            verdict         TEXT NOT NULL,
            num_detections  INTEGER DEFAULT 0,
            inference_ms    REAL DEFAULT 0.0,
            total_ms        REAL DEFAULT 0.0,
            image_path      TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS detections (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id   INTEGER NOT NULL,
            defect_type     TEXT NOT NULL,
            confidence      REAL NOT NULL,
            bbox_x          REAL NOT NULL,
            bbox_y          REAL NOT NULL,
            bbox_w          REAL NOT NULL,
            bbox_h          REAL NOT NULL,
            FOREIGN KEY(inspection_id) REFERENCES inspections(id)
        );

        CREATE INDEX IF NOT EXISTS idx_inspections_timestamp 
            ON inspections(timestamp);
        CREATE INDEX IF NOT EXISTS idx_inspections_verdict 
            ON inspections(verdict);
        CREATE INDEX IF NOT EXISTS idx_detections_type 
            ON detections(defect_type);
    )";

    return execute(sql);
}

bool DefectDatabase::insert_result(const InspectionResult& result) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!db_) return false;

    if (!execute("BEGIN;")) return false;

    std::string ts = current_timestamp();
    double inf_ms = std::chrono::duration<double, std::milli>(result.inference_time).count();
    double tot_ms = std::chrono::duration<double, std::milli>(result.total_time).count();

    const char* ins_sql = "INSERT INTO inspections "
        "(frame_id, timestamp, verdict, num_detections, inference_ms, total_ms, image_path) "
        "VALUES (?, ?, ?, ?, ?, ?, ?);";

    sqlite3_stmt* ins_stmt = nullptr;
    if (sqlite3_prepare_v2(db_, ins_sql, -1, &ins_stmt, nullptr) != SQLITE_OK) {
        LOG_ERROR("Database", "Failed to prepare insert: " + std::string(sqlite3_errmsg(db_)));
        execute("ROLLBACK;");
        return false;
    }

    std::string verdict_str = verdict_to_string(result.verdict);
    sqlite3_bind_int64(ins_stmt, 1, static_cast<int64_t>(result.frame_id));
    sqlite3_bind_text(ins_stmt, 2, ts.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(ins_stmt, 3, verdict_str.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(ins_stmt, 4, static_cast<int>(result.detections.size()));
    sqlite3_bind_double(ins_stmt, 5, inf_ms);
    sqlite3_bind_double(ins_stmt, 6, tot_ms);
    sqlite3_bind_text(ins_stmt, 7, result.image_path.c_str(), -1, SQLITE_TRANSIENT);

    bool ok = (sqlite3_step(ins_stmt) == SQLITE_DONE);
    sqlite3_finalize(ins_stmt);
    if (!ok) {
        LOG_ERROR("Database", "Failed to insert inspection: " + std::string(sqlite3_errmsg(db_)));
        execute("ROLLBACK;");
        return false;
    }

    int64_t inspection_id = sqlite3_last_insert_rowid(db_);

    if (!result.detections.empty()) {
        const char* det_sql = "INSERT INTO detections "
            "(inspection_id, defect_type, confidence, bbox_x, bbox_y, bbox_w, bbox_h) "
            "VALUES (?, ?, ?, ?, ?, ?, ?);";

        sqlite3_stmt* det_stmt = nullptr;
        if (sqlite3_prepare_v2(db_, det_sql, -1, &det_stmt, nullptr) != SQLITE_OK) {
            LOG_ERROR("Database", "Failed to prepare detection insert: "
                      + std::string(sqlite3_errmsg(db_)));
            execute("ROLLBACK;");
            return false;
        }

        for (const auto& det : result.detections) {
            sqlite3_reset(det_stmt);
            std::string type_str = defect_type_to_string(det.type);
            sqlite3_bind_int64(det_stmt, 1, inspection_id);
            sqlite3_bind_text(det_stmt, 2, type_str.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_double(det_stmt, 3, det.confidence);
            sqlite3_bind_double(det_stmt, 4, det.bbox.x);
            sqlite3_bind_double(det_stmt, 5, det.bbox.y);
            sqlite3_bind_double(det_stmt, 6, det.bbox.width);
            sqlite3_bind_double(det_stmt, 7, det.bbox.height);
            if (sqlite3_step(det_stmt) != SQLITE_DONE) {
                LOG_ERROR("Database", "Failed to insert detection: "
                          + std::string(sqlite3_errmsg(db_)));
                sqlite3_finalize(det_stmt);
                execute("ROLLBACK;");
                return false;
            }
        }

        sqlite3_finalize(det_stmt);
    }

    execute("COMMIT;");
    return true;
}

int64_t DefectDatabase::prune(int64_t max_records) {
    if (max_records <= 0) return 0;

    std::lock_guard<std::mutex> lock(mutex_);
    if (!db_) return 0;

    // Count current records
    const char* count_sql = "SELECT COUNT(*) FROM inspections;";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, count_sql, -1, &stmt, nullptr) != SQLITE_OK) return 0;

    int64_t count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int64(stmt, 0);
    }
    sqlite3_finalize(stmt);

    if (count <= max_records) return 0;

    int64_t to_delete = count - max_records;

    // Delete oldest detections whose inspection will be removed
    std::string del_det = "DELETE FROM detections WHERE inspection_id IN "
        "(SELECT id FROM inspections ORDER BY id ASC LIMIT "
        + std::to_string(to_delete) + ");";
    execute(del_det);

    // Delete oldest inspections
    std::string del_ins = "DELETE FROM inspections WHERE id IN "
        "(SELECT id FROM inspections ORDER BY id ASC LIMIT "
        + std::to_string(to_delete) + ");";
    execute(del_ins);

    LOG_INFO("Database", "Pruned " + std::to_string(to_delete) + " old inspection records");
    return to_delete;
}

std::vector<DefectRecord> DefectDatabase::get_recent_defects(int limit) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<DefectRecord> records;
    if (!db_) return records;

    std::string sql = R"(
        SELECT i.id, i.frame_id, i.timestamp, d.defect_type, d.confidence,
               d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h, i.verdict,
               i.inference_ms, i.image_path
        FROM inspections i
        JOIN detections d ON d.inspection_id = i.id
        WHERE i.verdict = 'REJECT'
        ORDER BY i.timestamp DESC
        LIMIT )" + std::to_string(limit) + ";";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return records;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        DefectRecord rec;
        rec.id           = sqlite3_column_int64(stmt, 0);
        rec.frame_id     = sqlite3_column_int64(stmt, 1);
        rec.timestamp    = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        rec.defect_type  = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        rec.confidence   = static_cast<float>(sqlite3_column_double(stmt, 4));
        rec.bbox_x       = static_cast<float>(sqlite3_column_double(stmt, 5));
        rec.bbox_y       = static_cast<float>(sqlite3_column_double(stmt, 6));
        rec.bbox_w       = static_cast<float>(sqlite3_column_double(stmt, 7));
        rec.bbox_h       = static_cast<float>(sqlite3_column_double(stmt, 8));
        rec.verdict      = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 9));
        rec.inference_ms = sqlite3_column_double(stmt, 10);
        rec.image_path   = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 11));
        records.push_back(rec);
    }

    sqlite3_finalize(stmt);
    return records;
}

std::vector<DailyStats> DefectDatabase::get_daily_stats(int days) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<DailyStats> stats;
    if (!db_) return stats;

    std::string sql = R"(
        SELECT 
            DATE(timestamp) as date,
            COUNT(*) as total,
            SUM(CASE WHEN verdict = 'REJECT' THEN 1 ELSE 0 END) as defects,
            SUM(CASE WHEN verdict = 'PASS' THEN 1 ELSE 0 END) as passed,
            AVG(inference_ms) as avg_inference
        FROM inspections
        WHERE timestamp >= DATE('now', '-)" + std::to_string(days) + R"( days')
        GROUP BY DATE(timestamp)
        ORDER BY date DESC;
    )";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return stats;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        DailyStats ds;
        ds.date             = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        ds.total_inspected  = sqlite3_column_int64(stmt, 1);
        ds.total_defects    = sqlite3_column_int64(stmt, 2);
        ds.total_passed     = sqlite3_column_int64(stmt, 3);
        ds.avg_inference_ms = sqlite3_column_double(stmt, 4);
        ds.defect_rate = (ds.total_inspected > 0)
            ? (static_cast<double>(ds.total_defects) / ds.total_inspected) * 100.0 : 0.0;
        stats.push_back(ds);
    }

    sqlite3_finalize(stmt);
    return stats;
}

std::vector<std::pair<std::string, int64_t>> DefectDatabase::get_defect_distribution() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<std::string, int64_t>> dist;
    if (!db_) return dist;

    const char* sql = R"(
        SELECT defect_type, COUNT(*) as cnt
        FROM detections
        GROUP BY defect_type
        ORDER BY cnt DESC;
    )";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return dist;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        std::string type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        int64_t count = sqlite3_column_int64(stmt, 1);
        dist.emplace_back(type, count);
    }

    sqlite3_finalize(stmt);
    return dist;
}

int64_t DefectDatabase::total_records() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!db_) return 0;

    const char* sql = "SELECT COUNT(*) FROM inspections;";
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return 0;

    int64_t count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int64(stmt, 0);
    }
    sqlite3_finalize(stmt);
    return count;
}

double DefectDatabase::overall_defect_rate() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!db_) return 0.0;

    const char* sql = R"(
        SELECT 
            CAST(SUM(CASE WHEN verdict = 'REJECT' THEN 1 ELSE 0 END) AS REAL) / 
            CAST(COUNT(*) AS REAL) * 100.0
        FROM inspections;
    )";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return 0.0;

    double rate = 0.0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        rate = sqlite3_column_double(stmt, 0);
    }
    sqlite3_finalize(stmt);
    return rate;
}

bool DefectDatabase::execute(const std::string& sql) {
    char* err_msg = nullptr;
    int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::string error = err_msg ? err_msg : "unknown error";
        sqlite3_free(err_msg);
        LOG_ERROR("Database", "SQL error: " + error);
        return false;
    }
    return true;
}

std::string DefectDatabase::current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
    localtime_r(&time, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

}  // namespace edgeai
