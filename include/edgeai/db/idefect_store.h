#pragma once

/**
 * @file idefect_store.h
 * @brief Abstract storage interface — enables swapping persistence backends.
 *
 * Implement this interface for any storage backend:
 *   - SQLite (DefectDatabase)
 *   - PostgreSQL / TimescaleDB
 *   - InfluxDB (time-series)
 *   - In-memory (for testing)
 *   - CSV/file-based logging
 */

#include "edgeai/common/types.h"

namespace edgeai {

class IDefectStore {
public:
    virtual ~IDefectStore() = default;

    /// Open/create the storage backend
    virtual bool open() = 0;

    /// Close the storage backend
    virtual void close() = 0;

    /// Insert an inspection result with all its detections
    virtual bool insert_result(const InspectionResult& result) = 0;

    /// Prune old records to keep at most max_records.
    /// Returns number of records deleted. Pass 0 to skip pruning.
    virtual int64_t prune(int64_t max_records) = 0;
};

}  // namespace edgeai
