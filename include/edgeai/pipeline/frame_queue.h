#pragma once

/**
 * @file frame_queue.h
 * @brief Lock-free bounded SPSC queue for producer-consumer frame pipeline.
 * 
 * Uses condition variables for blocking push/pop with timeout.
 * Designed for single-producer (camera) single-consumer (inference) pattern.
 */

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <chrono>
#include "edgeai/common/types.h"

namespace edgeai {

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity) : capacity_(capacity) {}

    /// Push item. Blocks if queue is full until space is available or timeout.
    /// Returns false if queue was stopped or timeout expired.
    bool push(T item, std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_push_.wait_for(lock, timeout, [this] {
            return queue_.size() < capacity_ || stopped_;
        })) {
            return false;  // timeout
        }
        if (stopped_) return false;
        queue_.push(std::move(item));
        cv_pop_.notify_one();
        return true;
    }

    /// Pop item. Blocks until item is available or timeout.
    /// Returns std::nullopt if queue was stopped or timeout expired.
    std::optional<T> pop(std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_pop_.wait_for(lock, timeout, [this] {
            return !queue_.empty() || stopped_;
        })) {
            return std::nullopt;  // timeout
        }
        if (stopped_ && queue_.empty()) return std::nullopt;
        T item = std::move(queue_.front());
        queue_.pop();
        cv_push_.notify_one();
        return item;
    }

    /// Try pop without blocking
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return std::nullopt;
        T item = std::move(queue_.front());
        queue_.pop();
        cv_push_.notify_one();
        return item;
    }

    /// Signal all waiting threads to stop
    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        cv_push_.notify_all();
        cv_pop_.notify_all();
    }

    /// Reset the queue for reuse
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty;
        queue_.swap(empty);
        stopped_ = false;
    }

    [[nodiscard]] size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    [[nodiscard]] bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    [[nodiscard]] bool is_stopped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stopped_;
    }

private:
    size_t                  capacity_;
    std::queue<T>           queue_;
    mutable std::mutex      mutex_;
    std::condition_variable cv_push_;
    std::condition_variable cv_pop_;
    bool                    stopped_ = false;
};

// Type alias for frame queue
using FrameQueue = BoundedQueue<Frame>;

}  // namespace edgeai
