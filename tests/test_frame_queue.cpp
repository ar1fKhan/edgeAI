/**
 * @file test_frame_queue.cpp
 * @brief Unit tests for the bounded producer-consumer queue.
 */

#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include "edgeai/pipeline/frame_queue.h"

using namespace edgeai;

TEST(FrameQueueTest, PushAndPop) {
    BoundedQueue<int> queue(10);

    EXPECT_TRUE(queue.push(42));
    auto val = queue.pop();
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, 42);
}

TEST(FrameQueueTest, EmptyPopReturnsNullopt) {
    BoundedQueue<int> queue(10);
    auto val = queue.try_pop();
    EXPECT_FALSE(val.has_value());
}

TEST(FrameQueueTest, FIFO_Order) {
    BoundedQueue<int> queue(10);
    queue.push(1);
    queue.push(2);
    queue.push(3);

    EXPECT_EQ(*queue.pop(), 1);
    EXPECT_EQ(*queue.pop(), 2);
    EXPECT_EQ(*queue.pop(), 3);
}

TEST(FrameQueueTest, CapacityRespected) {
    BoundedQueue<int> queue(2);
    EXPECT_TRUE(queue.push(1));
    EXPECT_TRUE(queue.push(2));
    // Queue is full — push with short timeout should fail
    EXPECT_FALSE(queue.push(3, std::chrono::milliseconds(10)));
}

TEST(FrameQueueTest, StopUnblocksPop) {
    BoundedQueue<int> queue(10);

    std::thread consumer([&queue]() {
        auto val = queue.pop(std::chrono::milliseconds(5000));
        EXPECT_FALSE(val.has_value());  // stopped
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    queue.stop();
    consumer.join();
}

TEST(FrameQueueTest, ProducerConsumer) {
    BoundedQueue<int> queue(8);
    std::atomic<int> sum{0};
    const int N = 100;

    std::thread producer([&]() {
        for (int i = 1; i <= N; ++i) {
            queue.push(i, std::chrono::milliseconds(500));
        }
        queue.stop();
    });

    std::thread consumer([&]() {
        while (true) {
            auto val = queue.pop(std::chrono::milliseconds(500));
            if (!val.has_value()) break;
            sum += *val;
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(sum.load(), N * (N + 1) / 2);  // sum of 1..N
}

TEST(FrameQueueTest, SizeTracking) {
    BoundedQueue<int> queue(10);
    EXPECT_EQ(queue.size(), 0u);
    EXPECT_TRUE(queue.empty());

    queue.push(1);
    queue.push(2);
    EXPECT_EQ(queue.size(), 2u);
    EXPECT_FALSE(queue.empty());

    queue.pop();
    EXPECT_EQ(queue.size(), 1u);
}

TEST(FrameQueueTest, Reset) {
    BoundedQueue<int> queue(10);
    queue.push(1);
    queue.push(2);
    queue.stop();

    queue.reset();
    EXPECT_TRUE(queue.empty());
    EXPECT_FALSE(queue.is_stopped());

    // Should work again after reset
    EXPECT_TRUE(queue.push(99));
    auto val = queue.pop();
    EXPECT_EQ(*val, 99);
}
