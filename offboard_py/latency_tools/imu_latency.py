#!/usr/bin/env python3
"""C1: IMU-to-policy latency measurement.

Subscribes to the IMU topic at the policy node location and computes
per-message latency as (receive_time - header.stamp). Outputs CSV with
per-message latency and prints summary statistics.

Usage:
    ros2 run offboard_py imu_latency --ros-args \
        -p imu_topic:=/px4_1/mavros/imu/data \
        -p num_samples:=2000 \
        -p output_dir:=/tmp/latency
"""
from __future__ import annotations

import csv
import os
import statistics
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Imu

QOS_BEST_EFFORT = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class ImuLatencyNode(Node):
    """Measures IMU message latency: receive_time - header.stamp."""

    def __init__(self):
        super().__init__("imu_latency")

        self.declare_parameter("imu_topic", "/mavros/imu/data")
        self.declare_parameter("num_samples", 2000)
        self.declare_parameter("output_dir", "/tmp/latency")

        self._topic = self.get_parameter("imu_topic").value
        self._num_samples = self.get_parameter("num_samples").value
        self._output_dir = self.get_parameter("output_dir").value

        self._latencies: List[float] = []
        self._stamps: List[float] = []

        os.makedirs(self._output_dir, exist_ok=True)

        self._sub = self.create_subscription(
            Imu, self._topic, self._imu_cb, QOS_BEST_EFFORT
        )
        self.get_logger().info(
            f"Measuring IMU latency on '{self._topic}', "
            f"collecting {self._num_samples} samples..."
        )

    def _imu_cb(self, msg: Imu):
        now = self.get_clock().now()
        header_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        receive_ns = now.nanoseconds
        latency_ms = (receive_ns - header_ns) / 1e6

        self._latencies.append(latency_ms)
        self._stamps.append(receive_ns / 1e9)

        if len(self._latencies) % 500 == 0:
            self.get_logger().info(
                f"  collected {len(self._latencies)}/{self._num_samples}"
            )

        if len(self._latencies) >= self._num_samples:
            self._finish()

    def _finish(self):
        self.destroy_subscription(self._sub)
        csv_path = os.path.join(self._output_dir, "c1_imu_latency.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_s", "latency_ms"])
            for t, lat in zip(self._stamps, self._latencies):
                writer.writerow([f"{t:.6f}", f"{lat:.4f}"])

        self._print_summary()
        self.get_logger().info(f"CSV saved to {csv_path}")
        raise SystemExit(0)

    def _print_summary(self):
        data = self._latencies
        data_sorted = sorted(data)
        n = len(data)
        p50 = data_sorted[int(n * 0.50)]
        p95 = data_sorted[int(n * 0.95)]
        p99 = data_sorted[min(int(n * 0.99), n - 1)]

        self.get_logger().info("=" * 60)
        self.get_logger().info("C1: IMU-to-Policy Latency Summary")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  Samples : {n}")
        self.get_logger().info(f"  Mean    : {statistics.mean(data):.3f} ms")
        self.get_logger().info(f"  Std     : {statistics.stdev(data):.3f} ms")
        self.get_logger().info(f"  Min     : {min(data):.3f} ms")
        self.get_logger().info(f"  Max     : {max(data):.3f} ms")
        self.get_logger().info(f"  p50     : {p50:.3f} ms")
        self.get_logger().info(f"  p95     : {p95:.3f} ms")
        self.get_logger().info(f"  p99     : {p99:.3f} ms")
        self.get_logger().info("=" * 60)


def main(args=None):
    rclpy.init(args=args)
    node = ImuLatencyNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
