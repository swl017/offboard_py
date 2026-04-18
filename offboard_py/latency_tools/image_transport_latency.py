#!/usr/bin/env python3
"""C4: Raw image transport latency measurement.

Measures camera-to-ROS2 pipeline latency by comparing the ROS2 message
receive time to the header stamp. This captures driver + middleware latency.

For ground-truth end-to-end measurement, point the camera at a display showing
a millisecond timer and compare visually (documented in output).

Usage:
    ros2 run offboard_py image_transport_latency --ros-args \
        -p image_topic:=/camera/image_raw \
        -p num_samples:=500 \
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
from sensor_msgs.msg import Image

QOS_BEST_EFFORT = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class ImageTransportLatencyNode(Node):
    """Measures image transport latency: receive_time - header.stamp.

    Measurement path: sensor → driver → DDS middleware → this subscriber.
    This does NOT include sensor exposure time or any post-processing.
    """

    def __init__(self):
        super().__init__("image_transport_latency")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("num_samples", 500)
        self.declare_parameter("output_dir", "/tmp/latency")

        self._topic = self.get_parameter("image_topic").value
        self._num_samples = self.get_parameter("num_samples").value
        self._output_dir = self.get_parameter("output_dir").value

        self._latencies: List[float] = []
        self._stamps: List[float] = []
        self._sizes: List[int] = []

        os.makedirs(self._output_dir, exist_ok=True)

        self._sub = self.create_subscription(
            Image, self._topic, self._image_cb, QOS_BEST_EFFORT
        )
        self.get_logger().info(
            f"Measuring image transport latency on '{self._topic}', "
            f"collecting {self._num_samples} samples..."
        )
        self.get_logger().info(
            "Measurement path: sensor → driver → DDS → this subscriber"
        )

    def _image_cb(self, msg: Image):
        now = self.get_clock().now()
        header_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        receive_ns = now.nanoseconds
        latency_ms = (receive_ns - header_ns) / 1e6

        self._latencies.append(latency_ms)
        self._stamps.append(receive_ns / 1e9)
        self._sizes.append(len(msg.data))

        if len(self._latencies) % 100 == 0:
            self.get_logger().info(
                f"  collected {len(self._latencies)}/{self._num_samples}"
            )

        if len(self._latencies) >= self._num_samples:
            self._finish()

    def _finish(self):
        self.destroy_subscription(self._sub)
        csv_path = os.path.join(self._output_dir, "c4_image_transport_latency.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_s", "latency_ms", "image_size_bytes"])
            for t, lat, sz in zip(self._stamps, self._latencies, self._sizes):
                writer.writerow([f"{t:.6f}", f"{lat:.4f}", sz])

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

        avg_size_kb = statistics.mean(self._sizes) / 1024

        self.get_logger().info("=" * 60)
        self.get_logger().info("C4: Image Transport Latency Summary")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  Topic        : {self._topic}")
        self.get_logger().info(
            f"  Measurement  : receive_time - header.stamp"
        )
        self.get_logger().info(
            f"  Path         : sensor → driver → DDS → subscriber"
        )
        self.get_logger().info(f"  Samples      : {n}")
        self.get_logger().info(f"  Avg img size : {avg_size_kb:.1f} KB")
        self.get_logger().info(f"  Mean         : {statistics.mean(data):.3f} ms")
        self.get_logger().info(f"  Std          : {statistics.stdev(data):.3f} ms")
        self.get_logger().info(f"  Min          : {min(data):.3f} ms")
        self.get_logger().info(f"  Max          : {max(data):.3f} ms")
        self.get_logger().info(f"  p50          : {p50:.3f} ms")
        self.get_logger().info(f"  p95          : {p95:.3f} ms")
        self.get_logger().info(f"  p99          : {p99:.3f} ms")
        self.get_logger().info("=" * 60)


def main(args=None):
    rclpy.init(args=args)
    node = ImageTransportLatencyNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
