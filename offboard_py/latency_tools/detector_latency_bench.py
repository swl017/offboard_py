#!/usr/bin/env python3
"""C2: Detection inference latency under realistic load.

Feeds images from a directory to the detector input topic at a configurable
FPS, then measures per-frame latency from publish_time to detection_result_time.

Usage:
    ros2 run offboard_py detector_latency_bench --ros-args \
        -p image_dir:=/path/to/test_images \
        -p publish_topic:=/detector/input \
        -p result_topic:=/detector/result \
        -p fps:=10.0 \
        -p output_dir:=/tmp/latency
"""
from __future__ import annotations

import csv
import os
import statistics
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Header

QOS_RELIABLE = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)

QOS_BEST_EFFORT = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)


class DetectorLatencyBench(Node):
    """Benchmarks detector inference latency by publishing images and timing results."""

    def __init__(self):
        super().__init__("detector_latency_bench")

        self.declare_parameter("image_dir", "")
        self.declare_parameter("publish_topic", "/detector/input")
        self.declare_parameter("result_topic", "/detector/result")
        self.declare_parameter("fps", 10.0)
        self.declare_parameter("output_dir", "/tmp/latency")

        image_dir = self.get_parameter("image_dir").value
        pub_topic = self.get_parameter("publish_topic").value
        res_topic = self.get_parameter("result_topic").value
        self._fps = self.get_parameter("fps").value
        self._output_dir = self.get_parameter("output_dir").value

        os.makedirs(self._output_dir, exist_ok=True)

        # Load images
        self._images = self._load_images(image_dir)
        if not self._images:
            self.get_logger().error(f"No images found in '{image_dir}'")
            raise SystemExit(1)

        self.get_logger().info(f"Loaded {len(self._images)} images from '{image_dir}'")

        self._bridge = CvBridge()
        self._pub = self.create_publisher(Image, pub_topic, QOS_RELIABLE)
        self._sub = self.create_subscription(
            Image, res_topic, self._result_cb, QOS_BEST_EFFORT
        )

        # Tracking: seq -> publish_time_ns
        self._seq = 0
        self._pending: Dict[int, int] = {}  # seq -> publish_time_ns
        self._latencies: List[float] = []
        self._publish_times: List[float] = []
        self._total_published = 0

        period = 1.0 / self._fps
        self._timer = self.create_timer(period, self._publish_next)

        self.get_logger().info(
            f"Publishing to '{pub_topic}' at {self._fps} FPS, "
            f"listening for results on '{res_topic}'"
        )

    def _load_images(self, image_dir: str) -> List[np.ndarray]:
        if not image_dir or not os.path.isdir(image_dir):
            return []
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = sorted(
            p
            for p in os.listdir(image_dir)
            if os.path.splitext(p)[1].lower() in exts
        )
        images = []
        for p in paths:
            img = cv2.imread(os.path.join(image_dir, p))
            if img is not None:
                images.append(img)
        return images

    def _publish_next(self):
        if self._seq >= len(self._images):
            # All images published — wait a bit for remaining results, then finish
            self._timer.cancel()
            self.create_timer(2.0, self._finish_once)
            return

        img = self._images[self._seq]
        msg = self._bridge.cv2_to_imgmsg(img, encoding="bgr8")

        # Encode sequence number in header stamp for tracking
        now_ns = self.get_clock().now().nanoseconds
        msg.header.stamp.sec = self._seq
        msg.header.stamp.nanosec = 0

        self._pending[self._seq] = now_ns
        self._publish_times.append(now_ns / 1e9)
        self._pub.publish(msg)
        self._total_published += 1
        self._seq += 1

    def _result_cb(self, msg: Image):
        seq = msg.header.stamp.sec
        if seq in self._pending:
            receive_ns = self.get_clock().now().nanoseconds
            publish_ns = self._pending.pop(seq)
            latency_ms = (receive_ns - publish_ns) / 1e6
            self._latencies.append(latency_ms)

    _finished = False

    def _finish_once(self):
        if self._finished:
            return
        self._finished = True
        self._finish()

    def _finish(self):
        csv_path = os.path.join(self._output_dir, "c2_detector_latency.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "publish_time_s", "latency_ms"])
            for i, lat in enumerate(self._latencies):
                t = self._publish_times[i] if i < len(self._publish_times) else 0
                writer.writerow([i, f"{t:.6f}", f"{lat:.4f}"])

        self._print_summary()
        self.get_logger().info(f"CSV saved to {csv_path}")
        raise SystemExit(0)

    def _print_summary(self):
        received = len(self._latencies)
        sent = self._total_published
        dropped = sent - received

        self.get_logger().info("=" * 60)
        self.get_logger().info("C2: Detector Inference Latency Summary")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  Published : {sent} frames at {self._fps} FPS")
        self.get_logger().info(f"  Received  : {received}")
        self.get_logger().info(f"  Dropped   : {dropped} ({100*dropped/max(sent,1):.1f}%)")

        if received > 1:
            data = self._latencies
            data_sorted = sorted(data)
            n = len(data)
            p50 = data_sorted[int(n * 0.50)]
            p95 = data_sorted[int(n * 0.95)]
            p99 = data_sorted[min(int(n * 0.99), n - 1)]

            self.get_logger().info(f"  Mean      : {statistics.mean(data):.3f} ms")
            self.get_logger().info(f"  Std       : {statistics.stdev(data):.3f} ms")
            self.get_logger().info(f"  Min       : {min(data):.3f} ms")
            self.get_logger().info(f"  Max       : {max(data):.3f} ms")
            self.get_logger().info(f"  p50       : {p50:.3f} ms")
            self.get_logger().info(f"  p95       : {p95:.3f} ms")
            self.get_logger().info(f"  p99       : {p99:.3f} ms")
        elif received == 1:
            self.get_logger().info(f"  Latency   : {self._latencies[0]:.3f} ms")
        else:
            self.get_logger().warn("  No results received!")
        self.get_logger().info("=" * 60)


def main(args=None):
    rclpy.init(args=args)
    node = DetectorLatencyBench()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
