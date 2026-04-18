#!/usr/bin/env python3
"""C5: Datalink latency and dropout measurement.

Two modes:
  --mode sender : publishes timestamped ping messages at configurable rate
  --mode receiver : receives pings, records latency and detects dropouts

Requires clock synchronization between machines (PTP/NTP) for one-way
latency accuracy. Without sync, one-way values are approximate — use the
round-trip estimate logged by the sender if a pong topic is configured.

Usage (sender on agent A):
    ros2 run offboard_py datalink_latency_ping --ros-args \
        -p mode:=sender \
        -p ping_topic:=/datalink/ping \
        -p rate_hz:=50.0 \
        -p num_pings:=5000

Usage (receiver on agent B):
    ros2 run offboard_py datalink_latency_ping --ros-args \
        -p mode:=receiver \
        -p ping_topic:=/datalink/ping \
        -p output_dir:=/tmp/latency \
        -p link_label:=wifi_5ghz
"""
from __future__ import annotations

import csv
import os
import statistics
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import String

QOS_BEST_EFFORT = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)


class PingSender(Node):
    """Publishes timestamped ping messages at a fixed rate."""

    def __init__(self):
        super().__init__("datalink_ping_sender")

        self.declare_parameter("ping_topic", "/datalink/ping")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("num_pings", 5000)

        topic = self.get_parameter("ping_topic").value
        rate = self.get_parameter("rate_hz").value
        self._num_pings = self.get_parameter("num_pings").value

        self._pub = self.create_publisher(String, topic, QOS_BEST_EFFORT)
        self._seq = 0

        self._timer = self.create_timer(1.0 / rate, self._send_ping)
        self.get_logger().info(
            f"Sending pings on '{topic}' at {rate} Hz, total {self._num_pings}"
        )

    def _send_ping(self):
        if self._seq >= self._num_pings:
            self._timer.cancel()
            self.get_logger().info(f"Done — sent {self._seq} pings")
            raise SystemExit(0)

        now_ns = self.get_clock().now().nanoseconds
        # Format: "seq,send_timestamp_ns"
        msg = String()
        msg.data = f"{self._seq},{now_ns}"
        self._pub.publish(msg)
        self._seq += 1


class PingReceiver(Node):
    """Receives pings, computes one-way latency and dropout statistics."""

    def __init__(self):
        super().__init__("datalink_ping_receiver")

        self.declare_parameter("ping_topic", "/datalink/ping")
        self.declare_parameter("output_dir", "/tmp/latency")
        self.declare_parameter("link_label", "unknown")
        self.declare_parameter("timeout_s", 10.0)

        topic = self.get_parameter("ping_topic").value
        self._output_dir = self.get_parameter("output_dir").value
        self._link_label = self.get_parameter("link_label").value
        self._timeout_s = self.get_parameter("timeout_s").value

        os.makedirs(self._output_dir, exist_ok=True)

        self._records: List[Tuple[int, int, int, float]] = []
        # (seq, send_ns, recv_ns, latency_ms)
        self._last_seq: Optional[int] = None
        self._dropped_seqs: List[int] = []
        self._burst_drops: List[int] = []  # lengths of consecutive drop bursts
        self._last_recv_time_ns: Optional[int] = None

        self._sub = self.create_subscription(
            String, topic, self._ping_cb, QOS_BEST_EFFORT
        )
        self.get_logger().info(
            f"Listening for pings on '{topic}', link='{self._link_label}'"
        )

        # Inactivity timer — finish after timeout_s with no messages
        self._watchdog = self.create_timer(self._timeout_s, self._check_timeout)

    def _ping_cb(self, msg: String):
        recv_ns = self.get_clock().now().nanoseconds
        self._last_recv_time_ns = recv_ns

        try:
            parts = msg.data.split(",")
            seq = int(parts[0])
            send_ns = int(parts[1])
        except (ValueError, IndexError):
            self.get_logger().warn(f"Malformed ping: {msg.data}")
            return

        latency_ms = (recv_ns - send_ns) / 1e6
        self._records.append((seq, send_ns, recv_ns, latency_ms))

        # Dropout detection
        if self._last_seq is not None:
            gap = seq - self._last_seq - 1
            if gap > 0:
                for s in range(self._last_seq + 1, seq):
                    self._dropped_seqs.append(s)
                self._burst_drops.append(gap)
        self._last_seq = seq

        # Reset watchdog
        self._watchdog.reset()

    def _check_timeout(self):
        if self._last_recv_time_ns is None and not self._records:
            return  # Still waiting for first message
        self.get_logger().info("Inactivity timeout — finishing...")
        self._finish()

    def _finish(self):
        self._watchdog.cancel()
        self.destroy_subscription(self._sub)

        # Write per-message CSV
        csv_path = os.path.join(
            self._output_dir, f"c5_datalink_latency_{self._link_label}.csv"
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seq", "send_time_ns", "recv_time_ns", "latency_ms"])
            for seq, send, recv, lat in self._records:
                writer.writerow([seq, send, recv, f"{lat:.4f}"])

        # Write dropout CSV
        dropout_path = os.path.join(
            self._output_dir, f"c5_datalink_dropout_{self._link_label}.csv"
        )
        with open(dropout_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dropped_seq"])
            for s in self._dropped_seqs:
                writer.writerow([s])

        self._print_summary()
        self.get_logger().info(f"CSV saved to {csv_path}")
        self.get_logger().info(f"Dropout CSV saved to {dropout_path}")
        raise SystemExit(0)

    def _print_summary(self):
        received = len(self._records)
        dropped = len(self._dropped_seqs)
        total = received + dropped

        self.get_logger().info("=" * 60)
        self.get_logger().info("C5: Datalink Latency & Dropout Summary")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  Link label : {self._link_label}")
        self.get_logger().info(f"  Received   : {received}")
        self.get_logger().info(f"  Dropped    : {dropped}")
        self.get_logger().info(
            f"  Loss rate  : {100*dropped/max(total,1):.2f}%"
        )

        if self._burst_drops:
            self.get_logger().info(f"  Burst drops: {len(self._burst_drops)} events")
            self.get_logger().info(
                f"    Max burst : {max(self._burst_drops)} consecutive"
            )
            self.get_logger().info(
                f"    Mean burst: {statistics.mean(self._burst_drops):.1f}"
            )

        if received > 1:
            lats = [r[3] for r in self._records]
            lats_sorted = sorted(lats)
            n = len(lats)
            p50 = lats_sorted[int(n * 0.50)]
            p95 = lats_sorted[int(n * 0.95)]
            p99 = lats_sorted[min(int(n * 0.99), n - 1)]

            self.get_logger().info(f"  Mean       : {statistics.mean(lats):.3f} ms")
            self.get_logger().info(f"  Std        : {statistics.stdev(lats):.3f} ms")
            self.get_logger().info(f"  Min        : {min(lats):.3f} ms")
            self.get_logger().info(f"  Max        : {max(lats):.3f} ms")
            self.get_logger().info(f"  p50        : {p50:.3f} ms")
            self.get_logger().info(f"  p95        : {p95:.3f} ms")
            self.get_logger().info(f"  p99        : {p99:.3f} ms")
        elif received == 1:
            self.get_logger().info(
                f"  Latency    : {self._records[0][3]:.3f} ms"
            )
        else:
            self.get_logger().warn("  No pings received!")

        self.get_logger().info("=" * 60)


def main(args=None):
    rclpy.init(args=args)

    # Peek at mode param to decide which node to create
    tmp_node = Node("_datalink_param_peek")
    tmp_node.declare_parameter("mode", "receiver")
    mode = tmp_node.get_parameter("mode").value
    tmp_node.destroy_node()

    if mode == "sender":
        node = PingSender()
    else:
        node = PingReceiver()

    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
