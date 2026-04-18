#!/usr/bin/env python3
"""Velocity step response flight test (A2).

Executes a configurable velocity step sequence on all specified axes:
    For each axis, for each speed:
        0 → +v (hold) → 0 (settle) → -v (hold) → 0 (settle)

Sequence is defined in a YAML config file (default: config/velocity_step.yaml).

Usage:
    ros2 run offboard_py flight_test_velocity_step --ros-args \
        -p vehicle_ns:=px4_1 \
        -p config_file:=/path/to/velocity_step.yaml \
        -p output_dir:=/tmp/flight_test

Flight procedure:
    1. [Pilot] Position mode → Takeoff → Hover at test altitude
    2. [Pilot] Switch to Offboard mode via RC
    3. [Script] Detects offboard → starts bag → settles → runs steps
    4. [Script] Completes → stops bag → publishes zero velocity
    5. [Pilot] Switch to Position mode → Land
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
import yaml

from offboard_py.flight_test_harness import FlightTestHarness


@dataclass
class VelocityStep:
    """A single step in the velocity sequence."""
    axis: str       # "x", "y", or "z"
    speed: float    # m/s (signed, 0.0 for settle)
    duration: float  # seconds


class FlightTestVelocityStep(FlightTestHarness):
    """Velocity step response test using the flight test harness."""

    def __init__(self) -> None:
        # Load config before super().__init__ so we can set test_name
        # We need to peek at the config_file parameter
        # Use a temporary node context to read parameters, then pass to base
        super().__init__("flight_test_velocity_step", "velocity_step")

        # Load and expand sequence
        config_path = self.get_parameter("config_file").value
        if config_path:
            self._load_config(config_path)
        else:
            # Use defaults from parameters already declared by base class
            self._axes = ["x", "y", "z"]
            self._speeds = [3.0, 5.0]
            self._hold_duration = 5.0

        self._sequence = self._expand_sequence()
        self._step_idx = 0
        self._step_start_wallclock: Optional[float] = None
        self._z_axis_skipped = False

        self.get_logger().info(
            f"Velocity step test: axes={self._axes}, speeds={self._speeds}, "
            f"hold={self._hold_duration}s, {len(self._sequence)} total steps"
        )

    def _load_config(self, config_path: str) -> None:
        """Load test configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to load config {config_path}: {e}")
            self.get_logger().info("Using default configuration")
            self._axes = ["x", "y", "z"]
            self._speeds = [3.0, 5.0]
            self._hold_duration = 5.0
            return

        self._axes = cfg.get("axes", ["x", "y", "z"])
        self._speeds = cfg.get("speeds", [3.0, 5.0])
        self._hold_duration = cfg.get("hold_duration", 5.0)

        # Override base class parameters if specified in YAML
        if "settle_duration" in cfg:
            self._settle_duration = cfg["settle_duration"]
        if "altitude_margin" in cfg:
            self._altitude_margin = cfg["altitude_margin"]
        if "drift_limit" in cfg:
            self._drift_limit = cfg["drift_limit"]
        if "drift_return_threshold" in cfg:
            self._drift_return_threshold = cfg["drift_return_threshold"]
        if "drift_kp" in cfg:
            self._drift_kp = cfg["drift_kp"]
        if "drift_vel_clamp" in cfg:
            self._drift_vel_clamp = cfg["drift_vel_clamp"]

        self.get_logger().info(f"Loaded config from {config_path}")

    def _expand_sequence(self) -> List[VelocityStep]:
        """Expand axes × speeds into the full step sequence.

        For each axis, for each speed:
            hold at +speed → settle at 0 → hold at -speed → settle at 0

        The harness inserts an initial settle before the first TEST,
        so we don't need a leading settle here.
        """
        steps: List[VelocityStep] = []
        for axis in self._axes:
            for speed in self._speeds:
                # +v hold
                steps.append(VelocityStep(axis, speed, self._hold_duration))
                # settle at 0
                steps.append(VelocityStep(axis, 0.0, self._settle_duration))
                # -v hold
                steps.append(VelocityStep(axis, -speed, self._hold_duration))
                # settle at 0
                steps.append(VelocityStep(axis, 0.0, self._settle_duration))
        return steps

    def _axis_to_velocity(self, axis: str, speed: float) -> Tuple[float, float, float, float]:
        """Convert axis name + speed to (vx, vy, vz, yaw_rate)."""
        if axis == "x":
            return (speed, 0.0, 0.0, 0.0)
        elif axis == "y":
            return (0.0, speed, 0.0, 0.0)
        elif axis == "z":
            return (0.0, 0.0, speed, 0.0)
        else:
            self.get_logger().error(f"Unknown axis: {axis}")
            return (0.0, 0.0, 0.0, 0.0)

    # -------------------------------------------------------------------
    # Harness hooks
    # -------------------------------------------------------------------

    def _on_offboard_entered(self) -> None:
        """Check altitude margin for z-axis steps, log sequence summary."""
        # Find max z speed in sequence
        max_vz = 0.0
        for step in self._sequence:
            if step.axis == "z":
                max_vz = max(max_vz, abs(step.speed))

        if max_vz > 0.0:
            if not self._check_altitude_margin(max_vz):
                # Remove z-axis steps
                self._sequence = [s for s in self._sequence if s.axis != "z"]
                self._z_axis_skipped = True
                self.get_logger().warn(
                    f"Z-axis steps removed. Remaining: {len(self._sequence)} steps"
                )

        # Log sequence summary
        active_steps = [s for s in self._sequence if s.speed != 0.0]
        self.get_logger().info(
            f"Test sequence: {len(active_steps)} velocity steps, "
            f"{len(self._sequence)} total steps (incl. settles)"
        )

    def _on_offboard_lost(self) -> None:
        """Log abort."""
        self.get_logger().warn(
            f"Offboard lost at step {self._step_idx}/{len(self._sequence)}"
        )

    def _get_current_command(self) -> Tuple[float, float, float, float]:
        """Return velocity command for current step."""
        if self._step_idx >= len(self._sequence):
            return (0.0, 0.0, 0.0, 0.0)

        step = self._sequence[self._step_idx]

        # Initialize step timer on first call
        if self._step_start_wallclock is None:
            self._step_start_wallclock = self.get_clock().now().nanoseconds * 1e-9
            if step.speed != 0.0:
                self.get_logger().info(
                    f"Step {self._step_idx + 1}/{len(self._sequence)}: "
                    f"{step.axis} {step.speed:+.1f} m/s for {step.duration}s"
                )

        now = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now - self._step_start_wallclock

        if elapsed >= step.duration:
            # Advance to next step
            self._step_idx += 1
            self._step_start_wallclock = None
            # Recurse to get the new step's command (or zero if done)
            if self._step_idx < len(self._sequence):
                return self._get_current_command()
            else:
                return (0.0, 0.0, 0.0, 0.0)

        return self._axis_to_velocity(step.axis, step.speed)

    def _is_test_complete(self) -> bool:
        """True when all steps have been executed."""
        return self._step_idx >= len(self._sequence)

    def _on_test_complete(self) -> None:
        """Save CSV with test name."""
        suffix = ""
        if self._z_axis_skipped:
            suffix = "_no_z"
        self._save_csv(f"{self._test_name}{suffix}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = FlightTestVelocityStep()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down flight_test_velocity_step.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
