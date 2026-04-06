#!/usr/bin/env python3
"""System identification node for iris_ma6 vs PX4 SITL comparison.

Runs step response tests on a single PX4 drone via MAVROS (all ENU),
records telemetry to CSV, for offline comparison against iris_ma6's
DroneController reference curves.

MAVROS-only interface — no fmu/* topics. MAVROS handles ENU↔NED.
Velocity-only offboard mode.

Control loop runs on a wallclock timer (matching mas_offboard pattern).
/clock is subscribed only for recording sim-time timestamps in CSVs.

Usage:
    ros2 run offboard_py sysid_node --ros-args \
        -p vehicle_ns:=px4_1 \
        -p output_dir:=/tmp/sysid
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

import rclpy
from geometry_msgs.msg import TwistStamped, Vector3
from mavros_msgs.msg import AttitudeTarget, State as MavrosState
from mavros_msgs.srv import CommandBool, SetMode
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu


# ---------------------------------------------------------------------------
# QoS profiles (matching mas_offboard)
# ---------------------------------------------------------------------------

QOS_BEST_EFFORT = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

QOS_RELIABLE = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

class TestPhase(Enum):
    INIT = auto()       # wait for MAVROS topics
    RAMP_UP = auto()    # stream setpoints before arming (PX4 requirement)
    ARM = auto()        # request OFFBOARD + arm, wait for confirmation
    TAKEOFF = auto()    # climb to target altitude
    SETTLE = auto()     # hover between tests
    TEST = auto()       # run active test
    DONE = auto()


@dataclass
class TestSpec:
    """Specification for a single sysid test."""

    name: str
    duration: float  # seconds
    description: str
    # command_fn(node, elapsed_time) — called each tick during TEST phase
    command_fn: Callable[["SysidNode", float], None] = field(repr=False, default=None)


def _cmd_hover(node: "SysidNode", _t: float) -> None:
    """Hold zero velocity."""
    node._publish_velocity(0.0, 0.0, 0.0, 0.0)


def _cmd_vel_step_5(node: "SysidNode", _t: float) -> None:
    """5 m/s forward (ENU +X)."""
    node._publish_velocity(5.0, 0.0, 0.0, 0.0)


def _cmd_vel_step_10(node: "SysidNode", _t: float) -> None:
    """10 m/s forward (ENU +X)."""
    node._publish_velocity(10.0, 0.0, 0.0, 0.0)


def _cmd_vel_impulse_recovery(node: "SysidNode", t: float) -> None:
    """5 m/s for 1s then zero — observe recovery."""
    if t < 1.0:
        node._publish_velocity(5.0, 0.0, 0.0, 0.0)
    else:
        node._publish_velocity(0.0, 0.0, 0.0, 0.0)


def _cmd_yaw_step(node: "SysidNode", t: float) -> None:
    """Yaw rate 0.5 rad/s for 2s, then zero."""
    if t < 2.0:
        node._publish_velocity(0.0, 0.0, 0.0, 0.5)
    else:
        node._publish_velocity(0.0, 0.0, 0.0, 0.0)


# -- Gimbal test commands --
# LOS rate convention: Vector3 x=azimuth_rate, y=elevation_rate (rad/s).
# Positive elevation = up (ENU standard).
# Gimbal RPY convention: ENU body-frame.

_GIMBAL_RATE_CMD = 1.0  # rad/s


def _cmd_gimbal_los_rate_step(node: "SysidNode", t: float) -> None:
    """Gimbal LOS rate step: pitch 0.5s → hold 0.5s → yaw 0.5s → hold 0.5s.

    Drone holds hover throughout. Short pulses to avoid hitting joint limits.
    Total duration: 2.0s.
    """
    node._publish_velocity(0.0, 0.0, 0.0, 0.0)

    if t < 0.5:
        node._publish_gimbal_rate(0.0, _GIMBAL_RATE_CMD)
    elif t < 1.0:
        node._publish_gimbal_rate(0.0, 0.0)
    elif t < 1.5:
        node._publish_gimbal_rate(_GIMBAL_RATE_CMD, 0.0)
    else:
        node._publish_gimbal_rate(0.0, 0.0)


def _cmd_gimbal_los_stabilization(node: "SysidNode", t: float) -> None:
    """Gimbal LOS stabilization: hold fixed LOS target while shaking drone.

    Sequence:
      0-2s:  Yaw drone (0.5 rad/s) to establish off-axis gimbal state
      2-3s:  Settle (hover)
      3-4s:  Pulse +X velocity (5 m/s) — induces pitch disturbance
      4-5s:  Settle (hover)
      5-6s:  Pulse +Y velocity (5 m/s) — induces roll disturbance
      6-7s:  Settle (hover)
    Total duration: 7.0s.
    """
    # Capture current LOS as stabilization target on first call
    if not node._stab_los_target_set and node._gimbal_los is not None:
        node._stab_los_az_target = node._gimbal_los.x
        node._stab_los_el_target = node._gimbal_los.y
        node._stab_los_target_set = True
        node.get_logger().info(
            f"Gimbal stabilization target: az={node._stab_los_az_target:.1f}°, "
            f"el={node._stab_los_el_target:.1f}°"
        )

    if node._stab_los_target_set:
        node._publish_gimbal_los_target(
            node._stab_los_az_target, node._stab_los_el_target
        )

    if t < 2.0:
        node._publish_velocity(0.0, 0.0, 0.0, 0.5)
    elif t < 3.0:
        node._publish_velocity(0.0, 0.0, 0.0, 0.0)
    elif t < 4.0:
        node._publish_velocity(5.0, 0.0, 0.0, 0.0)
    elif t < 5.0:
        node._publish_velocity(0.0, 0.0, 0.0, 0.0)
    elif t < 6.0:
        node._publish_velocity(0.0, 5.0, 0.0, 0.0)
    else:
        node._publish_velocity(0.0, 0.0, 0.0, 0.0)


def build_test_sequence() -> list[TestSpec]:
    """Build the full sysid test sequence (drone + gimbal)."""
    return [
        TestSpec("hover_drift", 10.0, "Zero velocity, measure drift", _cmd_hover),
        TestSpec("vel_step_5", 10.0, "5 m/s forward step (ENU +X)", _cmd_vel_step_5),
        TestSpec("vel_step_10", 10.0, "10 m/s forward step (ENU +X)", _cmd_vel_step_10),
        TestSpec("vel_impulse_recovery", 10.0, "5 m/s 1s then zero", _cmd_vel_impulse_recovery),
        TestSpec("yaw_step", 10.0, "Yaw rate 0.5 rad/s 2s then zero", _cmd_yaw_step),
        TestSpec("gimbal_los_rate_step", 2.0, "Gimbal LOS rate: pitch→hold→yaw→hold", _cmd_gimbal_los_rate_step),
        TestSpec("gimbal_los_stabilization", 7.0, "Hold gimbal LOS while shaking drone", _cmd_gimbal_los_stabilization),
    ]


# ---------------------------------------------------------------------------
# CSV recording
# ---------------------------------------------------------------------------

CSV_HEADER = [
    "timestamp_s",
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    "quat_x", "quat_y", "quat_z", "quat_w",
    "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "gimbal_los_az_deg", "gimbal_los_el_deg",
    "gimbal_rpy_r_deg", "gimbal_rpy_p_deg", "gimbal_rpy_y_deg",
    # PX4 cascade setpoints (from ATTITUDE_TARGET)
    # Attitude setpoint quaternion (xyzw, ENU/FLU)
    "att_sp_qx", "att_sp_qy", "att_sp_qz", "att_sp_qw",
    # Rate setpoint (body FLU, rad/s)
    "rate_sp_x", "rate_sp_y", "rate_sp_z",
    # Thrust command (normalized 0-1)
    "thrust_sp",
]


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class SysidNode(Node):
    """System identification test node.

    Drives a single PX4 drone through step response tests via MAVROS.
    Control loop runs on a wallclock timer (matching mas_offboard pattern).
    /clock is used only for sim-time timestamps in recorded data.
    """

    def __init__(self) -> None:
        super().__init__("sysid_node")

        # -- Parameters --
        self.declare_parameter("vehicle_ns", "px4_1")
        self.declare_parameter("output_dir", "/tmp/sysid")
        self.declare_parameter("update_rate", 100.0)
        self.declare_parameter("settle_duration", 10.0)
        self.declare_parameter("takeoff_altitude", 15.0)  # meters ENU
        self.declare_parameter("takeoff_speed", 3.0)  # m/s upward

        self._vehicle_ns: str = self.get_parameter("vehicle_ns").value
        self._output_dir: str = self.get_parameter("output_dir").value
        self._update_rate: float = self.get_parameter("update_rate").value
        self._settle_duration: float = self.get_parameter("settle_duration").value
        self._takeoff_altitude: float = self.get_parameter("takeoff_altitude").value
        self._takeoff_speed: float = self.get_parameter("takeoff_speed").value

        os.makedirs(self._output_dir, exist_ok=True)

        # -- State --
        self._phase = TestPhase.INIT
        self._tests = build_test_sequence()
        self._test_idx: int = 0
        self._ramp_up_ticks: int = 0
        self._arm_request_tick: int = 0
        self._sim_time: float = 0.0  # from /clock, for CSV timestamps
        self._test_start_time: Optional[float] = None  # sim time
        self._phase_start_wallclock: Optional[float] = None  # wallclock for settle timing

        # Telemetry caches
        self._mavros_state: Optional[MavrosState] = None
        self._odom: Optional[Odometry] = None
        self._imu: Optional[Imu] = None
        self._gimbal_los: Optional[Vector3] = None
        self._gimbal_rpy: Optional[Vector3] = None
        self._att_target: Optional[AttitudeTarget] = None  # PX4 cascade internals

        # Per-test recording buffer
        self._record_buffer: list[list] = []

        # Gimbal stabilization test state
        self._stab_los_target_set: bool = False
        self._stab_los_az_target: float = 0.0
        self._stab_los_el_target: float = 0.0

        # -- Topic prefixes --
        mavros_prefix = f"/{self._vehicle_ns}/mavros"
        vehicle_prefix = f"/{self._vehicle_ns}"

        # -- Subscribers --
        self.create_subscription(
            MavrosState, f"{mavros_prefix}/state",
            self._mavros_state_cb, QOS_BEST_EFFORT,
        )
        self.create_subscription(
            Odometry, f"{mavros_prefix}/local_position/odom",
            self._odom_cb, QOS_BEST_EFFORT,
        )
        self.create_subscription(
            Imu, f"{mavros_prefix}/imu/data",
            self._imu_cb, QOS_BEST_EFFORT,
        )
        self.create_subscription(
            Vector3, f"{vehicle_prefix}/gimbal_los_state_deg",
            self._gimbal_los_cb, QOS_BEST_EFFORT,
        )
        self.create_subscription(
            Vector3, f"{vehicle_prefix}/gimbal_state_rpy_deg",
            self._gimbal_rpy_cb, QOS_BEST_EFFORT,
        )
        self.create_subscription(
            Clock, "/clock", self._clock_cb, QOS_BEST_EFFORT,
        )
        # PX4 cascade internals: attitude setpoint, rate setpoint, thrust
        self.create_subscription(
            AttitudeTarget, f"{mavros_prefix}/setpoint_raw/target_attitude",
            self._att_target_cb, QOS_BEST_EFFORT,
        )

        # -- Publishers --
        # TwistStamped on cmd_vel (matching mas_offboard, NOT cmd_vel_unstamped)
        self._vel_pub = self.create_publisher(
            TwistStamped, f"{mavros_prefix}/setpoint_velocity/cmd_vel",
            QOS_RELIABLE,
        )
        self._gimbal_rate_pub = self.create_publisher(
            Vector3, f"{vehicle_prefix}/gimbal_cmd_los_rate",
            QOS_BEST_EFFORT,
        )
        self._gimbal_los_target_pub = self.create_publisher(
            Vector3, f"{vehicle_prefix}/gimbal_cmd_los_world_deg",
            QOS_RELIABLE,
        )

        # -- Service clients --
        self._arm_client = self.create_client(
            CommandBool, f"{mavros_prefix}/cmd/arming",
        )
        self._mode_client = self.create_client(
            SetMode, f"{mavros_prefix}/set_mode",
        )

        # -- Wallclock timer drives the entire control loop --
        # Matching mas_offboard: wallclock timer, NOT /clock-driven.
        self._timer = self.create_timer(
            1.0 / self._update_rate, self._timer_cb,
        )

        self.get_logger().info(
            f"SysidNode initialized: ns={self._vehicle_ns}, "
            f"rate={self._update_rate}Hz, settle={self._settle_duration}s, "
            f"takeoff_alt={self._takeoff_altitude}m, {len(self._tests)} tests"
        )

    # -------------------------------------------------------------------
    # Subscriber callbacks (cache only)
    # -------------------------------------------------------------------

    def _mavros_state_cb(self, msg: MavrosState) -> None:
        self._mavros_state = msg

    def _odom_cb(self, msg: Odometry) -> None:
        self._odom = msg

    def _imu_cb(self, msg: Imu) -> None:
        self._imu = msg

    def _gimbal_los_cb(self, msg: Vector3) -> None:
        self._gimbal_los = msg

    def _gimbal_rpy_cb(self, msg: Vector3) -> None:
        self._gimbal_rpy = msg

    def _att_target_cb(self, msg: AttitudeTarget) -> None:
        self._att_target = msg

    def _clock_cb(self, msg: Clock) -> None:
        """Update sim time for CSV timestamps. Does NOT drive state machine."""
        self._sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    # -------------------------------------------------------------------
    # Command publishers
    # -------------------------------------------------------------------

    def _publish_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float = 0.0) -> None:
        """Publish TwistStamped velocity setpoint in ENU (matching mas_offboard)."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        msg.twist.angular.z = yaw_rate
        self._vel_pub.publish(msg)

    def _publish_zero_velocity(self) -> None:
        self._publish_velocity(0.0, 0.0, 0.0)

    def _publish_gimbal_rate(self, az_rate: float, el_rate: float) -> None:
        """Publish gimbal LOS rate command (rad/s). Positive el = up."""
        msg = Vector3(x=az_rate, y=el_rate, z=0.0)
        self._gimbal_rate_pub.publish(msg)

    def _publish_gimbal_los_target(self, az_deg: float, el_deg: float) -> None:
        """Publish gimbal world-frame LOS target (degrees)."""
        msg = Vector3(x=az_deg, y=el_deg, z=0.0)
        self._gimbal_los_target_pub.publish(msg)

    # -------------------------------------------------------------------
    # Arming (matching mas_offboard pattern)
    # -------------------------------------------------------------------

    def _request_offboard_and_arm(self) -> None:
        """Call MAVROS services to set OFFBOARD mode and arm. Throttled ~1 Hz."""
        self._arm_request_tick += 1
        ticks_per_second = int(self._update_rate)
        if self._arm_request_tick % ticks_per_second != 0:
            return

        # Request OFFBOARD mode (only if not already in it)
        if self._mavros_state is not None and self._mavros_state.mode != "OFFBOARD":
            if self._mode_client.service_is_ready():
                req = SetMode.Request()
                req.custom_mode = "OFFBOARD"
                future = self._mode_client.call_async(req)
                future.add_done_callback(self._set_mode_done)
                self.get_logger().info("Requesting OFFBOARD mode")
            else:
                self.get_logger().warn("set_mode service not ready")

        # Request arming (only if not already armed)
        if self._mavros_state is not None and not self._mavros_state.armed:
            if self._arm_client.service_is_ready():
                req = CommandBool.Request()
                req.value = True
                future = self._arm_client.call_async(req)
                future.add_done_callback(self._arming_done)
                self.get_logger().info("Requesting ARM")
            else:
                self.get_logger().warn("arming service not ready")

    def _set_mode_done(self, future) -> None:
        try:
            resp = future.result()
            if resp.mode_sent:
                self.get_logger().info("OFFBOARD mode request accepted")
            else:
                self.get_logger().warn("OFFBOARD mode request rejected")
        except Exception as e:
            self.get_logger().error(f"set_mode failed: {e}")

    def _arming_done(self, future) -> None:
        try:
            resp = future.result()
            if resp.success:
                self.get_logger().info("ARM command accepted")
            else:
                self.get_logger().warn(f"ARM command rejected (result={resp.result})")
        except Exception as e:
            self.get_logger().error(f"arming failed: {e}")

    # -------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------

    def _record_sample(self) -> None:
        """Append current telemetry to the active test buffer."""
        if self._odom is not None:
            p = self._odom.pose.pose.position
            v = self._odom.twist.twist.linear
            q = self._odom.pose.pose.orientation
            pos = [p.x, p.y, p.z]
            vel = [v.x, v.y, v.z]
            quat = [q.x, q.y, q.z, q.w]
        else:
            pos = vel = [float("nan")] * 3
            quat = [float("nan")] * 4

        if self._imu is not None:
            av = self._imu.angular_velocity
            ang_vel = [av.x, av.y, av.z]
        else:
            ang_vel = [float("nan")] * 3

        if self._gimbal_los is not None:
            glos = [self._gimbal_los.x, self._gimbal_los.y]
        else:
            glos = [float("nan")] * 2

        if self._gimbal_rpy is not None:
            grpy = [self._gimbal_rpy.x, self._gimbal_rpy.y, self._gimbal_rpy.z]
        else:
            grpy = [float("nan")] * 3

        # PX4 cascade setpoints (ATTITUDE_TARGET)
        if self._att_target is not None:
            aq = self._att_target.orientation
            ar = self._att_target.body_rate
            att_sp = [aq.x, aq.y, aq.z, aq.w]
            rate_sp = [ar.x, ar.y, ar.z]
            thrust_sp = [float(self._att_target.thrust)]
        else:
            att_sp = [float("nan")] * 4
            rate_sp = [float("nan")] * 3
            thrust_sp = [float("nan")]

        row = [self._sim_time] + pos + vel + quat + ang_vel + glos + grpy + att_sp + rate_sp + thrust_sp
        self._record_buffer.append(row)

    def _save_test_csv(self, test_name: str) -> None:
        """Write recording buffer to CSV and clear it."""
        if not self._record_buffer:
            self.get_logger().warn(f"No data recorded for {test_name}")
            return

        path = os.path.join(self._output_dir, f"{test_name}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            writer.writerows(self._record_buffer)

        n = len(self._record_buffer)
        dt_total = self._record_buffer[-1][0] - self._record_buffer[0][0]
        self.get_logger().info(f"Saved {test_name}: {n} samples, {dt_total:.1f}s → {path}")
        self._record_buffer.clear()

    # -------------------------------------------------------------------
    # Timer callback — wallclock-driven state machine
    # -------------------------------------------------------------------

    def _timer_cb(self) -> None:
        """Main control loop at update_rate Hz (wallclock).

        Matching mas_offboard: wallclock timer drives everything.
        /clock is only used for CSV timestamp recording.
        """
        if self._phase == TestPhase.INIT:
            self._state_init()
        elif self._phase == TestPhase.RAMP_UP:
            self._state_ramp_up()
        elif self._phase == TestPhase.ARM:
            self._state_arm()
        elif self._phase == TestPhase.TAKEOFF:
            self._state_takeoff()
        elif self._phase == TestPhase.SETTLE:
            self._state_settle()
        elif self._phase == TestPhase.TEST:
            self._state_test()
        elif self._phase == TestPhase.DONE:
            self._publish_zero_velocity()

    def _state_init(self) -> None:
        """Stream zero velocity while waiting for MAVROS topics."""
        self._publish_zero_velocity()
        if self._mavros_state is not None and self._odom is not None:
            self.get_logger().info("MAVROS topics received, starting ramp-up")
            self._phase = TestPhase.RAMP_UP
            self._ramp_up_ticks = 0

    def _state_ramp_up(self) -> None:
        """Stream zero velocity for 11+ ticks before requesting arm/offboard."""
        self._publish_zero_velocity()
        self._ramp_up_ticks += 1
        if self._ramp_up_ticks >= 15:
            self.get_logger().info(
                f"Ramp-up complete ({self._ramp_up_ticks} ticks), "
                "requesting OFFBOARD + arm"
            )
            self._phase = TestPhase.ARM
            self._arm_request_tick = 0

    def _state_arm(self) -> None:
        """Stream zero velocity, request arm+offboard, wait for confirmation."""
        self._publish_zero_velocity()
        self._request_offboard_and_arm()

        if (self._mavros_state is not None
                and self._mavros_state.armed
                and self._mavros_state.mode == "OFFBOARD"):
            self.get_logger().info(
                f"Armed + OFFBOARD confirmed, taking off to {self._takeoff_altitude}m"
            )
            self._phase = TestPhase.TAKEOFF

    def _state_takeoff(self) -> None:
        """Climb at takeoff_speed until target altitude reached."""
        self._publish_velocity(0.0, 0.0, self._takeoff_speed)

        if self._odom is not None:
            alt = self._odom.pose.pose.position.z
            if alt >= self._takeoff_altitude:
                self.get_logger().info(
                    f"Target altitude {self._takeoff_altitude}m reached "
                    f"(current {alt:.1f}m), settling before first test"
                )
                self._phase = TestPhase.SETTLE
                self._phase_start_wallclock = self.get_clock().now().nanoseconds * 1e-9

    def _state_settle(self) -> None:
        """Hover and wait for settle_duration."""
        self._publish_zero_velocity()

        if self._phase_start_wallclock is None:
            self._phase_start_wallclock = self.get_clock().now().nanoseconds * 1e-9

        now = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now - self._phase_start_wallclock

        if elapsed >= self._settle_duration:
            if self._test_idx >= len(self._tests):
                self.get_logger().info("All tests complete.")
                self._phase = TestPhase.DONE
            else:
                test = self._tests[self._test_idx]
                self.get_logger().info(
                    f"Starting test {self._test_idx + 1}/{len(self._tests)}: "
                    f"{test.name} — {test.description} ({test.duration}s)"
                )
                self._phase = TestPhase.TEST
                self._test_start_time = self._sim_time

    def _state_test(self) -> None:
        """Run the active test: execute command and record."""
        test = self._tests[self._test_idx]

        if self._test_start_time is None:
            self._test_start_time = self._sim_time

        t = self._sim_time - self._test_start_time

        if t < test.duration:
            test.command_fn(self, t)
            self._record_sample()
        else:
            # Test complete
            self._save_test_csv(test.name)
            self._stab_los_target_set = False
            self._test_idx += 1
            self.get_logger().info(
                f"Test {test.name} complete. Settling for {self._settle_duration}s."
            )
            self._phase = TestPhase.SETTLE
            self._phase_start_wallclock = self.get_clock().now().nanoseconds * 1e-9


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = SysidNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down sysid_node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
