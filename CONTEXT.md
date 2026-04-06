# OffboardControl Node

## Purpose
Bridges PX4 autopilot (NED-FRD, uORB) to ROS2 ecosystem (ENU-FLU, MAVROS-compatible topics). Accepts velocity commands and publishes position/velocity setpoints to PX4 in offboard mode. Republishes vehicle state in ENU-FLU conventions.

## Subscriptions
- `fmu/out/vehicle_status` (`px4_msgs/msg/VehicleStatus`) — PX4 arming and navigation state
- `fmu/out/vehicle_local_position` (`px4_msgs/msg/VehicleLocalPosition`) — local position in NED
- `fmu/out/vehicle_odometry` (`px4_msgs/msg/VehicleOdometry`) — full odometry in NED-FRD
- `fmu/out/vehicle_global_position` (`px4_msgs/msg/VehicleGlobalPosition`) — GPS position
- `cmd_vel` (`geometry_msgs/msg/TwistStamped`) — velocity commands from external planner

## Publishers
- `fmu/in/offboard_control_mode` (`px4_msgs/msg/OffboardControlMode`) — offboard heartbeat
- `fmu/in/trajectory_setpoint` (`px4_msgs/msg/TrajectorySetpoint`) — position/velocity setpoint in NED
- `fmu/in/vehicle_command` (`px4_msgs/msg/VehicleCommand`) — arm/disarm/mode commands
- `mavros/local_position/odom` (`nav_msgs/msg/Odometry`) — odometry in ENU-FLU
- `mavros/global_position/global` (`sensor_msgs/msg/NavSatFix`) — GPS position
- `mavros/imu/data` (`sensor_msgs/msg/Imu`) — IMU data in ENU-FLU
- `initial_waypoint` (`nav_msgs/msg/Odometry`) — initial position on first fix

## Parameters
- `namespace` (`string`) — vehicle namespace (px4_1, px4_2, ...)
- `target_system` (`int`) — PX4 system ID
- `initial_position` (`list[float]`) — [x, y, z, yaw_deg]
- `update_rate` (`float`, default: `100.0`) — control loop rate (Hz)

## Dependencies
Subscribes to PX4 uORB topics (fmu/out/*). Publishes MAVROS-compatible topics consumed by downstream nodes (e.g., gimbal_stabilizer, thrust_and_rate_control).

## Key Files
- `offboard_py/offboard_control.py` — Node implementation
- `offboard_py/rotations.py` — NED/ENU and FRD/FLU conversion utilities
- `config/vehicles.yaml` — Multi-agent vehicle configuration
- `launch/offboard.launch.py` — Multi-agent launch file

## Calling Contract

**Pattern**: Decoupled (subscribe → cache, timer → publish)

- `vehicle_status_callback()`: Caches arming/nav state. No publishing.
- `vehicle_local_position_callback()`: Caches local position. No publishing.
- `vehicle_odometry_callback()`: Caches odometry. No publishing.
- `vehicle_global_position_callback()`: Caches GPS. No publishing.
- `cmd_vel_callback()`: Caches velocity command. No publishing.
- `timer_callback()`: Reads cached state, publishes offboard heartbeat, trajectory setpoint, and MAVROS-compatible topics. Sole periodic mutation point.

---

# ThrustAndRateControl Node

## Purpose
Low-level attitude and thrust controller. Converts velocity commands into attitude targets using a PID-based stabilizing controller. Operates at 250 Hz.

## Subscriptions
- `mavros/local_position/pose` (`geometry_msgs/msg/PoseStamped`) — vehicle pose
- `mavros/local_position/velocity_local` (`geometry_msgs/msg/TwistStamped`) — local frame velocity
- `mavros/local_position/velocity_body` (`geometry_msgs/msg/TwistStamped`) — body frame velocity
- `cmd_vel` (`geometry_msgs/msg/TwistStamped`) — velocity commands from external planner

## Publishers
- `mavros/setpoint_raw/attitude` (`mavros_msgs/msg/AttitudeTarget`) — attitude + thrust command

## Parameters
- PID gains are hardcoded in `stabilizer.py` (linear_p=2.8, linear_d=0.2, linear_i=1.7, yaw_p=2.8, attitude_p=5.5, attitude_d=1.0)

## Dependencies
Subscribes to MAVROS topics published by offboard_control node. Publishes attitude targets consumed by MAVROS/PX4.

## Key Files
- `offboard_py/thrust_and_rate_control_ros2.py` — Node implementation
- `offboard_py/stabilizer.py` — DroneStabilizingController (PID control logic)

## Calling Contract

**Pattern**: Decoupled (subscribe → cache, timer → publish)

- `pose_callback()`: Caches current pose. No publishing.
- `velocity_local_callback()`: Caches local velocity. No publishing.
- `velocity_body_callback()`: Caches body velocity. No publishing.
- `cmd_vel_callback()`: Caches velocity command. No publishing.
- `timer_callback()` (250 Hz): Runs DroneStabilizingController, publishes AttitudeTarget. Sole periodic mutation point.
