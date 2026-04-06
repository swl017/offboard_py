# offboard_py Architecture

Multi-agent offboard drone control for PX4 autopilot. Handles position/velocity setpoints via PX4 uORB interface and attitude/thrust control via MAVROS, with NED/ENU and FRD/FLU frame conversions.

## Node Graph

```
┌──────────────────────────────────────────────────────────┐
│               offboard.launch.py                         │
│  Spawns one node per vehicle from config/vehicles.yaml   │
└──────────────────┬───────────────────────────────────────┘
                   │ launches (per vehicle namespace)
    ┌──────────────┴──────────────────────────────────┐
    │                                                  │
    ▼                                                  ▼
 offboard_control                        thrust_and_rate_control
 (position/velocity setpoints)           (attitude/thrust control)
```

## Directed Dependencies

```
PX4 Autopilot ──[fmu/out/*]──→ offboard_control           (VehicleStatus, VehicleLocalPosition, VehicleOdometry)
offboard_control ──[fmu/in/*]──→ PX4 Autopilot             (OffboardControlMode, TrajectorySetpoint, VehicleCommand)
offboard_control ──[mavros/*]──→ downstream consumers      (Odometry, NavSatFix, Imu in ENU/FLU)
external_planner ──[cmd_vel]──→ offboard_control            (TwistStamped velocity commands)
external_planner ──[cmd_vel]──→ thrust_and_rate_control     (TwistStamped velocity commands)
MAVROS ──[mavros/local_position/*]──→ thrust_and_rate_control (PoseStamped, TwistStamped)
thrust_and_rate_control ──[mavros/setpoint_raw/attitude]──→ MAVROS (AttitudeTarget)
```

The launch file is the sole integration point. Nodes should be independently testable.

## Data Flow

```
offboard_control (Decoupled pattern):
  Subscriber Callbacks (fmu/out/*, cmd_vel)
    ├─ Cache vehicle state (position, velocity, status)
    ├─ Cache velocity commands from cmd_vel
         │
         ▼
  Timer Callback (100 Hz)
    ├─ Publish OffboardControlMode heartbeat
    ├─ Publish TrajectorySetpoint (position or velocity)
    ├─ Publish converted mavros topics (Odometry, NavSatFix, Imu)

thrust_and_rate_control (Decoupled pattern):
  Subscriber Callbacks (mavros/local_position/*, cmd_vel)
    ├─ Cache pose, velocity, commands
         │
         ▼
  Timer Callback (250 Hz)
    ├─ DroneStabilizingController computes attitude + thrust
    ├─ Publish AttitudeTarget
```

## Topic/Service Interface

| Name | Msg/Srv Type | Direction | QoS | Description |
|------|-------------|-----------|-----|-------------|
| `fmu/out/vehicle_status` | `px4_msgs/VehicleStatus` | Sub | BestEffort | Vehicle arming/nav state |
| `fmu/out/vehicle_local_position` | `px4_msgs/VehicleLocalPosition` | Sub | BestEffort | Local position in NED |
| `fmu/out/vehicle_odometry` | `px4_msgs/VehicleOdometry` | Sub | BestEffort | Full odometry in NED-FRD |
| `fmu/out/vehicle_global_position` | `px4_msgs/VehicleGlobalPosition` | Sub | BestEffort | GPS position |
| `cmd_vel` | `geometry_msgs/TwistStamped` | Sub | Reliable | External velocity commands |
| `fmu/in/offboard_control_mode` | `px4_msgs/OffboardControlMode` | Pub | BestEffort | Offboard mode heartbeat |
| `fmu/in/trajectory_setpoint` | `px4_msgs/TrajectorySetpoint` | Pub | BestEffort | Position/velocity setpoint |
| `fmu/in/vehicle_command` | `px4_msgs/VehicleCommand` | Pub | BestEffort | Arm/disarm/mode commands |
| `mavros/local_position/odom` | `nav_msgs/Odometry` | Pub | Reliable | Odometry in ENU-FLU |
| `mavros/global_position/global` | `sensor_msgs/NavSatFix` | Pub | Reliable | GPS position |
| `mavros/imu/data` | `sensor_msgs/Imu` | Pub | Reliable | IMU in ENU-FLU |
| `initial_waypoint` | `nav_msgs/Odometry` | Pub | Reliable | Initial position on first fix |
| `mavros/local_position/pose` | `geometry_msgs/PoseStamped` | Sub | Reliable | Pose (thrust_and_rate_control) |
| `mavros/local_position/velocity_local` | `geometry_msgs/TwistStamped` | Sub | Reliable | Local velocity (thrust_and_rate_control) |
| `mavros/local_position/velocity_body` | `geometry_msgs/TwistStamped` | Sub | Reliable | Body velocity (thrust_and_rate_control) |
| `mavros/setpoint_raw/attitude` | `mavros_msgs/AttitudeTarget` | Pub | Reliable | Attitude + thrust command |

## Parameters

| Parameter | Type | Default | Node | Description |
|-----------|------|---------|------|-------------|
| `namespace` | string | — | offboard_control | Vehicle namespace (px4_1, px4_2, ...) |
| `target_system` | int | — | offboard_control | PX4 system ID |
| `initial_position` | list[float] | — | offboard_control | [x, y, z, yaw_deg] |
| `update_rate` | float | 100.0 | offboard_control | Control loop rate (Hz) |

## Node Isolation

**Has dependencies** (connected via topics/services):
- `offboard_control` — subscribes to PX4 uORB topics, publishes setpoints and MAVROS-compatible topics
- `thrust_and_rate_control` — subscribes to MAVROS pose/velocity topics, publishes attitude targets

## Stateful Mutation Rule

Both nodes use the **decoupled pattern**: subscriber callbacks cache incoming data, timer callbacks are the sole periodic mutation + publish points. Single-threaded executor — no thread safety concerns.

## File Conventions

- `offboard_py/*.py` — Node implementations
- `offboard_py/px4/` — PX4 control abstractions (position control, state, transforms)
- `config/*.yaml` — Default parameter values (vehicles.yaml, node_config.yaml)
- `launch/*.launch.py` — Launch files
- `CONTEXT.md` — Node routing contracts
- `doc/*_spec.md` — Authoritative specifications
