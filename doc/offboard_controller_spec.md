# mas_offboard Specification

**Module:** `mas_offboard`
**Status:** Draft v2
**Last Updated:** 2026-03-24

---

## 1. Motivation and Design Philosophy

### 1.1 The Problem
The current ROS2 implementation of offboard controller is crude and randomly dysfunctional. We need a clean interface from the policy to PX4. Also, we need to fly the drone to set the initial conditions matching `iris_ma6`.

We are building SITL simulation as part of sim-to-real deployment using policy trained in Isaac Lab. The policy was trained with the Isaac Lab's own actuator API in the environment. Also, initial conditions are 'given' or 'set', which does not happen in the real world. So, to drive the drone to those initial conditions, we need an offboard interface and control.

### 1.2 Design Philosophy
This package should serve as the generic interface to PX4 for the rest of the ROS2 system `mas`, as well as a primitive mission control (takeoff on user input → waypoint → offboard (start the mission using the learned policy) on user input → stop (hover) on user input → return).

The primary control mode is **velocity setpoints** — PX4 handles its own inner cascade control (attitude, rate, motor mixing). Direct attitude/thrust control (ThrustAndRateControl node) is backlogged for future implementation if needed.

### 1.3 Design Decisions

**MAVROS over uxrce-DDS**: We use MAVROS for all PX4 communication (both SITL and real hardware). Under high-rate ROS 2 UDP traffic, PX4's `uxrce_dds_client` can stall and stop publishing `/fmu/out/*` topics entirely, and the system doesn't recover even after the ROS 2 publisher stops — requiring a full reboot. (https://github.com/PX4/PX4-Autopilot/pull/26161 only fixed very recently (2026-03-24).) Since PegasusSimulator already bridges Isaac Sim → PX4 SITL via MAVLink, MAVROS works for SITL too — no need for uxrce in either path.

> **Note**: The current `OffboardControl` node still uses uxrce (px4_msgs). Migration to MAVROS is required. uxrce integration is backlogged.

**SITL integration path**: PegasusSimulator bridges Isaac Sim → PX4 SITL via direct MAVLink (pymavlink, TCP on port `4560 + vehicle_id`). PX4 SITL then speaks MAVLink to MAVROS. The offboard controller talks only to MAVROS topics — same interface for SITL and real hardware. See `PegasusSimulator/.../px4_mavlink_backend.py` for the sim-side bridge.

**Shared world frame**: Use the `mas_common_frame` package (`/home/usrg/mas/src/mas_common_frame/`). It subscribes to each vehicle's `mavros/home_position/home` (GPS origin) and `mavros/global_position/global`, transforms all vehicles into a shared ENU frame via GPS → ECEF → ENU, and publishes `/{vehicle}/common_frame/pose` + TF2 broadcasts. Parameterized by `common_frame_origin` (lat, lon, alt).

### 1.4 Architecture Overview
```
                         ┌─────────────────┐
                         │  RL Policy Node  │
                         │  (25 Hz, ENU)    │
                         └───┬──────────┬───┘
                  cmd_vel    │          │  reads mavros/
              [vx,vy,vz,    │          │  local_position/odom
               yaw_rate]    │          │
                             ▼          │
               ┌─────────────────────┐  │
               │ OffboardControl     │  │
               │ (100 Hz)            │  │
               │                     │  │
               │ • Mission state     │  │
               │   machine           │  │
               │ • Velocity setpoint │  │
               │   mux               │  │
               └──────┬──────────────┘  │
                      │                 │
         mavros/setpoint_velocity/cmd_vel
         mavros/cmd/arming              │
         mavros/set_mode                │
                      │                 │
                      ▼                 │
               ┌─────────────┐          │
               │   MAVROS    │◄─────────┘
               │  (MAVLink   │
               │   bridge)   │──── mavros/local_position/* ───► gimbal_stabilizer
               │             │──── mavros/global_position/* ──► mas_common_frame
               └──────┬──────┘──── mavros/imu/data ──────────► gimbal_stabilizer
                      │
                MAVLink
                      │
               ┌──────▼──────┐
               │  PX4 SITL   │
               │  or real FC  │
               └──────▲──────┘
                      │
                MAVLink (SITL only)
                      │
               ┌──────┴──────┐
               │ Pegasus     │
               │ Simulator   │
               │ (Isaac Sim) │
               └─────────────┘
```

## 2. Mathematical Formulation

### 2.1 Frame Conversions

The offboard controller bridges between PX4's NED-FRD and ROS2's ENU-FLU conventions.

**Inertial frame (NED ↔ ENU)**:
```
Position:   ENU [x, y, z]  ↔  NED [y, x, -z]
Velocity:   ENU [vx,vy,vz] ↔  NED [vy,vx,-vz]
Quaternion: q_ENU_to_NED = [w=0.70711, x=0.70711, y=0, z=0]
```

**Body frame (FRD ↔ FLU)**:
```
q_FLU_to_FRD = [w=1, x=0, y=0, z=0]   (180° rotation about X axis)
```

**Full odometry conversion** (NED-FRD → ENU-FLU):
```
q_ENU_FLU = q_ENU_to_NED * q_NED_FRD * q_FLU_to_FRD⁻¹
```

Utility functions in `offboard_py/px4/numpy_transforms.py`:
- `quat_rotate(q, v)` — rotate vector by quaternion (wxyz format)
- `euler_xyz_from_quat(q)` — extract roll, pitch, yaw
- `quat_from_euler_xyz(r, p, y)` — build quaternion from Euler angles

### 2.2 Common Frame Transform

For multi-agent operation, `mas_common_frame` transforms each vehicle's GPS position into a shared ENU frame:

```
GPS (lat, lon, alt)  →  ECEF (x, y, z)  →  ENU (e, n, u)
```

The ECEF conversion uses the WGS84 ellipsoid. The ENU conversion is relative to a configurable `common_frame_origin` (lat, lon, alt) shared by all agents. See `mas_common_frame/common_frame.py` for implementation.

> This will be improved to convert the EKF pose from ENU origin, not from raw GPS.

### 2.3 Policy Action Mapping

The `iris_ma6` policy outputs a **7D action** per agent (normalized to [-1, 1]):

| Index | Action | Scaling | Unit | Consumer |
|-------|--------|---------|------|----------|
| 0-2 | `vx, vy, vz` | × `max_lin_vel` (default 10) | m/s, ENU | OffboardControl → PX4 velocity setpoint |
| 3 | `yaw_rate` | × π/4 | rad/s | OffboardControl → PX4 yawspeed setpoint |
| 4 | `gimbal_yaw_rate` | normalized [-1, 1] | - | gimbal_stabilizer |
| 5 | `gimbal_pitch_rate` | normalized [-1, 1] | - | gimbal_stabilizer |
| 6 | `zoom_rate` | normalized [-1, 1] | - | gimbal_stabilizer |

- Policy runs at **25 Hz** (decimation=4, sim dt=0.01s → policy dt=0.04s)
- OffboardControl heartbeat runs at **100 Hz**, holding the last received command

### 2.4 Cascade Control (Backlogged)

When using velocity setpoints, PX4 handles its own inner cascade control loops (velocity → attitude → rate → motor mixing). Direct attitude/thrust control via the `ThrustAndRateControl` node and `DroneStabilizingController` is backlogged. The reference cascade implementation is in `iris_ma6/controller/` (velocity_controller → attitude_controller → rate_controller → motor_dynamics → mixer).

## 3. Implementation Details

### 3.1 Configuration

**OffboardControl node parameters** (declared in `offboard_control.py`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vehicle_name` | string | `''` | Namespace prefix for topics, used in frame_id |
| `update_rate` | float | `100.0` | Timer callback frequency (Hz) |
| `target_system` | int | `1` | PX4 MAVLink system ID for vehicle commands |
| `position.x` | float | `0.0` | Initial waypoint X in ENU (meters) |
| `position.y` | float | `0.0` | Initial waypoint Y in ENU (meters) |
| `position.z` | float | `0.0` | Initial waypoint Z in ENU (meters, positive up) |
| `position.yaw_deg` | float | `0.0` | Initial waypoint yaw (degrees) |

**Multi-agent vehicle config** (`config/vehicles.yaml`):

| Namespace | target_system | Position (x, y, z) ENU | Yaw (deg) |
|-----------|---------------|------------------------|-----------|
| `px4_1` | 2 | (-15, 0, 15) | -90 |
| `px4_2` | 3 | (0, -15, 15) | 0 |
| `px4_3` | 4 | (-15, 0, 8) | -45 |
| `px4_4` | 5 | (0, 0, 11) | 0 |
| `px4_5` | 6 | (2, 1, 1) | 0 |
| `px4_6` | 7 | (3, 1, 1) | 0 |

Launch: `offboard.launch.py` reads `vehicles.yaml` and spawns one `OffboardControl` node per vehicle, namespaced (e.g., `/px4_1/offboard_control`).

### 3.2 OffboardControl State Machine

The node uses a decoupled callback/timer pattern: subscriber callbacks cache incoming data; a single timer callback (100 Hz) reads cached state and publishes.

**State transitions:**

```
┌───────────┐    all topics     ┌──────────┐    counter ≥ 11    ┌──────┐
│   INIT    │ ──── received ───►│ RAMP-UP  │ ──────────────────►│ ARM  │
│ wait for  │                   │ stream   │                    │ cmd  │
│ PX4 topics│                   │ zero vel │                    │ offb │
└───────────┘                   └──────────┘                    └──┬───┘
                                                                   │
                                                    armed + offboard mode
                                                                   │
┌───────────┐   dist < 2m      ┌──────────┐    alt reached     ┌──▼────┐
│  POLICY   │◄── yaw < 10° ────│  HOVER   │◄──────────────────│TAKEOFF│
│ forward   │   (DISABLED)     │ position │                    │ climb │
│ cmd_vel   │                   │ hold     │                    │ -10m/s│
└───────────┘                   └──────────┘                    └───────┘
```

1. **INIT**: Wait for `vehicle_status`, `vehicle_local_position`, and `vehicle_global_position` messages to be non-None.
2. **RAMP-UP** (counter < 11): Stream zero-velocity setpoints + offboard heartbeat for ~100ms. PX4 requires this before accepting offboard mode.
3. **ARM**: When `arming_state == STANDBY`, send `VEHICLE_CMD_DO_SET_MODE` (offboard) then `VEHICLE_CMD_COMPONENT_ARM_DISARM` (arm).
4. **TAKEOFF**: Publish velocity setpoint `[0, 0, -10]` NED (10 m/s upward) until altitude is reached.
5. **HOVER**: Publish position setpoint at the configured waypoint. Monitor distance (< 2m) and yaw error (< 10°).
6. **POLICY** (currently disabled): Forward `cmd_vel` messages as velocity setpoints to PX4. Transition gated by `pass_vehicle_cmd_vel` flag which is commented out.

## 4. Integration Points

### 4.1 Input Interface

**Target interface (MAVROS):**

| Topic | Message Type | Frame | QoS | Source |
|-------|-------------|-------|-----|--------|
| `mavros/state` | `mavros_msgs/State` | - | RELIABLE | MAVROS (armed, mode) |
| `mavros/local_position/pose` | `geometry_msgs/PoseStamped` | ENU-FLU | RELIABLE | MAVROS |
| `mavros/local_position/odom` | `nav_msgs/Odometry` | ENU-FLU | RELIABLE | MAVROS |
| `mavros/local_position/velocity_local` | `geometry_msgs/TwistStamped` | ENU | RELIABLE | MAVROS |
| `mavros/global_position/global` | `sensor_msgs/NavSatFix` | GPS | RELIABLE | MAVROS |
| `mavros/imu/data` | `sensor_msgs/Imu` | ENU-FLU | RELIABLE | MAVROS |
| `cmd_vel` | `geometry_msgs/TwistStamped` | ENU | RELIABLE, VOLATILE | Policy node |

> **Current state**: The node still subscribes to `fmu/out/*` (uxrce/px4_msgs). Migration to the MAVROS topics above is required. With MAVROS, frame conversion from NED-FRD to ENU-FLU is handled by MAVROS itself, simplifying the node.

### 4.2 Output Interface

**Target interface (MAVROS):**

| Topic | Message Type | Frame | Rate | Consumer |
|-------|-------------|-------|------|----------|
| `mavros/setpoint_velocity/cmd_vel` | `geometry_msgs/TwistStamped` | ENU (body or local) | 100 Hz | PX4 (velocity setpoint) |
| `mavros/cmd/arming` | `mavros_msgs/CommandBool` | - | On event | PX4 (arm/disarm) |
| `mavros/set_mode` | `mavros_msgs/SetMode` | - | On event | PX4 (offboard mode) |
| `initial_waypoint` | `nav_msgs/Odometry` | ENU | 100 Hz | Mission planner |

> **Current state**: The node still publishes to `fmu/in/*` (uxrce/px4_msgs). Migration to MAVROS services/topics above is required. With MAVROS, velocity setpoints are sent in ENU directly — no manual NED conversion needed. Downstream nodes (gimbal_stabilizer, mas_common_frame) subscribe directly to MAVROS topics, so the node no longer needs to republish odom/imu/gps.

### 4.3 Dependencies

**Upstream:**
- PX4 Autopilot via MAVROS (both SITL and real hardware)
- MAVROS node (MAVLink ↔ ROS2 bridge)
- For SITL: PegasusSimulator → MAVLink → PX4 SITL → MAVLink → MAVROS

**Downstream (subscribe to MAVROS directly, not through OffboardControl):**
- `gimbal_stabilizer`: Subscribes to `mavros/local_position/odom` and `mavros/imu/data` for vehicle orientation
- `mas_common_frame`: Subscribes to `mavros/home_position/home` and `mavros/global_position/global` for coordinate transforms
- Policy node: Reads `mavros/local_position/odom` for state feedback, publishes `cmd_vel` to OffboardControl

### 4.4 Calling Contract

**Target design (after MAVROS migration):**

| Callback | Type | Frequency | Notes |
|----------|------|-----------|-------|
| `timer_callback` | WRITE (publish) | 100 Hz | Sole publish point; reads cached state, publishes velocity setpoint to MAVROS |
| `state_callback` | CACHE | ~1 Hz | Caches `mavros/state` (armed, mode) |
| `pose_callback` | CACHE | ~30 Hz | Caches pose from `mavros/local_position/pose` (already ENU-FLU) |
| `odom_callback` | CACHE | ~30 Hz | Caches odometry from `mavros/local_position/odom` (already ENU-FLU) |
| `cmd_vel_callback` | CACHE + PUBLISH | On message | Caches cmd_vel; if `pass_vehicle_cmd_vel`, publishes velocity setpoint directly (bypasses timer) |

**Stateful invariants:**
- Single-threaded executor — no thread safety concerns
- Subscriber callbacks only cache data; `timer_callback` is the sole publisher (except cmd_vel pass-through)
- MAVROS offboard mode requires continuous setpoint streaming at ≥2 Hz (we publish at 100 Hz)
- `offboard_setpoint_counter` tracks ramp-up phase (must reach 11 before arm command)
- With MAVROS, frame conversion (NED↔ENU) is handled by MAVROS — the node works entirely in ENU-FLU

## 5. Validation and Testing

### 5.1 Unit Tests
No unit tests currently exist. Recommended tests:

- **Frame conversion correctness**: Verify NED-FRD → ENU-FLU round-trip for known values. Compare quaternion-based rotation vs manual axis-swap methods (they should agree for all orientations).
- **State machine transitions**: Mock PX4 status messages, verify correct command sequence (offboard → arm → takeoff → hover).
- **Policy action scaling**: Verify normalized [-1, 1] actions are correctly scaled to m/s and rad/s.

### 5.2 Integration Tests

- **Single vehicle SITL**: Launch PX4 SITL + MAVROS + OffboardControl. Verify vehicle arms, takes off, and reaches waypoint within tolerance.
- **Multi-agent SITL**: Launch 6 vehicles. Verify all reach their waypoints without interference.
- **Policy pass-through**: Enable `pass_vehicle_cmd_vel`, publish known `cmd_vel`, verify PX4 receives correct velocity setpoint in NED.
- **End-to-end with downstream**: Verify gimbal_stabilizer and mas_common_frame receive valid odometry/GPS from OffboardControl.

## 6. Known Limitations

### Active Issues

1. **Requires MAVROS migration**: Current OffboardControl uses px4_msgs via uxrce-DDS. Must be rewritten to use MAVROS topics/services (see Section 1.3). This is the primary blocker — most other issues below will be resolved as part of this migration.

2. **Policy control disabled**: `pass_vehicle_cmd_vel = True` is commented out (`offboard_control.py:367`). The state machine never transitions from HOVER to POLICY mode.

3. **Takeoff height hardcoded**: `self.takeoff_height = -1.0` (NED) means the drone only climbs 1m before transitioning to hover, regardless of actual waypoint altitude.

4. **Typo**: `vehcile_cmd_vel` (missing 'h') in callback parameter and instance variable (`offboard_control.py:216-222`).

5. **mas_common_frame not yet wired**: The integration with `mas_common_frame` for shared world frame is not yet implemented in offboard_py.

### Resolved by MAVROS Migration

These issues in the current uxrce-based implementation will be eliminated when migrating to MAVROS:
- MAVROS publishers commented out (downstream subscribes to MAVROS directly)
- Inconsistent NED↔ENU frame conversion (MAVROS handles this)
- Manual quaternion rotation and axis-swap code (no longer needed)

### Backlogged (ThrustAndRateControl Node)

The direct attitude/thrust control path has additional issues but is not needed for velocity-command mode:
- Hardcoded `cmd_vel = [-5, 0, 0]` overwrites subscription every tick
- `DroneStabilizingController` reinstantiated every tick (integral state lost)
- Undocumented thrust scaling magic numbers (`T/W * 1.5 / 14.6 * 0.7`)
- No integral anti-windup on velocity PID

## 7. References
- **`iris_ma6` environment**: `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/CLAUDE.md`
- **`iris_ma6` controller**: `IsaacLab/.../iris_ma6/controller/` (reference cascade implementation)
- **`mas` system**: `/home/usrg/mas/src/CLAUDE.md`
- **`mas_common_frame`**: `/home/usrg/mas/src/mas_common_frame/`
- **`PX4`**: `/home/usrg/IsaacPX4/PX4-Autopilot/CLAUDE.md`
- **PegasusSimulator**: `PegasusSimulator/.../px4_mavlink_backend.py` (SITL MAVLink bridge)
- **Package architecture**: `ARCHITECTURE.md` (node graph and topic/service flow)
- **Node contracts**: `CONTEXT.md` (per-node interface details)
