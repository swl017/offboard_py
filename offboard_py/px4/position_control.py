import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class VehicleAttitudeSetpoint:
    q_d: np.ndarray  # Desired quaternion
    thrust_body: np.ndarray  # Desired thrust in body frame
    roll_body: float  # Desired roll angle
    pitch_body: float  # Desired pitch angle
    yaw_body: float  # Desired yaw angle
    yaw_sp_move_rate: float  # Desired yaw rate

@dataclass
class PositionControlStates:
    position: np.ndarray  # Current position
    velocity: np.ndarray  # Current velocity
    acceleration: np.ndarray  # Current acceleration
    yaw: float  # Current yaw angle

@dataclass
class TrajectorySetpoint:
    """Trajectory setpoint in NED frame"""
    timestamp: int
    position: np.ndarray  # Desired position
    velocity: np.ndarray  # Desired velocity
    acceleration: np.ndarray  # Desired acceleration
    jerk: np.ndarray  # Desired jerk
    yaw: float  # Desired yaw angle
    yawspeed: float  # Desired yaw speed

class ControlMath:
    @staticmethod
    def limit_tilt(body_unit: np.ndarray, world_unit: np.ndarray, max_angle: float) -> None:
        """Limits the tilt angle between two unit vectors."""
        # Determine tilt
        dot_product_unit = np.dot(body_unit, world_unit)
        angle = np.arccos(dot_product_unit)
        
        # Limit tilt
        angle = min(angle, max_angle)
        rejection = body_unit - (dot_product_unit * world_unit)
        
        # Handle case of exactly parallel vectors
        if np.linalg.norm(rejection) < np.finfo(float).eps:
            rejection[0] = 1.0
            
        body_unit[:] = np.cos(angle) * world_unit + np.sin(angle) * rejection / np.linalg.norm(rejection)

    @staticmethod
    def add_if_not_nan(setpoint: float, addition: float) -> float:
        """Add two values, handling NaN cases."""
        if np.isfinite(setpoint) and np.isfinite(addition):
            return setpoint + addition
        elif not np.isfinite(setpoint):
            return addition
        return setpoint

    @staticmethod
    def add_if_not_nan_vector3(setpoint: np.ndarray, addition: np.ndarray) -> np.ndarray:
        """Add two vectors element-wise, handling NaN cases."""
        result = setpoint.copy()
        for i in range(3):
            result[i] = ControlMath.add_if_not_nan(setpoint[i], addition[i])
        return result

    @staticmethod
    def set_zero_if_nan_vector3(vector: np.ndarray) -> None:
        """Set NaN elements of vector to zero."""
        vector[~np.isfinite(vector)] = 0.0

class PositionControl:
    """Position controller for multicopter vehicles."""
    
    HOVER_THRUST_MIN = 0.05
    HOVER_THRUST_MAX = 0.9
    CONSTANTS_ONE_G = 9.81

    def __init__(self):
        # Gains
        self._gain_pos_p = np.zeros(3)  # Position control proportional gain
        self._gain_vel_p = np.zeros(3)  # Velocity control proportional gain
        self._gain_vel_i = np.zeros(3)  # Velocity control integral gain
        self._gain_vel_d = np.zeros(3)  # Velocity control derivative gain

        # Limits
        self._lim_vel_horizontal = 0.0
        self._lim_vel_up = 0.0
        self._lim_vel_down = 0.0
        self._lim_thr_min = 0.0
        self._lim_thr_max = 0.0
        self._lim_thr_xy_margin = 0.0
        self._lim_tilt = 0.0
        
        # Hover thrust
        self._hover_thrust = 0.0
        
        # States
        self._pos = np.zeros(3)
        self._vel = np.zeros(3)
        self._vel_dot = np.zeros(3)
        self._vel_int = np.zeros(3)
        self._yaw = 0.0
        
        # Setpoints
        self._pos_sp = np.zeros(3)
        self._vel_sp = np.zeros(3)
        self._acc_sp = np.zeros(3)
        self._thr_sp = np.zeros(3)
        self._yaw_sp = 0.0
        self._yawspeed_sp = 0.0

    def set_position_gains(self, gains: np.ndarray) -> None:
        """Set position control gains."""
        self._gain_pos_p = gains

    def set_velocity_gains(self, p_gains: np.ndarray, i_gains: np.ndarray, d_gains: np.ndarray) -> None:
        """Set velocity control gains."""
        self._gain_vel_p = p_gains
        self._gain_vel_i = i_gains
        self._gain_vel_d = d_gains

    def set_velocity_limits(self, vel_horizontal: float, vel_up: float, vel_down: float) -> None:
        """Set velocity limits."""
        self._lim_vel_horizontal = vel_horizontal
        self._lim_vel_up = vel_up
        self._lim_vel_down = vel_down

    def set_thrust_limits(self, min_thrust: float, max_thrust: float) -> None:
        """Set thrust limits."""
        self._lim_thr_min = max(min_thrust, 1e-4)
        self._lim_thr_max = max_thrust

    def set_horizontal_thrust_margin(self, margin: float) -> None:
        """Set margin for horizontal thrust."""
        self._lim_thr_xy_margin = margin

    def set_tilt_limit(self, tilt: float) -> None:
        """Set maximum tilt angle."""
        self._lim_tilt = tilt

    def set_hover_thrust(self, hover_thrust: float) -> None:
        """Set hover thrust."""
        self._hover_thrust = np.clip(hover_thrust, self.HOVER_THRUST_MIN, self.HOVER_THRUST_MAX)

    def update_hover_thrust(self, hover_thrust_new: float) -> None:
        """Update hover thrust with compensation for the integrator."""
        previous_hover_thrust = self._hover_thrust
        self.set_hover_thrust(hover_thrust_new)
        
        # Compensate for hover thrust change in the integrator
        self._vel_int[2] += (self._acc_sp[2] - self.CONSTANTS_ONE_G) * previous_hover_thrust / self._hover_thrust + \
                           self.CONSTANTS_ONE_G - self._acc_sp[2]

    def set_state(self, states: PositionControlStates) -> None:
        """Set current vehicle state."""
        self._pos = states.position
        self._vel = states.velocity
        self._yaw = states.yaw
        self._vel_dot = states.acceleration

    def set_input_setpoint(self, setpoint: TrajectorySetpoint) -> None:
        """Set desired setpoints."""
        self._pos_sp = setpoint.position
        self._vel_sp = setpoint.velocity
        self._acc_sp = setpoint.acceleration
        self._yaw_sp = setpoint.yaw
        self._yawspeed_sp = setpoint.yawspeed

    def update(self, dt: float) -> bool:
        """Update position controller."""
        valid = self._input_valid()

        if valid:
            self._position_control()
            self._velocity_control(dt)
            
            self._yawspeed_sp = self._yawspeed_sp if np.isfinite(self._yawspeed_sp) else 0.0
            self._yaw_sp = self._yaw_sp if np.isfinite(self._yaw_sp) else self._yaw

        return valid and np.all(np.isfinite(self._acc_sp)) and np.all(np.isfinite(self._thr_sp))

    def _position_control(self) -> None:
        """Position proportional control."""
        vel_sp_position = (self._pos_sp - self._pos) * self._gain_pos_p
        
        # Add feed-forward velocity setpoint
        self._vel_sp = ControlMath.add_if_not_nan_vector3(self._vel_sp, vel_sp_position)
        
        # Make sure there are no NAN elements
        ControlMath.set_zero_if_nan_vector3(vel_sp_position)
        
        # Constrain horizontal velocity
        xy_vel = np.array([self._vel_sp[0], self._vel_sp[1]])
        if np.linalg.norm(xy_vel) > self._lim_vel_horizontal[0]:
            xy_vel = xy_vel / np.linalg.norm(xy_vel) * self._lim_vel_horizontal
            self._vel_sp[0:2] = xy_vel
            
        # Constrain vertical velocity
        self._vel_sp[2] = np.clip(self._vel_sp[2], -self._lim_vel_up, self._lim_vel_down)

    def _velocity_control(self, dt: float) -> None:
        """Velocity PID control with improved anti-windup.
        
        Args:
            dt: Time step in seconds
        """
        # Constrain vertical velocity integral
        self._vel_int[2] = np.clip(self._vel_int[2], -self.CONSTANTS_ONE_G, self.CONSTANTS_ONE_G)
        
        # PID velocity control
        vel_error = self._vel_sp - self._vel
        acc_sp_velocity = vel_error * self._gain_vel_p + self._vel_int - self._vel_dot * self._gain_vel_d
        
        # Add feed-forward acceleration
        self._acc_sp = ControlMath.add_if_not_nan_vector3(self._acc_sp, acc_sp_velocity)
        
        # Run acceleration control
        self._acceleration_control()
        
        # Integrator anti-windup for vertical direction
        if (self._thr_sp[2] >= -self._lim_thr_min and vel_error[2] >= 0.0) or \
           (self._thr_sp[2] <= -self._lim_thr_max and vel_error[2] <= 0.0):
            vel_error[2] = 0.0
            
        # Prioritize vertical control while keeping horizontal margin
        thrust_sp_xy = self._thr_sp[0:2]
        thrust_sp_xy_norm = np.linalg.norm(thrust_sp_xy)
        thrust_max_squared = self._lim_thr_max ** 2
        
        # Determine how much vertical thrust is left keeping horizontal margin
        allocated_horizontal_thrust = min(thrust_sp_xy_norm, self._lim_thr_xy_margin)
        thrust_z_max_squared = thrust_max_squared - allocated_horizontal_thrust ** 2
        
        # Saturate maximum vertical thrust
        self._thr_sp[2] = max(self._thr_sp[2], -np.sqrt(thrust_z_max_squared))
        
        # Determine how much horizontal thrust is left after prioritizing vertical control
        thrust_max_xy_squared = thrust_max_squared - self._thr_sp[2] ** 2
        thrust_max_xy = 0.0
        
        if thrust_max_xy_squared > 0.0:
            thrust_max_xy = np.sqrt(thrust_max_xy_squared)
            
        # Saturate thrust in horizontal direction
        if thrust_sp_xy_norm > thrust_max_xy:
            self._thr_sp[0:2] = thrust_sp_xy / thrust_sp_xy_norm * thrust_max_xy
            
        # Use tracking Anti-Windup for horizontal direction
        # During saturation, the integrator is used to unsaturate the output
        # see Anti-Reset Windup for PID controllers, L.Rundqwist, 1990
        arw_gain = 2.0 / self._gain_vel_p[0]
        
        acc_sp_xy_produced = self._thr_sp[0:2] * (self.CONSTANTS_ONE_G / self._hover_thrust)
        acc_sp_xy = self._acc_sp[0:2]
        
        # If produced and desired accelerations are in the same direction or if we're not saturated,
        # then don't apply anti-windup in that direction
        acc_sp_xy_norm = np.linalg.norm(acc_sp_xy)
        acc_sp_xy_produced_norm = np.linalg.norm(acc_sp_xy_produced)
        
        if acc_sp_xy_norm > acc_sp_xy_produced_norm:
            acc_limited_xy = acc_sp_xy_produced
        else:
            acc_limited_xy = acc_sp_xy
            
        # Apply anti-windup only to velocity directions that are indeed saturated
        vel_error[0:2] = vel_error[0:2] - arw_gain * (acc_sp_xy - acc_limited_xy)
        
        # Make sure integral doesn't get NaN
        ControlMath.set_zero_if_nan_vector3(vel_error)
        
        # Update integral component
        self._vel_int += vel_error * self._gain_vel_i * dt

    def _acceleration_control(self) -> None:
        """Convert acceleration setpoint to thrust vector.
        
        This method determines the desired body z-axis based on the acceleration setpoint
        and generates the appropriate thrust vector.
        """
        # Compute body z axis direction from desired acceleration
        body_z = np.array([-self._acc_sp[0], -self._acc_sp[1], self.CONSTANTS_ONE_G])
        
        # Check for zero vector case
        if np.all(np.abs(body_z) < np.finfo(float).eps):
            body_z = np.array([0.0, 0.0, 1.0])
        else:
            body_z = body_z / np.linalg.norm(body_z)
            
        # Apply tilt limit
        ControlMath.limit_tilt(body_z, np.array([0.0, 0.0, 1.0]), self._lim_tilt)
        
        # Scale thrust assuming hover thrust produces standard gravity
        collective_thrust = self._acc_sp[2] * (self._hover_thrust / self.CONSTANTS_ONE_G) - self._hover_thrust
        
        # Project thrust to body z axis
        z_projection = np.dot(np.array([0.0, 0.0, 1.0]), body_z)
        if abs(z_projection) > 0.0:  # Avoid division by zero
            collective_thrust = collective_thrust / z_projection
            
        # Limit thrust
        collective_thrust = min(collective_thrust, -self._lim_thr_min)
        
        # Generate final thrust vector
        self._thr_sp = body_z * collective_thrust

    def _input_valid(self) -> bool:
        """Check if input setpoints are valid."""
        valid = True
        
        # Check if at least one component (x,y,z) has a setpoint
        for i in range(3):
            valid = valid and (np.isfinite(self._pos_sp[i]) or 
                             np.isfinite(self._vel_sp[i]) or 
                             np.isfinite(self._acc_sp[i]))
        
        # xy components must come in pairs
        valid = valid and (np.isfinite(self._pos_sp[0]) == np.isfinite(self._pos_sp[1]))
        valid = valid and (np.isfinite(self._vel_sp[0]) == np.isfinite(self._vel_sp[1]))
        valid = valid and (np.isfinite(self._acc_sp[0]) == np.isfinite(self._acc_sp[1]))
        
        # State must be valid for controlled states
        for i in range(3):
            if np.isfinite(self._pos_sp[i]):
                valid = valid and np.isfinite(self._pos[i])
            if np.isfinite(self._vel_sp[i]):
                valid = valid and np.isfinite(self._vel[i]) and np.isfinite(self._vel_dot[i])
                
        return valid

    def get_attitude_setpoint(self) -> VehicleAttitudeSetpoint:
        """Get attitude setpoint from thrust vector."""
        # Calculate attitude from thrust vector
        body_z = -self._thr_sp / np.linalg.norm(self._thr_sp)
        
        # Vector of desired yaw direction in XY plane, rotated by PI/2
        y_C = np.array([-np.sin(self._yaw_sp), np.cos(self._yaw_sp), 0.0])
        
        # Desired body_x axis, orthogonal to body_z
        body_x = np.cross(y_C, body_z)
        
        # Handle singularity cases
        if body_z[2] < 0.0:
            body_x = -body_x
        
        if abs(body_z[2]) < 1e-6:
            body_x = np.array([0.0, 0.0, 1.0])
            
        body_x = body_x / np.linalg.norm(body_x)
        
        # Desired body_y axis
        body_y = np.cross(body_z, body_x)
        
        # Create rotation matrix
        R = np.column_stack((body_x, body_y, body_z))
        
        # Convert to quaternion
        trace = np.trace(R)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2,1] - R[1,2]) / S
            qy = (R[0,2] - R[2,0]) / S
            qz = (R[1,0] - R[0,1]) / S
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
                qw = (R[2,1] - R[1,2]) / S
                qx = 0.25 * S
                qy = (R[0,1] + R[1,0]) / S
                qz = (R[0,2] + R[2,0]) / S
            elif R[1,1] > R[2,2]:
                S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
                qw = (R[0,2] - R[2,0]) / S
                qx = (R[0,1] + R[1,0]) / S
                qy = 0.25 * S
                qz = (R[1,2] + R[2,1]) / S
            else:
                S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
                qw = (R[1,0] - R[0,1]) / S
                qx = (R[0,2] + R[2,0]) / S
                qy = (R[1,2] + R[2,1]) / S
                qz = 0.25 * S

        # Create attitude setpoint
        att_sp = VehicleAttitudeSetpoint(
            q_d=np.array([qw, qx, qy, qz]),
            thrust_body=np.array([0.0, 0.0, -np.linalg.norm(self._thr_sp)]),
            roll_body=np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)),
            pitch_body=np.arcsin(2 * (qw * qy - qz * qx)),
            yaw_body=np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)),
            yaw_sp_move_rate=self._yawspeed_sp
        )
        
        return att_sp

    def get_local_position_setpoint(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Get local position setpoint.
        
        Returns:
            Tuple containing:
            - position setpoint (np.ndarray)
            - velocity setpoint (np.ndarray)
            - acceleration setpoint (np.ndarray)
            - yaw setpoint (float)
            - yawspeed setpoint (float)
        """
        return (
            self._pos_sp,
            self._vel_sp,
            self._acc_sp,
            self._yaw_sp,
            self._yawspeed_sp
        )

    def reset_integral(self) -> None:
        """Reset the integral term of the controller."""
        self._vel_int = np.zeros(3)

class VehicleControlMode:
    """Vehicle control mode flags."""
    
    def __init__(self):
        self.flag_armed = False
        self.flag_multicopter_position_control_enabled = False
        self.flag_control_manual_enabled = False
        self.flag_control_auto_enabled = False
        self.flag_control_offboard_enabled = False
        self.flag_control_rates_enabled = False
        self.flag_control_attitude_enabled = False
        self.flag_control_velocity_enabled = False
        self.flag_control_position_enabled = False
        self.flag_control_altitude_enabled = False
        self.flag_control_climb_rate_enabled = False
        self.flag_control_acceleration_enabled = False
        self.flag_control_termination_enabled = False

    def set_flag(self, flag_name: str, value: bool) -> None:
        """Set control mode flag.
        
        Args:
            flag_name: Name of the flag to set
            value: Boolean value to set
        """
        if hasattr(self, flag_name):
            setattr(self, flag_name, value)

    def get_flag(self, flag_name: str) -> bool:
        """Get control mode flag value.
        
        Args:
            flag_name: Name of the flag to get
            
        Returns:
            Boolean value of the flag
        """
        if hasattr(self, flag_name):
            return getattr(self, flag_name)
        return False