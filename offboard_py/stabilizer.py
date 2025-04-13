import torch
# from omni.isaac.lab.utils.math import (
#     quat_rotate, 
#     quat_rotate_inverse, 
#     euler_xyz_from_quat, 
#     quat_from_euler_xyz,
#     )
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat
from transforms3d.euler import quat2euler, euler2quat

# For quat_rotate
def quat_rotate(q, v):
    """Rotate a vector by a quaternion"""
    return quat2mat(q) @ v

# For quat_rotate_inverse
def quat_rotate_inverse(q, v):
    """Rotate a vector by the inverse of a quaternion"""
    return quat2mat(q).T @ v  # Transpose of rotation matrix = inverse rotation

# For euler_xyz_from_quat
def euler_xyz_from_quat(q):
    """Convert quaternion to Euler angles (XYZ convention)"""
    euler = quat2euler(q, 'sxyz')
    return euler[0], euler[1], euler[2]  # 's' means static frame

# For quat_from_euler_xyz
def quat_from_euler_xyz(roll, pitch, yaw):
    """Convert Euler angles (XYZ convention) to quaternion"""
    return euler2quat(roll, pitch, yaw, 'sxyz')

class DroneStabilizingController:
    """Controller for drone stabilization with velocity commands in world frame.
    
    The controller takes linear velocity commands in world frame and yaw velocity command,
    while automatically managing roll and pitch for stability. It generates appropriate 
    accelerations in the body frame.
    """

    def __init__(
        self,
        linear_p_gain: float = 2.8,
        linear_d_gain: float = 0.2,
        linear_i_gain: float = 1.7,
        yaw_p_gain: float = 2.8,
        yaw_d_gain: float = 0.0,
        attitude_p_gain: float = 5.5,
        attitude_d_gain: float = 1.0,
        device: str = "cuda"
    ):
        """Initialize the drone stabilizing controller.
        
        Args:
            linear_p_gain: Proportional gain for linear velocity control. Defaults to 2.0.
            linear_d_gain: Derivative gain for linear velocity control. Defaults to 1.0.
            yaw_p_gain: Proportional gain for yaw control. Defaults to 2.0.
            yaw_d_gain: Derivative gain for yaw control. Defaults to 1.0.
            attitude_p_gain: Proportional gain for roll/pitch control. Defaults to 3.0.
            attitude_d_gain: Derivative gain for roll/pitch control. Defaults to 1.0.
            device: Device to create tensors on. Defaults to "cpu".
        """
        self.device = device
        # Store gains as tensors
        self.kp_lin = linear_p_gain #torch.tensor(linear_p_gain, device=device)
        self.kd_lin = linear_d_gain #torch.tensor(linear_d_gain, device=device)
        self.ki_lin = linear_i_gain #torch.tensor(linear_i_gain, device=device)
        self.kp_yaw = yaw_p_gain #torch.tensor(yaw_p_gain, device=device)
        self.kd_yaw = yaw_d_gain #torch.tensor(yaw_d_gain, device=device)
        self.kp_att = attitude_p_gain #torch.tensor(attitude_p_gain, device=device)
        self.kd_att = attitude_d_gain #torch.tensor(attitude_d_gain, device=device)
        self.cumul_lin_vel_error = None
        self.last_lin_vel_error = None
        self.last_ang_vel_error_b = None
        
    def compute_control(
        self,
        cmd_lin_vel_w, #: torch.Tensor,  # Command linear velocity in world frame
        cmd_yaw_vel, #: torch.Tensor,    # Command yaw velocity (scalar)
        curr_quat_w, #: torch.Tensor,    # Current orientation in world frame (w,x,y,z)
        curr_lin_vel_w, #: torch.Tensor, # Current linear velocity in world frame
        curr_ang_vel_b, #: torch.Tensor, # Current angular velocity in body frame
        dt: float = 0.02             # Time step
    ):
        # batch_size = cmd_lin_vel_w.shape[0]
        
        # Extract current roll, pitch, yaw
        roll, pitch, yaw = euler_xyz_from_quat(curr_quat_w)
        
        # Compute linear velocity error in world frame
        lin_vel_error_w = cmd_lin_vel_w - curr_lin_vel_w
        
        # Convert linear velocity error to body frame for control
        curr_yaw_quat_w = quat_from_euler_xyz(np.zeros_like(roll), np.zeros_like(pitch), yaw)
        lin_vel_error_b_w = quat_rotate_inverse(curr_yaw_quat_w, lin_vel_error_w)
        
        # Compute control outputs with proper broadcasting
        # Linear acceleration in body frame
        if self.cumul_lin_vel_error is None:
            self.cumul_lin_vel_error = np.zeros_like(lin_vel_error_w)
        self.cumul_lin_vel_error += lin_vel_error_w
        # self.cumul_lin_vel_error.clip(-10.0 * 0.1 / dt, 10.0 * 0.1 / dt)
        if self.last_lin_vel_error is None:
            self.last_lin_vel_error = lin_vel_error_w
        lin_acc_cmd_w = (
            self.kp_lin * lin_vel_error_w +
            self.kd_lin * (-self.last_lin_vel_error) +
            self.ki_lin * self.cumul_lin_vel_error
        )
        # lin_acc_cmd_w = quat_rotate(curr_yaw_quat_w, lin_acc_cmd_w)
        lin_acc_cmd_w[:2] = lin_acc_cmd_w[:2].clip(-5.0, 5.0)
        lin_acc_cmd_w[2] = lin_acc_cmd_w[2].clip(-1.0, 5.0)

        # Get desired tilt direction from velocity command
        # Compute xy velocity command magnitude
        xy_vel_cmd = cmd_lin_vel_w[:2]
        xy_vel_norm = np.linalg.norm(xy_vel_cmd)
        


        z_w = np.array([0.0, 0.0, 1.0])
        # desired_thrust_per_weight = (9.82 + lin_acc_cmd_b_w[2]) / (
        #     z_w * quat_rotate_inverse(curr_quat_w, z_w))

        desired_thrust_per_weight = ((9.81 + lin_acc_cmd_w[2]) / np.cos(roll)) / np.cos(pitch)
        # desired_thrust_per_weight = desired_thrust_per_weight.clip(0.0, 5.0)

        # desired_thrust_per_weight = np.sqrt(
        #     (lin_acc_cmd_b_w[2] + 9.81)**2 * (1.0 + np.tan(-roll)**2 + np.tan(pitch)**2))

        # desired_thrust_per_weight = np.sqrt(lin_acc_cmd_b_w[0]**2 + lin_acc_cmd_b_w[1]**2 + (lin_acc_cmd_b_w[2]-9.81)**2)

        # Initialize desired roll and pitch tensors
        desired_roll = np.zeros_like(roll)
        desired_pitch = np.zeros_like(pitch)

        # Compute desired roll and pitch only where velocity is significant
        # tilt_threshold = 0.0
        # mask = xy_vel_norm > tilt_threshold
        # angle_limit = np.pi / 180.0 * 40.0
        # desired_roll[mask] -= np.arctan(lin_acc_cmd_b_w[mask, 1])
        # desired_roll.clip(-angle_limit, angle_limit)
        # desired_pitch[mask] += np.arctan(lin_acc_cmd_b_w[mask, 0])
        # desired_pitch.clip(-angle_limit, angle_limit)

        desired_roll_pitch_sin = (1/desired_thrust_per_weight) * np.linalg.inv(
                np.array([[np.cos(roll)*np.cos(yaw), np.sin(yaw)],
                          [np.cos(roll)*np.sin(yaw), -np.cos(yaw)]])
            ) @ np.array([(lin_acc_cmd_w[0]), (lin_acc_cmd_w[1])])
        desired_roll = np.arcsin(desired_roll_pitch_sin[1])
        desired_pitch = np.arcsin(desired_roll_pitch_sin[0])
        tilt_limit = np.pi / 180.0 * 35.0
        desired_roll = desired_roll.clip(-tilt_limit, tilt_limit)
        desired_pitch = desired_pitch.clip(-tilt_limit, tilt_limit)
        self.desired_roll = desired_roll
        self.desired_pitch = desired_pitch

    
        lin_acc_cmd_b = quat_rotate_inverse(
            quat_from_euler_xyz(desired_roll, desired_pitch, yaw), 
            lin_acc_cmd_w)
        
        # (lin_acc_cmd_b_w[1] / np.sin(-desired_roll) +
        #                             lin_acc_cmd_b_w[0] / np.sin(desired_pitch) +
        #                             lin_acc_cmd_b[2])

        # Compute attitude error
        roll_error = (desired_roll - roll)
        pitch_error = (desired_pitch - pitch)
        yaw_vel_error = (cmd_yaw_vel - curr_ang_vel_b[2])
        
        # Compute desired body rates with proper broadcasting
        desired_ang_vel_b = np.zeros_like(curr_ang_vel_b)
        desired_ang_vel_b[0] = self.kp_att * roll_error
        desired_ang_vel_b[1] = self.kp_att * pitch_error
        desired_ang_vel_b[2] = cmd_yaw_vel
        
        # Angular velocity error in body frame
        ang_vel_error_b = desired_ang_vel_b - curr_ang_vel_b
        
        # Angular acceleration in body frame
        ang_acc_cmd_b = np.zeros_like(curr_ang_vel_b)
        
        if self.last_ang_vel_error_b is None:
            self.last_ang_vel_error_b = np.zeros_like(curr_ang_vel_b)

        # Roll and pitch control
        ang_acc_cmd_b[:2] = (
            self.kp_att * ang_vel_error_b[:2] +
            self.kd_att * (-self.last_ang_vel_error_b[:2])
        )
        self.last_ang_vel_error_b = ang_vel_error_b
        
        # Yaw control
        ang_acc_cmd_b[2] = (
            self.kp_yaw * yaw_vel_error +
            self.kd_yaw * (-curr_ang_vel_b[2])
        )
        
        return desired_thrust_per_weight, ang_acc_cmd_b, desired_ang_vel_b
    
    def reset(self):
        """Reset the controller state."""
        pass  # No internal state to reset in this implementation

if __name__ == "__main__":
    # Test the controller with dummy inputs
    controller = DroneStabilizingController()
    cmd_lin_vel_w = np.array([1.0, 0.0, 0.0]) #torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    cmd_yaw_vel = 0.0 #torch.tensor([0.0], dtype=torch.float32)
    curr_quat_w = np.array([1.0, 0.0, 0.0, 0.0]) #torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    curr_lin_vel_w = np.array([0.0, 0.0, 0.0]) #torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    curr_ang_vel_b = np.array([0.0, 0.0, 0.0]) # torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    control = controller.compute_control(
        cmd_lin_vel_w, cmd_yaw_vel, curr_quat_w, curr_lin_vel_w, curr_ang_vel_b)
    thrust = control[0]
    ang_acc = control[1]
    ang_vel = control[2]
    print("Thrust per weight:", thrust)
    print("Angular acceleration:", ang_acc)
    print("Angular velocity:", ang_acc)