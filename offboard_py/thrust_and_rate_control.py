#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np

# Import message types
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Imu
from mavros_msgs.msg import AttitudeTarget

# Assuming these custom libraries are available in your ROS2 workspace
# or Python path. No changes are needed for these imports themselves.
from .px4 import position_control as pc
from .px4 import numpy_transforms as nt
from .px4 import states as states
from .px4.states import rot_ENU_to_NED, rot_FLU_to_FRD
from scipy.spatial.transform import Rotation

from .stabilizer import DroneStabilizingController

class ThrustAndRateControl(Node):
    """
    A ROS2 node for controlling a drone's thrust and body rates based on velocity commands.
    """
    def __init__(self):
        # Initialize the node
        super().__init__('thrust_and_rate_control')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_profile_durability = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Create subscribers
        self.pose_w_sub = self.create_subscription(
            PoseStamped, 'mavros/local_position/pose', self.pose_callback, 10)
        self.twist_w_sub = self.create_subscription(
            TwistStamped, 'mavros/local_position/velocity_local', self.twist_w_callback, 10)
        self.twist_b_sub = self.create_subscription(
            TwistStamped, 'mavros/local_position/velocity_body', self.twist_b_callback, 10)
        self.cmd_vel_sub = self.create_subscription(
            TwistStamped, 'cmd_vel', self.cmd_vel_callback, 10)

        # Create publisher
        self.attitude_pub = self.create_publisher(
            AttitudeTarget, 'mavros/setpoint_raw/attitude', 10)

        # Initialize state variables
        self.curr_pose = None
        self.curr_lin_vel_w = None
        self.curr_ang_vel_b = None
        self.cmd_lin_vel_w = None
        self.cmd_yaw_vel = None
        self.dt = None
        self.last_twist = None
        self.last_twist_time = None
        self.last_ros_time = None
        self.latest_stamp = None
        self.att_target = AttitudeTarget() # Pre-initialize message

        # Drone parameters
        self.thrust_per_mass = 0.7 / 1.5
        self.inertia = np.array([0.029125, 0.029125, 0.055225])

        # Setup controllers
        self.pc = pc.PositionControl()
        self.setup_position_controller()
        self.states = states.State()

        # Create a timer to run the control loop at 250 Hz
        timer_period = 1.0 / 250.0  # seconds
        self.timer = self.create_timer(timer_period, self.compute_control)

    def setup_position_controller(self):
        """Sets up the parameters for the PX4 position controller."""
        self.pc.set_position_gains(np.array([0.95, 0.95, 1.0]))
        self.pc.set_velocity_gains(
            p_gains=np.array([1.8, 1.8, 4]),
            d_gains=np.array([0.2, 0.2, 0.0]),
            i_gains=np.array([0.4, 0.4, 2.0]))
        self.pc.set_velocity_limits(
            vel_horizontal=np.array([1.0, 1.0]),
            vel_up=5.0,
            vel_down=-3.0)
        self.pc.set_thrust_limits(
            min_thrust=0.1,
            max_thrust=1.0)
        self.pc.set_horizontal_thrust_margin(0.3)
        self.pc.set_tilt_limit(np.pi / 4.0)
        self.pc.set_hover_thrust(0.7)
    
    def get_ros_time_seconds(self, stamp):
        """Converts a ROS2 time stamp to seconds."""
        return stamp.sec + stamp.nanosec / 1e9

    def set_states(self, position_w, velocity_w, acceleration_w, yaw):
        """Sets the current state for the position controller."""
        pc_states = pc.PositionControlStates(
            position=position_w,
            velocity=velocity_w,
            acceleration=acceleration_w,
            yaw=yaw,
        )
        self.pc.set_state(pc_states)

    def set_setpoint(self, value):
        """Sets the trajectory setpoint for the position controller."""
        timestamp_sec = 0.0 if self.latest_stamp is None else self.get_ros_time_seconds(self.latest_stamp)
        pc_setpoint = pc.TrajectorySetpoint(
            timestamp=timestamp_sec,
            position=np.array([np.nan, np.nan, np.nan]),
            velocity=value,
            acceleration=np.array([np.nan, np.nan, np.nan]),
            jerk=np.array([np.nan, np.nan, np.nan]),
            yaw=0.0,
            yawspeed=0.0,
        )
        self.pc.set_input_setpoint(pc_setpoint)

    # --- Subscriber Callbacks ---
    def pose_callback(self, msg):
        self.latest_stamp = msg.header.stamp
        self.curr_pose = msg
        self.curr_quat_w = np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        self.states.position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.states.set_attitude(self.curr_quat_w)

    def twist_w_callback(self, msg):
        self.latest_stamp = msg.header.stamp
        self.curr_lin_vel_w = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.states.linear_velocity = self.curr_lin_vel_w

        lin_acc = np.array([0.0, 0.0, 0.0])
        current_time_sec = self.get_ros_time_seconds(msg.header.stamp)

        if self.last_twist is None:
            self.last_twist = self.curr_lin_vel_w
            self.last_twist_time = current_time_sec
        else:
            dt = current_time_sec - self.last_twist_time
            if dt < 0.001:
                dt = 0.001
            lin_acc = (self.curr_lin_vel_w - self.last_twist) / dt
        
        self.states.linear_acceleration = lin_acc
        self.last_twist = self.curr_lin_vel_w
        self.last_twist_time = current_time_sec


    def twist_b_callback(self, msg):
        self.latest_stamp = msg.header.stamp
        self.curr_ang_vel_b = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])
        self.states.linear_body_velocity = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.states.angular_velocity = self.curr_ang_vel_b

    def cmd_vel_callback(self, msg):
        self.cmd_lin_vel_w = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.cmd_yaw_vel = msg.twist.angular.z

    def compute_control(self):
        """Main control loop for calculating and publishing control commands."""
        if self.curr_pose is None or self.curr_lin_vel_w is None or self.curr_ang_vel_b is None:
            return

        ros_time = self.get_clock().now().nanoseconds / 1e9
        if self.last_ros_time is None:
            self.last_ros_time = ros_time - 0.01
            return
        
        self.dt = ros_time - self.last_ros_time
        if self.dt < 0.001:
            if self.latest_stamp:
                self.att_target.header.stamp = self.latest_stamp
                self.attitude_pub.publish(self.att_target)
            return

        self.last_ros_time = ros_time
        
        # --- The original logic from the ROS1 node starts here ---
        
        self.cmd_lin_vel_w = np.array([-5.0, 0.0, 0.0])
        self.cmd_yaw_vel = 0.0
        
        ctrl = DroneStabilizingController()
        desired_thrust_per_weight, ang_acc_cmd_b, desired_ang_vel_b = ctrl.compute_control(
            cmd_lin_vel_w=self.cmd_lin_vel_w,
            cmd_yaw_vel=self.cmd_yaw_vel,
            curr_quat_w=self.curr_quat_w,
            curr_lin_vel_w=self.curr_lin_vel_w,
            curr_ang_vel_b=self.curr_ang_vel_b,
            dt=self.dt
        )

        if not np.isfinite(desired_thrust_per_weight) or not np.isfinite(desired_ang_vel_b).all():
            self.get_logger().warn('Control computation resulted in non-finite values.')
            if self.latest_stamp:
                 self.att_target.header.stamp = self.latest_stamp
                 self.attitude_pub.publish(self.att_target)
            return

        self.att_target.header.stamp = self.get_clock().now().to_msg()
        self.att_target.type_mask = AttitudeTarget.IGNORE_ATTITUDE
        self.att_target.thrust = float(np.max([0.0, desired_thrust_per_weight]) * 1.5 / 14.6 * 0.7)
        self.att_target.body_rate.x = float(desired_ang_vel_b[0])
        self.att_target.body_rate.y = float(desired_ang_vel_b[1])
        self.att_target.body_rate.z = float(desired_ang_vel_b[2])
        
        self.attitude_pub.publish(self.att_target)

        # Logging for debug
        self.get_logger().info(f"Thrust: {self.att_target.thrust:.2f} | "
                             f"Rates: {self.att_target.body_rate.x:.2f}, "
                             f"{self.att_target.body_rate.y:.2f}, "
                             f"{self.att_target.body_rate.z:.2f}")
        self.get_logger().info(f"Desired Roll: {(ctrl.desired_roll * 180.0 / np.pi):.2f}, "
                             f"Desired Pitch: {(ctrl.desired_pitch * 180.0 / np.pi):.2f}")


def main(args=None):
    """The main function to run the ROS2 node."""
    rclpy.init(args=args)
    controller = ThrustAndRateControl()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Keyboard interrupt, shutting down.')
    finally:
        # Cleanly destroy the node and shut down rclpy
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()