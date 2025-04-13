#!/usr/bin/env python3

import rospy

import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Imu
from mavros_msgs.msg import AttitudeTarget

import px4.position_control as pc
from px4.numpy_transforms import *
import px4.states as states
from px4.states import rot_ENU_to_NED, rot_FLU_to_FRD
from scipy.spatial.transform import Rotation

from stabilizer import DroneStabilizingController

class ThrustAndRateControl:
    def __init__(self):
        rospy.init_node('thrust_and_rate_control', anonymous=True)
        self.pose_w_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_callback)
        self.twist_w_sub = rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, self.twist_w_callback)
        self.twist_b_sub = rospy.Subscriber('/mavros/local_position/velocity_body', TwistStamped, self.twist_b_callback)
        self.attitude_pub = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)
        self.cmd_vel_sub = rospy.Subscriber('/cmd_vel', TwistStamped, self.cmd_vel_callback)
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

        self.thrust_per_mass = 0.7 / 1.5
        self.inertia = np.array([0.029125, 0.029125, 0.055225])

        self.pc = pc.PositionControl()
        self.setup_position_controller()

        self.states = states.State()

    def setup_position_controller(self):
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
        self.pc.set_tilt_limit(np.pi/4.0)
        self.pc.set_hover_thrust(0.7)

    def set_states(self, position_w, velocity_w, acceleration_w, yaw):
        pc_states = pc.PositionControlStates(
            position=position_w,
            velocity=velocity_w,
            acceleration=acceleration_w,
            yaw=yaw,
        )
        self.pc.set_state(pc_states)

    def set_setpoint(self, value):
        pc_setpoint = pc.TrajectorySetpoint(
            timestamp=0.0 if self.latest_stamp is None else self.latest_stamp.to_sec(),
            position=np.array([np.nan, np.nan, np.nan]),
            velocity=value,#np.array([np.nan, np.nan, np.nan]),#np.array([0.0, 1.0, 0.0]),
            acceleration=np.array([np.nan, np.nan, np.nan]),#np.array([0.0, 0.0, 0.0]),
            jerk=np.array([np.nan, np.nan, np.nan]),
            yaw=0.0,
            yawspeed=0.0,
        )
        self.pc.set_input_setpoint(pc_setpoint)

    def pose_callback(self, msg):
        self.latest_stamp = msg.header.stamp
        self.curr_pose = msg
        self.curr_quat_w = np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        self.states.position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.states.set_attitude(self.curr_quat_w)

    def twist_w_callback(self, msg):
        self.latest_stamp = msg.header.stamp
        self.curr_lin_vel_w = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.states.linear_velocity = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        lin_acc = np.array([0.0, 0.0, 0.0])
        if self.last_twist is None:
            self.last_twist = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
            self.last_twist_time = msg.header.stamp.to_sec()
        else:
            dt = msg.header.stamp.to_sec() - self.last_twist_time
            if dt < 0.001:
                dt = 0.001
            lin_acc = (self.curr_lin_vel_w - self.last_twist) / dt
        self.states.linear_velocity = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.states.linear_acceleration = lin_acc
        self.last_twist = self.curr_lin_vel_w

    def twist_b_callback(self, msg):
        self.latest_stamp = msg.header.stamp
        self.curr_ang_vel_b = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])
        self.states.linear_body_velocity = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.states.angular_velocity = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])

    def cmd_vel_callback(self, msg):
        self.cmd_lin_vel_w = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.cmd_yaw_vel = msg.twist.angular.z


    def compute_control(self):
        if self.curr_pose is None or self.curr_lin_vel_w is None or self.curr_ang_vel_b is None:
            return
        ros_time = rospy.Time.now().to_sec()
        if self.last_ros_time is None:
            self.last_ros_time = ros_time - 0.01
            return
        self.dt = ros_time - self.last_ros_time
        if self.dt < 0.001:
            self.attitude_pub.publish(self.att_target)
            return

        self.set_states(
            position_w=self.states.get_position_ned_euler(),
            velocity_w=self.states.get_linear_velocity_ned_euler(),
            acceleration_w=self.states.get_linear_acceleration_ned_euler(),
            yaw=-euler_xyz_from_quat(self.states.attitude)[2]
        )

        self.set_setpoint((np.array([0.0, 0.0, 0.0])))
        # self.pc.update(self.dt)
        # att_sp: pc.VehicleAttitudeSetpoint = self.pc.get_attitude_setpoint()

        # # Placeholder for control computation logic
        # thrust = att_sp.thrust_body
        # # rate = np.array([0.0, 0.0, 0.0])  # Example rate values

        # print(f"Thrust: {thrust}")
        # att_target = AttitudeTarget()
        # att_target.header.stamp = self.latest_stamp
        # # if not np.isfinite(thrust).all():
        # #     return
        # att_target.thrust = -thrust[2]
        # print(f"rpy: {att_sp.roll_body:.2f}, {att_sp.pitch_body:.2f}, {att_sp.yaw_body:.2f}")
        # q_target = quat_from_euler_xyz(att_sp.roll_body, att_sp.pitch_body, att_sp.yaw_body)
        # q_target_frd = np.zeros(4)
        # q_target_frd[0] = q_target[1]
        # q_target_frd[1] = q_target[2]
        # q_target_frd[2] = q_target[3]
        # q_target_frd[3] = q_target[0]
        # # q_target_enu = (rot_ENU_to_NED * Rotation.from_quat(q_target_frd) * rot_FLU_to_FRD).as_quat()
        # q_target_enu = quat_from_euler_xyz(att_sp.pitch_body, att_sp.roll_body, -att_sp.yaw_body)
        # rpy = euler_xyz_from_quat(q_target_enu)
        # print(f"rpy enu: {rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f}")
        # # if not np.isfinite(q_target).all():
        # #     return
        # att_target.orientation.w = q_target_enu[3]
        # att_target.orientation.x = q_target_enu[0]
        # att_target.orientation.y = q_target_enu[1]
        # att_target.orientation.z = q_target_enu[2]
        # att_target.type_mask = AttitudeTarget.IGNORE_ROLL_RATE | AttitudeTarget.IGNORE_PITCH_RATE | AttitudeTarget.IGNORE_YAW_RATE
        
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
        if not np.isfinite(desired_thrust_per_weight):
            self.attitude_pub.publish(self.att_target)
            return
        if not np.isfinite(desired_ang_vel_b).all():
            self.attitude_pub.publish(self.att_target)
            return
        self.att_target = AttitudeTarget()
        self.att_target.header.stamp = self.latest_stamp
        self.att_target.type_mask = AttitudeTarget.IGNORE_ATTITUDE
        self.att_target.thrust = np.max([0.0, desired_thrust_per_weight]) * 1.5 / 14.6 * 0.7
        self.att_target.body_rate.x = desired_ang_vel_b[0]
        self.att_target.body_rate.y = desired_ang_vel_b[1]
        self.att_target.body_rate.z = desired_ang_vel_b[2]
        # self.att_target.body_rate.x = rate[0]
        # self.att_target.body_rate.y = rate[1]
        # self.att_target.body_rate.z = rate[2]
        # self.att_target.type_mask = AttitudeTarget.IGNORE_ATTITUDE
        print(f"Thrust: {self.att_target.thrust:.2f}")
        print(f"Rates: {self.att_target.body_rate.x:.2f}, {self.att_target.body_rate.y:.2f}, {self.att_target.body_rate.z:.2f}")
        print(f"roll_des: {(ctrl.desired_roll * 180.0 / np.pi):.2f}, pitch_des: {(ctrl.desired_pitch * 180.0 / np.pi):.2f}")
        self.attitude_pub.publish(self.att_target)

        self.last_ros_time = ros_time

    def run(self):
        rate = rospy.Rate(250)  # Hz
        while not rospy.is_shutdown():
            self.compute_control()
            rate.sleep()

if __name__ == '__main__':
    controller = ThrustAndRateControl()
    controller.run()