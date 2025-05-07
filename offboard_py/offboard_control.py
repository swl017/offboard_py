#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleGlobalPosition, VehicleLocalPosition, VehicleOdometry, VehicleStatus
from geometry_msgs.msg import TwistStamped, Vector3
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Odometry

import numpy as np
from scipy.spatial.transform import Rotation

class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('offboard_control')

        # Declare parameters with default values
        self.declare_parameters(
            namespace='',
            parameters=[
                ('vehicle_name', ''),
                ('update_rate', 100.0),
                ('target_system', 1),
                ('position.x', 0.0),
                ('position.y', 0.0),
                ('position.z', 0.0),
                ('position.yaw_deg', 0.0),
            ]
        )
        
        # Get parameters
        self.vehicle_name = self.get_parameter('vehicle_name').value
        self.update_rate = self.get_parameter('update_rate').value
        self.target_system = self.get_parameter('target_system').value
        self.waypoint = Vector3()
        self.waypoint.x = self.get_parameter('position.x').value
        self.waypoint.y = self.get_parameter('position.y').value
        self.waypoint.z = self.get_parameter('position.z').value
        self.waypoint_yaw_deg = self.get_parameter('position.yaw_deg').value
        
        # Log parameter values and indicate if using defaults
        self.get_logger().info(f'Initialized offboard control with parameters:')
        self.get_logger().info(f'  - Vehicle name: {self.vehicle_name}' + 
                             (' (default)' if self.vehicle_name == '' else ''))
        self.get_logger().info(f'  - Update rate: {self.update_rate} Hz' +
                             (' (default)' if self.update_rate == 100.0 else ''))
        self.get_logger().info(f'  - Target system: {self.target_system}' +
                             (' (default)' if self.target_system == 1 else ''))

        # Configure QoS profile for publishing and subscribing
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
        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, 'fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, 'fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, 'fmu/in/vehicle_command', qos_profile)
        self.vehicle_odom_publisher = self.create_publisher(
            Odometry, 'mavros/local_position/odom', qos_profile)
        self.vehicle_global_position_publisher = self.create_publisher(
            NavSatFix, 'mavros/global_position/global', qos_profile)
        self.vehicle_imu_publisher = self.create_publisher(
            Imu, 'mavros/imu/data', qos_profile)
        self.initial_waypoint_publisher = self.create_publisher(
            Odometry, 'initial_waypoint', qos_profile)
        
        # Create subscribers
        self.vehicle_global_position_subscriber = self.create_subscription(
            VehicleGlobalPosition, 'fmu/out/vehicle_global_position', self.vehicle_global_position_callback, qos_profile)
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, 'fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, 'fmu/out/vehicle_odometry', self.vehicle_odometry_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, 'fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        self.vehicle_cmd_vel_subscriber = self.create_subscription(
            TwistStamped, 'cmd_vel', self.vehicle_cmd_vel_callback, qos_profile_durability)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_status_counter = 0
        self.vehicle_global_position = None
        self.vehicle_local_position = None
        self.vehicle_status = None
        self.takeoff_height = -1.0
        self.vehicle_cmd_vel = TwistStamped()
        self.pass_vehicle_cmd_vel = False

        # Create a timer to publish control commands
        timer_period = 1.0 / self.update_rate  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Quaternion for rotation between ENU and NED INERTIAL frames
        # NED to ENU: +PI/2 rotation about Z (Down) followed by a +PI rotation around X (old North/new East)
        # ENU to NED: +PI/2 rotation about Z (Up) followed by a +PI rotation about X (old East/new North)
        # This rotation is symmetric, so q_ENU_to_NED == q_NED_to_ENU.
        # Note: this quaternion follows the convention [qx, qy, qz, qw]
        q_ENU_to_NED = np.array([0.70711, 0.70711, 0.0, 0.0])

        # A scipy rotation from the ENU inertial frame to the NED inertial frame of reference
        self.rot_ENU_to_NED = Rotation.from_quat(q_ENU_to_NED)

        # Quaternion for rotation between body FLU and body FRD frames
        # +PI rotation around X (Forward) axis rotates from Forward, Right, Down (aircraft)
        # to Forward, Left, Up (base_link) frames and vice-versa.
        # This rotation is symmetric, so q_FLU_to_FRD == q_FRD_to_FLU.
        # Note: this quaternion follows the convention [qx, qy, qz, qw]
        q_FLU_to_FRD = np.array([1.0, 0.0, 0.0, 0.0])

        # A scipe rotation from the FLU body frame to the FRD body frame
        self.rot_FLU_to_FRD = Rotation.from_quat(q_FLU_to_FRD)

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_global_position_callback(self, vehicle_global_position):
        """Callback function for vehicle_global_position topic subscriber."""
        self.vehicle_global_position = vehicle_global_position
        nav = NavSatFix()
        nav.header.stamp.sec = int(vehicle_global_position.timestamp * 1e-9)
        nav.header.stamp.nanosec = vehicle_global_position.timestamp - int(nav.header.stamp.sec * 1e9)
        nav.header.frame_id = 'map'
        nav.latitude = vehicle_global_position.lat
        nav.longitude = vehicle_global_position.lon
        nav.altitude = vehicle_global_position.alt
        nav.position_covariance[0] = vehicle_global_position.eph * vehicle_global_position.eph / 2
        nav.position_covariance[1] = vehicle_global_position.eph * vehicle_global_position.eph / 2
        nav.position_covariance[2] = vehicle_global_position.epv * vehicle_global_position.epv
        # self.vehicle_global_position_publisher.publish(nav)

    def vehicle_odometry_callback(self, vehicle_odometry):
        """Callback function for vehicle_odometry topic subscriber."""
        self.vehicle_odometry = vehicle_odometry
        odom = Odometry()
        odom.header.stamp.sec = int(vehicle_odometry.timestamp * 1e-9)
        odom.header.stamp.nanosec = vehicle_odometry.timestamp - int(odom.header.stamp.sec * 1e9)
        odom.header.frame_id = 'map'
        odom.child_frame_id = self.vehicle_name + '/base_link' if self.vehicle_name else 'base_link'
        # ENU from NED
        odom.pose.pose.position.x = float(vehicle_odometry.position[1])
        odom.pose.pose.position.y = float(vehicle_odometry.position[0])
        odom.pose.pose.position.z =  float(-vehicle_odometry.position[2])
        euler0 = Rotation.from_quat(vehicle_odometry.q).as_euler('xyz', degrees=False)
        euler1 = np.zeros(3)
        euler1[0] = euler0[1]
        euler1[1] = euler0[2]
        euler1[2] = euler0[0] - np.pi/2
        attitude_flu_enu = Rotation.from_euler('xyz', euler1, degrees=False).as_quat()
        odom.pose.pose.orientation.x = float(attitude_flu_enu[0]) #attitude_flu_enu[0]
        odom.pose.pose.orientation.y = float(attitude_flu_enu[1]) #attitude_flu_enu[1]
        odom.pose.pose.orientation.z = float(attitude_flu_enu[2]) #attitude_flu_enu[2]
        odom.pose.pose.orientation.w = float(attitude_flu_enu[3]) #attitude_flu_enu[3]
        odom.pose.covariance[0] = float(vehicle_odometry.position_variance[1])
        odom.pose.covariance[7] = float(vehicle_odometry.position_variance[0])
        odom.pose.covariance[14] = float(vehicle_odometry.position_variance[2])
        odom.pose.covariance[21] = float(vehicle_odometry.orientation_variance[1])
        odom.pose.covariance[28] = float(vehicle_odometry.orientation_variance[0])
        odom.pose.covariance[35] = float(vehicle_odometry.orientation_variance[2])
        odom.twist.twist.linear.x = float(vehicle_odometry.velocity[1])
        odom.twist.twist.linear.y = float(vehicle_odometry.velocity[0])
        odom.twist.twist.linear.z = float(-vehicle_odometry.velocity[2])
        odom.twist.twist.angular.x = float(vehicle_odometry.angular_velocity[1])
        odom.twist.twist.angular.y = float(vehicle_odometry.angular_velocity[0])
        odom.twist.twist.angular.z = float(-vehicle_odometry.angular_velocity[2])
        odom.twist.covariance[0] = float(vehicle_odometry.velocity_variance[1])
        odom.twist.covariance[7] = float(vehicle_odometry.velocity_variance[0])
        odom.twist.covariance[14] = float(vehicle_odometry.velocity_variance[2])
        # self.vehicle_odom_publisher.publish(odom)

        imu = Imu()
        imu.header.stamp.sec = int(vehicle_odometry.timestamp * 1e-9)
        imu.header.stamp.nanosec = vehicle_odometry.timestamp - int(imu.header.stamp.sec * 1e9)
        imu.header.frame_id = self.vehicle_name + '/base_link' if self.vehicle_name else 'base_link'
        imu.orientation.x = float(attitude_flu_enu[0])
        imu.orientation.y = float(attitude_flu_enu[1])
        imu.orientation.z = float(attitude_flu_enu[2])
        imu.orientation.w = float(attitude_flu_enu[3])
        imu.orientation_covariance[0] = float(vehicle_odometry.orientation_variance[1])
        imu.orientation_covariance[4] = float(vehicle_odometry.orientation_variance[0])
        imu.orientation_covariance[8] = float(vehicle_odometry.orientation_variance[2])
        imu.angular_velocity.x = float(vehicle_odometry.angular_velocity[1])
        imu.angular_velocity.y = float(vehicle_odometry.angular_velocity[0])
        imu.angular_velocity.z = float(-vehicle_odometry.angular_velocity[2])
        # self.vehicle_imu_publisher.publish(imu)

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status
        self.vehicle_status_counter += 1

    def vehicle_cmd_vel_callback(self, vehcile_cmd_vel):
        self.vehcile_cmd_vel = vehcile_cmd_vel
        if self.pass_vehicle_cmd_vel:
            self.publish_velocity_setpoint(self.vehcile_cmd_vel.twist.linear.x,
                                           self.vehcile_cmd_vel.twist.linear.y,
                                           self.vehcile_cmd_vel.twist.linear.z,
                                           self.vehcile_cmd_vel.twist.angular.z)

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info(f'{self.vehicle_name}: Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info(f'{self.vehicle_name}: Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info(f'{self.vehicle_name}: Switching to offboard mode')

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info(f'{self.vehicle_name}: Switching to land mode')

    def publish_offboard_control_heartbeat_signal(self, mode: str) -> None:
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 1.57079  # (90 degree)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_velocity_setpoint(self, vx_enu: float, vy_enu: float, vz_enu: float, vyaw: float):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [vy_enu, vx_enu, -vz_enu]
        msg.yawspeed = -vyaw
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = self.target_system
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def timer_callback(self) -> None:
        """Callback function for the timer."""
        self.publish_offboard_control_heartbeat_signal('velocity')

        waypoint = Odometry()
        waypoint.header.stamp.sec = int(self.get_clock().now().nanoseconds / 1e9)
        waypoint.header.stamp.nanosec = int(self.get_clock().now().nanoseconds % 1e9)
        waypoint.header.frame_id = 'common_frame'
        waypoint.child_frame_id = 'waypoint'
        waypoint.pose.pose.position.x = self.waypoint.x
        waypoint.pose.pose.position.y = self.waypoint.y
        waypoint.pose.pose.position.z = self.waypoint.z
        q = Rotation.from_euler('xyz', [0.0, 0.0, self.waypoint_yaw_deg], degrees=True).as_quat()
        waypoint.pose.pose.orientation.x = q[0]
        waypoint.pose.pose.orientation.y = q[1]
        waypoint.pose.pose.orientation.z = q[2]
        waypoint.pose.pose.orientation.w = q[3]
        self.initial_waypoint_publisher.publish(waypoint)

        # Start after FC is initialized
        if self.vehicle_local_position is None \
            or self.vehicle_global_position is None \
            or self.vehicle_status is None:
            return
        
        if self.vehicle_status.arming_state == self.vehicle_status.ARMING_STATE_STANDBY:
        #if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        if self.vehicle_local_position.z > self.takeoff_height \
            and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD \
            and not self.pass_vehicle_cmd_vel:
            # self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)
            """Publish the trajectory setpoint."""
            msg = TrajectorySetpoint()
            msg.position = [float('nan'), float('nan'), float('nan')]
            msg.velocity = [0.0, 0.0, -10.0]
            msg.yawspeed = 0.0
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_setpoint_publisher.publish(msg)

        # elif self.pass_vehicle_cmd_vel:
        #     # self.publish_offboard_control_heartbeat_signal('velocity')
        #     """Publish the trajectory setpoint."""
        #     msg = TrajectorySetpoint()
        #     msg.position = [float('nan'), float('nan'), -2.0]
        #     msg.velocity = [0.0, 0.0, 0.0]
        #     msg.yawspeed = 0.0
        #     msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        #     self.trajectory_setpoint_publisher.publish(msg)
        elif not self.pass_vehicle_cmd_vel:
            msg = TrajectorySetpoint()
            msg.position = [self.waypoint.y, self.waypoint.x, -self.waypoint.z]
            msg.yaw = -self.waypoint_yaw_deg / 180.0 * np.pi # (90 degree)
            msg.velocity = [float('nan'), float('nan'), float('nan')]
            msg.yawspeed = float('nan')
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_setpoint_publisher.publish(msg)
            dist = np.linalg.norm(self.vehicle_odometry.position - msg.position)
            # yaw_diff_deg = 
            print(f"dist = {dist}")
            euler = Rotation.from_quat(self.vehicle_odometry.q).as_euler('xyz', degrees=True)
            print(f"yaw_diff = {euler[0] - 180.0 - self.waypoint_yaw_deg}")
            if dist < 2.0 and (np.abs(euler[0] - 180.0 - self.waypoint_yaw_deg) < 10.0 or
                np.abs(euler[0] - 180.0 - self.waypoint_yaw_deg) > 350.0):
                # self.pass_vehicle_cmd_vel = True
                pass


        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
